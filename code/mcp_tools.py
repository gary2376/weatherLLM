"""把cwa-mcp-server的四個天氣工具(天氣預報/地震/特報/颱風)包成LangChain
`StructuredTool`，透過stdio連線。

不用`langchain-mcp-adapters`：這個函式庫新舊版都跟我們的環境有版本衝突(舊版要的
mcp SDK API跟新版mcp套件對不上、新版要的langchain-core跟ragas對不上)，詳見
docs/ARCHITECTURE.md「環境」段落的決策記錄。我們只需要固定的4個工具，不需要通用
MCP client框架，直接用`mcp` SDK本身寫stdio連線邏輯就好。

用法：
    async with WeatherMCPTools(cwa_api_key) as weather:
        tools = weather.tools  # List[StructuredTool]，掛進agent的工具清單
"""
import asyncio
import json
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import create_model

# cwa-mcp-server已經pip安裝進這個專案的venv(見docs/ARCHITECTURE.md「環境」段落)，
# 用venv自己的console script路徑當command，不额外依賴uvx。
CWA_MCP_SERVER_COMMAND = str(Path(sys.exec_prefix) / "bin" / "cwa-mcp-server")

# CWA預報以小時計更新，不需要每次工具呼叫都重打API。取架構文件規劃的
# 30-60分鐘中位數；集中成常數，之後可依實際更新頻率調整。
WEATHER_CACHE_TTL_SECONDS = 45 * 60

# LangGraph可能在同一輪併發呼叫多個工具。cache與in-flight表的每次讀寫都必須
# 受同一把threading.Lock保護；in-flight Future也確保相同key同時miss時只有一個
# coroutine真的呼叫MCP，其餘共用該次結果。Lock不能跨await持有，否則會卡住event loop。
_weather_cache: Dict[tuple[str, str], tuple[str, float]] = {}
_weather_cache_inflight: Dict[tuple[str, str], Future[str]] = {}
_weather_cache_lock = threading.Lock()

_JSON_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def _weather_cache_key(tool_name: str, kwargs: Dict[str, Any]) -> tuple[str, str]:
    """MCP工具參數本來就必須是JSON可序列化值；排序後讓參數順序不影響key。"""
    serialized_args = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return tool_name, serialized_args


def _schema_to_pydantic(tool_name: str, schema: Optional[Dict[str, Any]]):
    """cwa-mcp-server的4個工具參數都是簡單的字串/整數(county/limit)，不需要
    處理巢狀物件，這個轉換只覆蓋這個範圍，不是通用JSON Schema轉換器。"""
    schema = schema or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_TYPE_MAP.get(prop_schema.get("type"), str)
        default = ... if prop_name in required else prop_schema.get("default")
        fields[prop_name] = (py_type, default)
    return create_model(f"{tool_name}_Args", **fields)


class WeatherMCPTools:
    """管理跟cwa-mcp-server的stdio連線生命週期，包出LangChain tools清單。
    整個agent對話過程共用同一條連線(進入一次async context，對話結束才離開)，
    不用每次呼叫工具都重新啟動子process。"""

    def __init__(self, cwa_api_key: str):
        self._cwa_api_key = cwa_api_key
        self._stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self.tools: List[StructuredTool] = []

    async def __aenter__(self) -> "WeatherMCPTools":
        try:
            params = StdioServerParameters(
                command=CWA_MCP_SERVER_COMMAND,
                args=[],
                env={"CWA_API_KEY": self._cwa_api_key},
            )
            read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._session = session

            listed = await session.list_tools()
            for tool in listed.tools:
                self.tools.append(self._build_tool(tool.name, tool.description, tool.input_schema))
            return self
        except BaseException:
            # A failed __aenter__ is not followed by __aexit__; close any stdio
            # process/session that was already registered with the exit stack.
            await self._stack.aclose()
            raise

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    def _build_tool(self, name: str, description: Optional[str], input_schema: Optional[dict]) -> StructuredTool:
        args_model = _schema_to_pydantic(name, input_schema)

        async def _acall(**kwargs) -> str:
            assert self._session is not None, "WeatherMCPTools必須用`async with`進入才能呼叫工具"
            cache_key = _weather_cache_key(name, kwargs)
            now = time.monotonic()

            with _weather_cache_lock:
                cached = _weather_cache.get(cache_key)
                if cached is not None and now - cached[1] < WEATHER_CACHE_TTL_SECONDS:
                    return cached[0]

                # Double-check與登記in-flight工作必須在同一個critical section，否則
                # 兩個併發miss仍可能各自打一次CWA API。
                pending = _weather_cache_inflight.get(cache_key)
                if pending is None:
                    pending = Future()
                    _weather_cache_inflight[cache_key] = pending
                    owns_request = True
                else:
                    owns_request = False

            if not owns_request:
                return await asyncio.wrap_future(pending)

            try:
                result = await self._session.call_tool(name, kwargs)
                parts = []
                for item in result.content:
                    text = getattr(item, "text", None)
                    parts.append(text if text is not None else str(item))
                value = "\n".join(parts) if parts else "(無回傳內容)"
                if getattr(result, "isError", False):
                    raise RuntimeError(f"CWA MCP工具{name}回傳錯誤：{value}")
            except BaseException as exc:
                with _weather_cache_lock:
                    _weather_cache_inflight.pop(cache_key, None)
                pending.set_exception(exc)
                # owner也從同一個Future取回例外，避免沒有waiter時留下
                # "Future exception was never retrieved"警告。
                return await asyncio.wrap_future(pending)

            with _weather_cache_lock:
                _weather_cache[cache_key] = (value, time.monotonic())
                _weather_cache_inflight.pop(cache_key, None)
            pending.set_result(value)
            return await asyncio.wrap_future(pending)

        return StructuredTool.from_function(
            coroutine=_acall,
            name=name,
            description=description or name,
            args_schema=args_model,
        )
