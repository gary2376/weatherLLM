"""weather MCP TTL cache的輕量驗收——不需要CWA_API_KEY，用假的session計數驗證
cache行為，pytest可直接discover。"""
import asyncio
import time
from types import SimpleNamespace

import mcp_tools
import pytest


class CountingSession:
    def __init__(self):
        self.call_count = 0

    async def call_tool(self, name, kwargs):
        self.call_count += 1
        # 留一個排程點，讓併發驗收確實有機會撞到同一個cache miss。
        await asyncio.sleep(0.01)
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"response-{self.call_count}:{name}:{kwargs}")]
        )


class FailingSession:
    def __init__(self):
        self.call_count = 0

    async def call_tool(self, name, kwargs):
        self.call_count += 1
        await asyncio.sleep(0.01)
        raise TimeoutError("forced CWA timeout")


class ErrorResultSession:
    def __init__(self):
        self.call_count = 0

    async def call_tool(self, name, kwargs):
        self.call_count += 1
        await asyncio.sleep(0.01)
        return SimpleNamespace(
            isError=True,
            content=[SimpleNamespace(text="bad response from CWA")],
        )


def _make_tool(session):
    weather = mcp_tools.WeatherMCPTools("test-key")
    weather._session = session
    return weather._build_tool(
        "get_weather_forecast",
        "test tool",
        {
            "type": "object",
            "properties": {"county": {"type": "string"}},
            "required": ["county"],
        },
    )


def test_partial_mcp_entry_failure_closes_exit_stack(monkeypatch):
    class FakeExitStack:
        def __init__(self):
            self.enter_count = 0
            self.closed = False

        async def enter_async_context(self, _context_manager):
            self.enter_count += 1
            if self.enter_count == 1:
                return "read", "write"
            raise RuntimeError("forced session startup failure")

        async def aclose(self):
            self.closed = True

    stack = FakeExitStack()
    weather = mcp_tools.WeatherMCPTools("test-key")
    weather._stack = stack
    monkeypatch.setattr(mcp_tools, "stdio_client", lambda _params: object())
    monkeypatch.setattr(mcp_tools, "ClientSession", lambda *_args: object())

    with pytest.raises(RuntimeError, match="forced session startup failure"):
        asyncio.run(weather.__aenter__())

    assert stack.closed


def test_cache_hit_on_same_key():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = CountingSession()
        tool = _make_tool(session)
        first = await tool.ainvoke({"county": "臺北市"})
        second = await tool.ainvoke({"county": "臺北市"})
        assert session.call_count == 1
        assert first == second

    asyncio.run(run())


def test_expired_key_refetches():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = CountingSession()
        tool = _make_tool(session)
        first = await tool.ainvoke({"county": "臺北市"})

        key = mcp_tools._weather_cache_key("get_weather_forecast", {"county": "臺北市"})
        with mcp_tools._weather_cache_lock:
            value, _ = mcp_tools._weather_cache[key]
            mcp_tools._weather_cache[key] = (
                value,
                time.monotonic() - mcp_tools.WEATHER_CACHE_TTL_SECONDS - 1,
            )
        expired = await tool.ainvoke({"county": "臺北市"})
        assert session.call_count == 2
        assert expired != first

    asyncio.run(run())


def test_different_args_are_independent():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = CountingSession()
        tool = _make_tool(session)
        taipei = await tool.ainvoke({"county": "臺北市"})
        kaohsiung = await tool.ainvoke({"county": "高雄市"})
        taipei_again = await tool.ainvoke({"county": "臺北市"})
        assert session.call_count == 2  # 高雄一次、臺北快取命中
        assert taipei != kaohsiung
        assert taipei == taipei_again

    asyncio.run(run())


def test_concurrent_same_key_miss_dedupes_to_one_call():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = CountingSession()
        tool = _make_tool(session)
        results = await asyncio.gather(
            tool.ainvoke({"county": "南投縣"}),
            tool.ainvoke({"county": "南投縣"}),
        )
        assert session.call_count == 1
        assert results[0] == results[1]

    asyncio.run(run())


def test_concurrent_same_key_failure_reaches_every_waiter_without_hanging():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = FailingSession()
        tool = _make_tool(session)
        results = await asyncio.wait_for(
            asyncio.gather(
                tool.ainvoke({"county": "南投縣"}),
                tool.ainvoke({"county": "南投縣"}),
                return_exceptions=True,
            ),
            timeout=1.0,
        )
        assert session.call_count == 1
        assert len(results) == 2
        assert all(isinstance(result, TimeoutError) for result in results)
        assert all(str(result) == "forced CWA timeout" for result in results)
        with mcp_tools._weather_cache_lock:
            assert not mcp_tools._weather_cache_inflight
            assert not mcp_tools._weather_cache

    asyncio.run(run())


def test_mcp_error_result_is_not_cached_or_returned_as_weather_data():
    async def run():
        with mcp_tools._weather_cache_lock:
            mcp_tools._weather_cache.clear()
            mcp_tools._weather_cache_inflight.clear()
        session = ErrorResultSession()
        tool = _make_tool(session)
        results = await asyncio.wait_for(
            asyncio.gather(
                tool.ainvoke({"county": "臺北市"}),
                tool.ainvoke({"county": "臺北市"}),
                return_exceptions=True,
            ),
            timeout=1.0,
        )
        assert session.call_count == 1
        assert all(isinstance(result, RuntimeError) for result in results)
        assert all("bad response from CWA" in str(result) for result in results)
        with mcp_tools._weather_cache_lock:
            assert not mcp_tools._weather_cache_inflight
            assert not mcp_tools._weather_cache

    asyncio.run(run())


if __name__ == "__main__":
    test_cache_hit_on_same_key()
    test_expired_key_refetches()
    test_different_args_are_independent()
    test_concurrent_same_key_miss_dedupes_to_one_call()
    test_concurrent_same_key_failure_reaches_every_waiter_without_hanging()
    test_mcp_error_result_is_not_cached_or_returned_as_weather_data()
    print("PASS: weather MCP TTL cache checks all passed")
