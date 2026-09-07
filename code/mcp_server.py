"""weatherLLM MCP server：把 repository.py 的景點/住宿/餐廳/行政區座標查詢包成MCP
tool，讓外部agent（Claude Desktop/Claude Code等MCP client）能直接查這些結構化資料，
不用透過Streamlit介面。跟既有 cwa-mcp-server（見該repo `src/cwa_mcp/server.py`）
同一套 `mcp[cli]` SDK、同一種 `MCPServer`／`@mcp.tool()` 寫法，維持團隊內部MCP server
的一致慣例（見 ROADMAP.md Phase 2）。

只包 repository.py 這幾個「答案已經確定存在資料庫裡」的查詢函式——RAG語意檢索／
天氣即時查詢／LLM意圖解析不在這裡包，那些是weatherLLM app本身的核心流程（要串
Chroma/OpenAI/Open-Meteo這些外部依賴），不是給外部agent當通用查詢工具用的。

用法：
    python3 code/mcp_server.py          # 直接跑（stdio transport）
    uv run mcp dev code/mcp_server.py   # 本地用 inspector 手動測試 tools（若有裝uv）
"""
import math
import sqlite3
from typing import Optional

import pandas as pd
from mcp.server.mcpserver import MCPServer

import repository

mcp = MCPServer("weatherllm-mcp-server")


def _clean(value):
    """DataFrame轉dict後，NaN是Python float('nan')——json.dumps()預設會原樣輸出
    字面NaN token，不是合法JSON（RFC 8259不允許NaN/Infinity），嚴格的JSON parser
    或`json.dumps(allow_nan=False)`會直接報錯。座標缺失（少數景點/餐廳沒有
    lat/lon）本來就該表示成JSON null（「這個值不存在」），不是NaN。"""
    return None if isinstance(value, float) and math.isnan(value) else value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


@mcp.resource("weatherllm://cities")
def cities() -> list[str]:
    """全台已收錄資料的縣市清單（district_coords表裡出現過的縣市，共22縣市）。"""
    with sqlite3.connect(repository.DB_PATH) as conn:
        rows = conn.execute("SELECT DISTINCT city FROM district_coords ORDER BY city")
        return [r[0] for r in rows]


@mcp.tool()
def get_attractions(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """查詢景點清單，含名稱/座標/分類/描述/天氣適合度
    （weather_fit：indoor_ok/outdoor_only/mixed/unknown）。

    city 用官方縣市全名（如「臺中市」，「台」會自動正規化成「臺」）；不給city會查全台
    （`repository.query_attractions()`本身就支援city=None不加篩選條件，這裡的參數
    簽章要跟它一致，不能比它更嚴格）。
    """
    return _records(repository.query_attractions(city=city, district=district, top_n=top_n))


@mcp.tool()
def get_lodgings(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 3) -> list[dict]:
    """查詢住宿清單，含名稱/座標/合法登記字號/描述/價格區間。不給city會查全台。"""
    return _records(repository.query_lodgings(city=city, district=district, top_n=top_n))


@mcp.tool()
def get_restaurants(city: Optional[str] = None, district: Optional[str] = None, top_n: int = 5) -> list[dict]:
    """查詢餐廳清單，含名稱/座標/料理分類/描述/營業時間。不給city會查全台。"""
    return _records(repository.query_restaurants(city=city, district=district, top_n=top_n))


@mcp.tool()
def get_district_coordinates(city: str, district: str) -> Optional[dict]:
    """查詢指定縣市行政區的中心座標（從景點/住宿/餐廳資料的座標平均推算，見
    collector/derive_district_coords.py）。查無該行政區時退回同縣市其他區的平均座標，
    連縣市都查無資料則回傳 None。"""
    coords = repository.get_district_coords(city, district)
    if coords is None:
        return None
    lat, lon = coords
    return {"latitude": lat, "longitude": lon}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
