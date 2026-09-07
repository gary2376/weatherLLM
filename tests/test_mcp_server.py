"""Phase 2 MCP tool server的deterministic測試：monkeypatch掉`repository.py`的查詢
函式，不需要真的`Data/travel.db`（gitignore掉的真實資料，CI checkout不會有，跟
`tests/`其他測試同一個約束）。見 ROADMAP.md Phase 2。

`mcp`套件要求Python>=3.10，CI（`.github/workflows/ci.yml`）用3.11所以裝得到、這個
檔案的測試會真的跑；如果本機開發環境是3.9（例如這個repo原本的`.venv`），
`import mcp`會直接`ModuleNotFoundError`——`pytest.importorskip`讓這種情況下整個
檔案優雅跳過而不是讓collection error中斷整個`pytest tests/`（一個檔案import失敗
不該把其他跟mcp無關的測試也一起擋下來）。
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mcp")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import mcp_server  # noqa: E402
import repository  # noqa: E402


def _fake_attractions_df():
    return pd.DataFrame([
        {"id": "A1", "name": "某公園", "city": "臺中市", "district": "西屯區",
         "latitude": 24.18, "longitude": 120.64, "category": "公園綠地類",
         "description": "測試用描述", "weather_fit": "indoor_ok"},
    ])


def test_get_attractions_returns_records_from_repository(monkeypatch):
    captured = {}

    def fake_query_attractions(city=None, district=None, top_n=5):
        captured.update(city=city, district=district, top_n=top_n)
        return _fake_attractions_df()

    monkeypatch.setattr(repository, "query_attractions", fake_query_attractions)
    result = mcp_server.get_attractions(city="臺中市", district="西屯區", top_n=3)

    assert captured == {"city": "臺中市", "district": "西屯區", "top_n": 3}
    assert result == [{"id": "A1", "name": "某公園", "city": "臺中市", "district": "西屯區",
                        "latitude": 24.18, "longitude": 120.64, "category": "公園綠地類",
                        "description": "測試用描述", "weather_fit": "indoor_ok"}]


def test_get_attractions_converts_nan_coordinates_to_none(monkeypatch):
    """回歸測試（codex複查抓到的真bug）：真實`Data/travel.db`裡有24筆景點/2筆餐廳
    座標是NULL，`pandas.read_sql_query()`把NULL的REAL欄位讀成`float('nan')`，
    `.to_dict(orient="records")`直接轉出來的是Python的NaN——`json.dumps()`預設會
    印出字面`NaN` token，不是合法JSON（RFC 8259不允許），嚴格的JSON parser或
    `json.dumps(allow_nan=False)`會直接報錯。座標缺失該表示成JSON null，不是NaN。"""
    def fake_query_attractions(city=None, district=None, top_n=5):
        return pd.DataFrame([
            {"id": "A2", "name": "座標缺失景點", "city": "臺南市", "district": "中西區",
             "latitude": float("nan"), "longitude": float("nan"), "category": "文化類",
             "description": "測試", "weather_fit": "unknown"},
        ])

    monkeypatch.setattr(repository, "query_attractions", fake_query_attractions)
    result = mcp_server.get_attractions(city="臺南市")

    assert result[0]["latitude"] is None
    assert result[0]["longitude"] is None
    import json
    json.dumps(result, allow_nan=False)  # 不該丟例外


def test_get_attractions_city_is_optional_and_defaults_to_all_taiwan(monkeypatch):
    """回歸測試：`repository.query_attractions()`本身`city`參數就是`Optional[str]=None`
    （不給city就查全台，不加WHERE city篩選），第一版`mcp_server.py`把tool的簽章寫成
    `city: str`（必填），比底層函式更嚴格，等於MCP工具悄悄縮窄了repository.py原本
    支援的能力——外部agent沒辦法透過這個tool問「全台最推薦的景點」這種跨縣市查詢。"""
    captured = {}

    def fake_query_attractions(city=None, district=None, top_n=5):
        captured.update(city=city, district=district, top_n=top_n)
        return _fake_attractions_df()

    monkeypatch.setattr(repository, "query_attractions", fake_query_attractions)
    mcp_server.get_attractions(top_n=3)  # 完全不傳city，不應該報TypeError

    assert captured["city"] is None


def test_get_lodgings_returns_records_from_repository(monkeypatch):
    def fake_query_lodgings(city=None, district=None, top_n=3):
        return pd.DataFrame([{"id": "L1", "name": "某旅館", "city": city, "district": district,
                               "latitude": 24.0, "longitude": 120.0, "license_no": "L123",
                               "description": "測試", "price_range": "1000-2000"}])

    monkeypatch.setattr(repository, "query_lodgings", fake_query_lodgings)
    result = mcp_server.get_lodgings(city="高雄市", district="三民區")

    assert len(result) == 1
    assert result[0]["name"] == "某旅館"
    assert result[0]["city"] == "高雄市"


def test_get_restaurants_returns_records_from_repository(monkeypatch):
    def fake_query_restaurants(city=None, district=None, top_n=5):
        return pd.DataFrame([{"id": "R1", "name": "某餐廳", "city": city, "district": district,
                               "latitude": 24.0, "longitude": 120.0, "category": "台灣小吃/台菜",
                               "description": "測試", "open_hours": "11:00-21:00"}])

    monkeypatch.setattr(repository, "query_restaurants", fake_query_restaurants)
    result = mcp_server.get_restaurants(city="臺南市")

    assert len(result) == 1
    assert result[0]["name"] == "某餐廳"


def test_get_district_coordinates_returns_dict_when_found(monkeypatch):
    monkeypatch.setattr(repository, "get_district_coords", lambda city, district: (24.18, 120.64))
    result = mcp_server.get_district_coordinates(city="臺中市", district="西屯區")
    assert result == {"latitude": 24.18, "longitude": 120.64}


def test_get_district_coordinates_returns_none_when_not_found(monkeypatch):
    monkeypatch.setattr(repository, "get_district_coords", lambda city, district: None)
    result = mcp_server.get_district_coordinates(city="不存在市", district="不存在區")
    assert result is None


def test_cities_resource_reads_distinct_cities_from_district_coords(tmp_path, monkeypatch):
    db_path = tmp_path / "fake_travel.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE district_coords (city TEXT, district TEXT, latitude REAL, longitude REAL)")
    conn.executemany(
        "INSERT INTO district_coords VALUES (?, ?, ?, ?)",
        [("臺中市", "西屯區", 24.18, 120.64), ("高雄市", "三民區", 22.65, 120.30),
         ("臺中市", "北區", 24.15, 120.68)],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(repository, "DB_PATH", db_path)
    result = mcp_server.cities()

    assert result == ["臺中市", "高雄市"]  # 去重＋排序


def test_all_tools_and_resource_are_registered_on_mcp_server():
    """驗證的是`@mcp.tool()`/`@mcp.resource()`裝飾器真的有把函式註冊進MCP server
    物件本身（不是只確認函式本身能被直接呼叫）——用`mcp.list_tools()`這個MCP
    SDK自己的內省API去查，這樣才是真的驗證外部MCP client連上來看得到這些工具。"""
    async def _list():
        tools = await mcp_server.mcp.list_tools()
        resources = await mcp_server.mcp.list_resources()
        return tools, resources

    tools, resources = asyncio.run(_list())
    tool_names = {t.name for t in tools}
    assert tool_names == {"get_attractions", "get_lodgings", "get_restaurants", "get_district_coordinates"}
    assert str(resources[0].uri) == "weatherllm://cities"


def test_call_tool_through_mcp_interface_not_just_direct_python_call(monkeypatch):
    """跟上面那組測試的差別：這裡不直接呼叫`mcp_server.get_attractions(...)`這個
    Python函式，是走`mcp.call_tool()`——這才是外部MCP client實際會用的路徑
    （client端不會、也不能直接import這個模組呼叫Python函式，只能透過MCP協定的
    call_tool），驗證這條路徑本身也是通的，不是只有底層函式邏輯是對的。"""
    monkeypatch.setattr(repository, "query_attractions", lambda city=None, district=None, top_n=5: _fake_attractions_df())

    async def _call():
        return await mcp_server.mcp.call_tool("get_attractions", {"city": "臺中市", "district": "西屯區", "top_n": 1})

    result = asyncio.run(_call())
    assert result.structured_content["result"][0]["name"] == "某公園"
