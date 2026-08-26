"""Deterministic 單元測試：不打 LLM、不需要 Data/travel.db（那是gitignore掉的
真實資料，CI checkout不會有），跑得快，接進CI擋壞PR用。需要真的呼叫OpenAI的
live eval在 eval/ 底下，走 .github/workflows/nightly-eval.yml 手動/排程觸發，
不塞進這裡——見 ROADMAP.md Phase 5「eval分兩層」決策。
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))

import system  # noqa: E402
import repository  # noqa: E402
import attraction_classes  # noqa: E402
import fetch_tourism  # noqa: E402
import streamlit_app  # noqa: E402


def test_wants_indoor_detects_rain_keywords():
    assert system._wants_indoor("適合下雨天的室內景點")
    assert system._wants_indoor("如果明天下雨要去哪玩")
    assert not system._wants_indoor("台中親子景點推薦")


def test_resolve_coords_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(system, "get_district_coords", lambda city, area: None)
    assert system._resolve_coords("不存在的區", "不存在的市") == system.DEFAULT_COORDS


def test_resolve_coords_uses_district_lookup(monkeypatch):
    monkeypatch.setattr(system, "get_district_coords", lambda city, area: (24.0, 121.0))
    assert system._resolve_coords("西屯區", "臺中市") == (24.0, 121.0)


def test_classify_weather_fit_indoor_outdoor_mixed():
    assert attraction_classes.classify_weather_fit([1]) == "indoor_ok"
    assert attraction_classes.classify_weather_fit([2]) == "outdoor_only"
    assert attraction_classes.classify_weather_fit([1, 2]) == "mixed"
    assert attraction_classes.classify_weather_fit([]) == "unknown"


def test_clean_latlon_treats_zero_zero_as_missing():
    assert fetch_tourism._clean_latlon(0, 0) == (None, None)
    assert fetch_tourism._clean_latlon(24.1, 120.6) == (24.1, 120.6)


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """小型合成資料庫，不是真的travel.db——只放測query邏輯需要的少數幾筆。"""
    db_path = tmp_path / "fixture.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE attractions (
            id TEXT, name TEXT, city TEXT, district TEXT,
            latitude REAL, longitude REAL, category TEXT, description TEXT, weather_fit TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO attractions VALUES ('a1','某景點','臺中市','西屯區',24.1,120.6,'文化類','desc','indoor_ok')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(repository, "DB_PATH", db_path)
    return db_path


def test_query_attractions_normalizes_simplified_city(fixture_db):
    # 用簡體「台」查，資料庫存的是正體「臺」，要能查得到
    df = repository.query_attractions(city="台中市", district="西屯區", top_n=5)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "某景點"


def test_query_attractions_city_disambiguates_same_district_name(fixture_db):
    conn = sqlite3.connect(fixture_db)
    conn.execute(
        "INSERT INTO attractions VALUES ('a2','基隆的東西','基隆市','中山區',25.1,121.7,'文化類','desc','outdoor_only')"
    )
    conn.execute(
        "INSERT INTO attractions VALUES ('a3','台北的東西','臺北市','中山區',25.0,121.5,'文化類','desc','mixed')"
    )
    conn.commit()
    conn.close()
    df = repository.query_attractions(city="基隆市", district="中山區", top_n=5)
    assert len(df) == 1
    assert df.iloc[0]["city"] == "基隆市"


def test_get_attractions_for_area_returns_name_list(fixture_db):
    names = repository.get_attractions_for_area("西屯區", top_n=5, city="臺中市")
    assert names == ["某景點"]


def test_rate_limit_allows_requests_under_the_cap():
    now = 1000.0
    timestamps = []
    for _ in range(streamlit_app.RATE_LIMIT_MAX_REQUESTS):
        allowed, timestamps = streamlit_app._rate_limit_check(timestamps, now)
        assert allowed is True


def test_rate_limit_blocks_request_over_the_cap():
    now = 1000.0
    timestamps = []
    for _ in range(streamlit_app.RATE_LIMIT_MAX_REQUESTS):
        _, timestamps = streamlit_app._rate_limit_check(timestamps, now)
    allowed, timestamps = streamlit_app._rate_limit_check(timestamps, now)
    assert allowed is False


def test_rate_limit_forgets_requests_outside_the_window():
    now = 1000.0
    timestamps = [now - streamlit_app.RATE_LIMIT_WINDOW_SECONDS - 1] * streamlit_app.RATE_LIMIT_MAX_REQUESTS
    allowed, timestamps = streamlit_app._rate_limit_check(timestamps, now)
    assert allowed is True
    assert len(timestamps) == 1


# _is_machine_like() 原本是 generate_answer_from_user_prompt_and_data() 內部的
# nested function，沒辦法直接測——搬到 module 層級（Phase 3）之後才補得了這組測試。
def test_is_machine_like_detects_empty_or_json_only():
    assert system._is_machine_like(None)
    assert system._is_machine_like("")
    assert system._is_machine_like("[]")
    assert system._is_machine_like("[null, null]")
    assert not system._is_machine_like("今天台中天氣晴朗，適合外出")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_forecast_open_meteo_matches_target_hour(monkeypatch):
    payload = {
        "hourly": {
            "time": ["2026-08-26T06:00", "2026-08-26T18:00"],
            "temperature_2m": [24.0, 28.0],
            "precipitation": [0.0, 1.5],
            "weathercode": [1, 61],
        }
    }
    monkeypatch.setattr(system.requests, "get", lambda *a, **k: _FakeResponse(payload))
    target_date = datetime(2026, 8, 26)
    result = system.fetch_forecast_open_meteo(24.1, 120.6, target_date, target_hour="18:00")
    assert "18:00" in result
    assert "28.0" in result
