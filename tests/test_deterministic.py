"""Deterministic 單元測試：不打 LLM、不需要 Data/travel.db（那是gitignore掉的
真實資料，CI checkout不會有），跑得快，接進CI擋壞PR用。需要真的呼叫OpenAI的
live eval在 eval/ 底下，走 .github/workflows/nightly-eval.yml 手動/排程觸發，
不塞進這裡——見 ROADMAP.md Phase 5「eval分兩層」決策。
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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


@pytest.fixture
def fixture_district_coords_db(tmp_path, monkeypatch):
    """小型合成district_coords表，測`get_district_coords()`的查詢/備援邏輯，
    不是真的travel.db。"""
    db_path = tmp_path / "fixture_coords.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE district_coords (city TEXT, district TEXT, latitude REAL, longitude REAL, sample_count INTEGER)"
    )
    conn.execute("INSERT INTO district_coords VALUES ('彰化縣','彰化市',24.08,120.54,10)")
    conn.execute("INSERT INTO district_coords VALUES ('彰化縣','鹿港鎮',24.06,120.43,10)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(repository, "DB_PATH", db_path)
    return db_path


def test_get_district_coords_recovers_from_concatenated_city_name(fixture_district_coords_db):
    """LLM有時候會把「彰化市」誤判成「彰化縣彰化市」——這種黏在一起的字串精確查詢
    會落空，也不該直接退回預設座標，應該從裡面找出真實存在的縣市名稱重查一次。"""
    coords = repository.get_district_coords("彰化縣彰化市", "彰化市")
    assert coords == (24.08, 120.54)


def test_get_district_coords_returns_none_when_city_truly_unrecognized(fixture_district_coords_db):
    assert repository.get_district_coords("不存在的縣市", "不存在的區") is None


def test_get_all_city_districts_groups_by_city(fixture_district_coords_db):
    assert repository.get_all_city_districts() == {"彰化縣": ["彰化市", "鹿港鎮"]}


def test_resolve_unique_city_for_district_returns_unique_match(fixture_district_coords_db):
    assert repository.resolve_unique_city_for_district("鹿港鎮") == "彰化縣"


def test_resolve_unique_city_for_district_none_when_ambiguous_or_unknown(fixture_district_coords_db):
    conn = sqlite3.connect(fixture_district_coords_db)
    conn.execute("INSERT INTO district_coords VALUES ('臺北市','中山區',25.06,121.52,10)")
    conn.execute("INSERT INTO district_coords VALUES ('基隆市','中山區',25.13,121.74,10)")
    conn.commit()
    conn.close()
    assert repository.resolve_unique_city_for_district("中山區") is None
    assert repository.resolve_unique_city_for_district("不存在的區") is None


class _FakeChatResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def test_analyze_prompt_computes_relative_date_in_code(monkeypatch):
    """大後天這種相對日期，Phase 6 eval發現LLM自己算常常算錯（+2天當成+3天）——
    改成LLM只回傳relative_day_offset分類，實際日期用timedelta在程式碼裡算，不
    信任LLM自己的日期算術。"""
    fake_json = json.dumps({
        "type": "forecast", "city": "臺中市", "area": "西屯區", "relative_day_offset": 3,
    })
    monkeypatch.setattr(system, "_create_chat_completion", lambda *a, **k: _FakeChatResponse(fake_json))
    info = system.analyze_prompt_with_llm("大後天西屯區天氣如何")
    expected_date = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    assert info["date"] == expected_date
    assert "relative_day_offset" not in info


def test_analyze_prompt_corrects_city_for_unique_district(monkeypatch, fixture_district_coords_db):
    """LLM把只存在單一縣市的冷門行政區猜成別的縣市時，用查表結果校正，不信任
    LLM對冷門鄉鎮的世界知識。"""
    fake_json = json.dumps({"type": "radar", "city": "臺中市", "area": "鹿港鎮"})
    monkeypatch.setattr(system, "_create_chat_completion", lambda *a, **k: _FakeChatResponse(fake_json))
    info = system.analyze_prompt_with_llm("鹿港鎮現在天氣如何")
    assert info["city"] == "彰化縣"


def test_analyze_prompt_moves_county_administered_city_from_city_to_area(monkeypatch, fixture_district_coords_db):
    """LLM有時候把「彰化市」這種縣轄市整個塞進city欄位、area留空（沒拆成
    city=彰化縣/area=彰化市）——city本身如果剛好是全台唯一對應一個上級縣的行政區
    名稱，就該當作area處理。"""
    fake_json = json.dumps({"type": "radar", "city": "彰化市"})
    monkeypatch.setattr(system, "_create_chat_completion", lambda *a, **k: _FakeChatResponse(fake_json))
    info = system.analyze_prompt_with_llm("彰化市的天氣")
    assert info["city"] == "彰化縣"
    assert info["area"] == "彰化市"


def test_analyze_prompt_does_not_override_ambiguous_district_city(monkeypatch, fixture_district_coords_db):
    """同名跨縣市的行政區（例如中山區）查表結果不唯一，保留LLM自己的判斷，不亂
    覆寫——LLM可能從問句上下文看出真正的縣市。"""
    conn = sqlite3.connect(fixture_district_coords_db)
    conn.execute("INSERT INTO district_coords VALUES ('臺北市','中山區',25.06,121.52,10)")
    conn.execute("INSERT INTO district_coords VALUES ('基隆市','中山區',25.13,121.74,10)")
    conn.commit()
    conn.close()
    fake_json = json.dumps({"type": "radar", "city": "基隆市", "area": "中山區"})
    monkeypatch.setattr(system, "_create_chat_completion", lambda *a, **k: _FakeChatResponse(fake_json))
    info = system.analyze_prompt_with_llm("中山區現在天氣如何")
    assert info["city"] == "基隆市"


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


def test_bridge_secrets_to_env_fills_missing_keys_from_secrets():
    """部署環境（例如Streamlit Community Cloud）沒有`.env`檔案時，這三把key
    要能從`st.secrets`橋接進`os.environ`，下游`system.py`的`os.getenv(...)`
    才讀得到值。"""
    fake_secrets = SimpleNamespace(get=lambda k, default="": {
        "OPENAI_API_KEY": "sk-from-secrets", "LANGFUSE_PUBLIC_KEY": "pub-from-secrets",
    }.get(k, default))
    environ = {}
    streamlit_app._bridge_secrets_to_env(fake_secrets, environ)
    assert environ["OPENAI_API_KEY"] == "sk-from-secrets"
    assert environ["LANGFUSE_PUBLIC_KEY"] == "pub-from-secrets"
    assert environ["LANGFUSE_SECRET_KEY"] == ""  # secrets沒有這把，補空字串不是報錯


def test_bridge_secrets_to_env_does_not_override_existing_env_value():
    """本機開發`.env`已經有值時（透過`load_dotenv()`載入到`os.environ`），
    `st.secrets`橋接不能蓋掉它——`setdefault`是這個保證的核心，不是巧合。"""
    fake_secrets = SimpleNamespace(get=lambda k, default="": "sk-from-secrets-should-not-win")
    environ = {"OPENAI_API_KEY": "sk-from-dotenv"}
    streamlit_app._bridge_secrets_to_env(fake_secrets, environ)
    assert environ["OPENAI_API_KEY"] == "sk-from-dotenv"


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


def test_weather_chatbot_resolves_zhubei_to_hsinchu_county_not_city():
    """審查發現的真bug：city_districts_map把「竹北」歸類在「新竹」分組底下，
    CITY_NAME_MAP把整個「新竹」分組都對應到新竹市——但竹北市其實屬於新竹縣，
    查詢會查到錯的縣市。已用DISTRICT_CITY_OVERRIDE校正，這裡驗證修好了。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            captured["district"] = district
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message("竹北有什麼景點")
    assert captured["city"] == "新竹縣"


def test_weather_chatbot_bare_hsinchu_still_resolves_to_hsinchu_city():
    """確認修正沒有連帶弄壞「新竹」本身（沒指定竹北/湖口/竹東時）預設查新竹市的
    行為。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message("新竹有什麼景點")
    assert captured["city"] == "新竹市"


@pytest.mark.parametrize("query,expected_city", [
    ("台北大安區有什麼景點", "臺北市"),  # 大安區台中/台北都有，原本沒看訊息裡的縣市，一律先比對到台中的
    ("台南北區有什麼景點", "臺南市"),    # 北區台中/台南都有
    ("台南東區有什麼景點", "臺南市"),    # 東區台中/台南都有
])
def test_weather_chatbot_explicit_city_wins_over_same_name_district_in_other_city(query, expected_city):
    """codex複查時抓到的真bug：city_districts_map裡有幾個同名跨縣市的行政區
    （大安區/北區/東區），原本比對邏輯不管訊息裡有沒有明確提到縣市，一律照
    dict順序（台中最先）找到就用——「台北大安區」會被誤判成台中的大安區。已修：
    訊息裡有明確提到某縣市時，該縣市的行政區清單優先比對。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message(query)
    assert captured["city"] == expected_city


def test_weather_chatbot_ambiguous_district_without_city_still_defaults_to_taichung():
    """沒有明確縣市、行政區名稱本身又同名跨縣市時（例如單獨問「北區」），維持
    原本的預設行為（退回台中），不強制猜城市——這種情況本來就無法從文字判斷，
    跟審查時codex確認過的「已知限制」一致，不是這次要修的範圍。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message("北區有什麼景點")
    assert captured["city"] == "臺中市"


@pytest.mark.parametrize("query,expected_city", [
    ("南投縣的景點跟住宿推薦", "南投縣"),
    ("南投有什麼景點", "南投縣"),
    ("屏東的美食", "屏東縣"),
    ("花蓮有什麼住宿", "花蓮縣"),
    ("澎湖有什麼景點", "澎湖縣"),
    ("雲林有什麼美食", "雲林縣"),
    ("金門有什麼景點", "金門縣"),
    ("連江縣的景點", "連江縣"),
])
def test_weather_chatbot_covers_all_22_counties_not_just_original_5(query, expected_city):
    """線上使用者實測發現：原本city_districts_map只手key了台中/台北/新竹/台南/
    高雄5個縣市，資料庫明明有全台22縣市（見DATA_LICENSE.md／README），問南投
    這類沒被涵蓋到的縣市會被誤導成搜尋台中。改成從repository.get_all_city_districts()
    動態產生對照表，這裡驗證全部22縣市都真的搜得到，不是只有原本5個。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message(query)
    assert captured["city"] == expected_city


@pytest.mark.parametrize("query,expected_city", [
    ("嘉義縣的景點", "嘉義縣"),
    ("嘉義市的景點", "嘉義市"),
    ("嘉義有什麼景點", "嘉義市"),  # 沒指定市/縣時維持「先猜市」的既有慣例
    ("新竹縣的景點", "新竹縣"),
    ("新竹市的景點", "新竹市"),
])
def test_weather_chatbot_explicit_county_suffix_disambiguates_city_vs_county(query, expected_city):
    """新竹/嘉義都是市縣同名（去掉市/縣尾字後撞名）——原本只有靠行政區名稱（例如
    「竹北」）才能校正到縣，使用者直接打「嘉義縣」這種明確字尾反而還是被導去
    預設的嘉義市。這裡驗證明確打出「市」/「縣」字尾時能正確對應。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message(query)
    assert captured["city"] == expected_city


def test_weather_chatbot_nonexistent_taichung_county_does_not_redirect_to_bogus_city():
    """台中縣2010年就併入臺中市了，現實中不存在——防禦性測試：確保
    `city_suffix_variants`只收錄DB裡真的存在的字尾（台中只有「市」），不會因為
    程式碼一律組出`alias+「縣」`字串去比對訊息，就把使用者打錯的「台中縣」
    誤導去查一個不存在的縣市名稱，而是正常退回臺中市。"""
    captured = {}

    class _SpyAttractionManager:
        def get_attractions(self, city=None, district=None, attraction_type=None, num_attractions=5):
            captured["city"] = city
            return pd.DataFrame()

    chatbot = streamlit_app.WeatherChatbot(
        cctv_manager=SimpleNamespace(cctv_data=pd.DataFrame(columns=["name", "geohash"])),
        attraction_manager=_SpyAttractionManager(),
    )
    chatbot.process_message("台中縣的景點")
    assert captured["city"] == "臺中市"
