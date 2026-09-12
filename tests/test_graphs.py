"""Phase 2 LangGraph 節點圖的 deterministic 測試：不打 LLM、不需要真的
`system`/`WeatherChatbot`，用假的 sysmod/chatbot 替身驗證圖的路由與組合邏輯。
見 ROADMAP.md Phase 2。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import weather_graph  # noqa: E402
import orchestrator  # noqa: E402


def _make_fake_sysmod(**overrides):
    base = dict(
        ERROR_PREFIX="ERROR:",
        WARN_PREFIX="WARN:",
        DEFAULT_CITY="臺中市",
        _make_safe=lambda s: s,
        analyze_prompt_with_llm=lambda prompt, api_key=None: {
            "type": "forecast", "city": "臺中市", "area": "西屯區",
            "date": "2026-01-01", "hour": "06:00",
        },
        fetch_radar_data=lambda area, city: "雷達資料",
        fetch_forecast_data=lambda area, city, target_date, target_hour: "預報資料",
        get_latest_radar_summary=lambda: "雷達摘要",
        fetch_current_weather_open_meteo=lambda area=None, city=None: "即時天氣",
        _get_places_for_area=lambda user_prompt, area, city, place_type, top_n: (
            ["某景點"] if place_type == "attraction" else ["某住宿"]
        ),
        generate_answer_from_user_prompt_and_data=lambda user_prompt, raw_data, api_key=None: "這是最終建議文字",
        _score_node_failure=lambda layer, failed: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_weather_graph_happy_path_produces_final_answer():
    sysmod = _make_fake_sysmod()
    result = weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert "這是最終建議文字" in result
    assert "回應建議" in result


def test_weather_graph_merges_parallel_attraction_and_lodging_into_raw_data():
    """attraction_node/lodging_node 是圖上兩條平行分支，各自寫入 state 的
    `attractions`/`lodgings` 兩個不同 key——驗證兩邊都真的合併進最後送給
    synthesis_node 的 raw_data，不是其中一支被另一支蓋掉。"""
    captured = {}

    def spy_generate(user_prompt, raw_data, api_key=None):
        captured["raw_data"] = raw_data
        return "最終建議"

    sysmod = _make_fake_sysmod(generate_answer_from_user_prompt_and_data=spy_generate)
    weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert "某景點" in captured["raw_data"]
    assert "某住宿" in captured["raw_data"]


def test_weather_graph_attraction_node_failure_degrades_gracefully():
    """`_get_places_for_area()`的SQL退回路徑本身出錯（例如DB出問題）時，
    attraction_node/lodging_node不該把例外往上炸穿整個graph——應該當作查無資料
    處理，讓synthesis_node照樣產生回覆（帶「沒有景點資料」的訊息），不是整個
    請求失敗。"""
    def boom(user_prompt, area, city, place_type, top_n):
        raise RuntimeError("DB掛了")

    sysmod = _make_fake_sysmod(_get_places_for_area=boom)
    result = weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert "這是最終建議文字" in result


def test_weather_graph_intent_error_short_circuits_downstream_nodes():
    calls = {"weather": 0, "attraction": 0, "synthesis": 0}

    def boom(prompt, api_key=None):
        raise RuntimeError("LLM掛了")

    def track_weather(area, city, target_date, target_hour):
        calls["weather"] += 1
        return "不該被呼叫"

    def track_places(user_prompt, area, city, place_type, top_n):
        calls["attraction"] += 1
        return []

    def track_synthesis(user_prompt, raw_data):
        calls["synthesis"] += 1
        return "不該被呼叫"

    sysmod = _make_fake_sysmod(
        analyze_prompt_with_llm=boom,
        fetch_forecast_data=track_weather,
        _get_places_for_area=track_places,
        generate_answer_from_user_prompt_and_data=track_synthesis,
    )
    result = weather_graph.run_weather_graph(sysmod, "隨便問點什麼")
    assert "ERROR:" in result
    assert "LLM掛了" in result
    # intent_node 出錯後，其他節點應該直接跳過，不執行外部呼叫
    assert calls == {"weather": 0, "attraction": 0, "synthesis": 0}


def test_weather_graph_tags_per_node_failure_not_just_one_flag():
    """審查（FDE面試指南比對）發現原本只有單一rag_fallback布林值，分不出
    intent/weather/synthesis到底是哪一層出錯。驗證每個節點失敗時都標記到
    正確的layer名稱，成功時也要標記False（Langfuse失敗率分母才會對）。"""
    scores = []
    sysmod = _make_fake_sysmod(_score_node_failure=lambda layer, failed: scores.append((layer, failed)))
    weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert ("intent", False) in scores
    assert ("weather", False) in scores
    assert ("synthesis", False) in scores


def test_weather_graph_tags_intent_failure_on_llm_exception():
    scores = []

    def boom(prompt, api_key=None):
        raise RuntimeError("LLM掛了")

    sysmod = _make_fake_sysmod(
        analyze_prompt_with_llm=boom,
        _score_node_failure=lambda layer, failed: scores.append((layer, failed)),
    )
    weather_graph.run_weather_graph(sysmod, "隨便問點什麼")
    assert ("intent", True) in scores
    # intent失敗後續節點應該被跳過，不該有weather/synthesis的記錄
    assert not any(layer == "weather" for layer, _ in scores)


def test_weather_graph_tags_synthesis_failure_via_error_prefix():
    """generate_answer_from_user_prompt_and_data() 是自己接住例外回傳
    ERROR_PREFIX字串，不是往上拋例外——驗證synthesis_node用字串前綴正確判斷
    成敗，不是誤判成功。"""
    scores = []
    sysmod = _make_fake_sysmod(
        generate_answer_from_user_prompt_and_data=lambda p, r, api_key=None: "ERROR: LLM 呼叫失敗：boom",
        _score_node_failure=lambda layer, failed: scores.append((layer, failed)),
    )
    weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert ("synthesis", True) in scores


def test_weather_graph_area_override_wins_over_llm_parsed_area():
    seen = {}

    def track_forecast(area, city, target_date, target_hour):
        seen["area"] = area
        seen["city"] = city
        return "預報資料"

    sysmod = _make_fake_sysmod(fetch_forecast_data=track_forecast)
    weather_graph.run_weather_graph(sysmod, "問句", area_override="北屯區", city_override="臺中市")
    assert seen["area"] == "北屯區"
    assert seen["city"] == "臺中市"


def test_weather_graph_inherits_area_city_from_previous_info_when_area_missing():
    """「那附近有推薦餐廳嗎」這種沒有重講地點的問句，LLM解析不出area時，
    應該繼承上一輪的area/city，而不是直接進錯誤分支。"""
    sysmod = _make_fake_sysmod(
        analyze_prompt_with_llm=lambda prompt, api_key=None: {"type": "radar", "city": "臺中市"},
    )
    info_out = {}
    result = weather_graph.run_weather_graph(
        sysmod, "那附近有推薦餐廳嗎",
        previous_info={"area": "鹽埕區", "city": "高雄市", "date": "2026-01-02", "hour": "18:00"},
        info_out=info_out,
    )
    assert "這是最終建議文字" in result
    assert info_out["area"] == "鹽埕區"
    assert info_out["city"] == "高雄市"  # 不是LLM這輪預設回傳的臺中市


def test_weather_graph_does_not_inherit_stale_city_when_new_city_has_no_area():
    """回歸測試（codex審查抓到的真bug）：上一輪問臺北市大安區，這輪問「那高雄呢」
    ——LLM正確解析出city=高雄市，但沒有area。原本的邏輯只看area是否缺席就無條件
    繼承上一輪的city，會把這輪明確講的高雄市蓋掉、變成查詢「高雄市大安區」這種
    不存在的組合。修正後：city不等於DEFAULT_CITY時代表使用者這輪明確講了新縣市，
    不該繼承，維持原本「無法辨識您想查詢的區域」的錯誤。"""
    sysmod = _make_fake_sysmod(
        analyze_prompt_with_llm=lambda prompt, api_key=None: {"type": "radar", "city": "高雄市"},
    )
    result = weather_graph.run_weather_graph(
        sysmod, "那高雄呢",
        previous_info={"area": "大安區", "city": "臺北市"},
    )
    assert "ERROR:" in result
    assert "無法辨識您想查詢的區域" in result


def test_weather_graph_does_not_inherit_when_new_area_is_resolved():
    """這輪LLM有解析出新area，代表使用者明確換了地點，不該被上一輪蓋掉。"""
    sysmod = _make_fake_sysmod()  # 固定回傳 area=西屯區, city=臺中市
    info_out = {}
    weather_graph.run_weather_graph(
        sysmod, "西屯區明天天氣如何",
        previous_info={"area": "鹽埕區", "city": "高雄市"},
        info_out=info_out,
    )
    assert info_out["area"] == "西屯區"
    assert info_out["city"] == "臺中市"


def test_weather_graph_without_previous_info_still_errors_on_missing_area():
    """沒有previous_info可繼承時（例如對話的第一輪），維持原本「無法辨識區域」
    的錯誤行為，不因為加了多輪對話功能而改變單輪場景。"""
    sysmod = _make_fake_sysmod(
        analyze_prompt_with_llm=lambda prompt, api_key=None: {"type": "radar", "city": "臺中市"},
    )
    result = weather_graph.run_weather_graph(sysmod, "那附近有推薦餐廳嗎")
    assert "ERROR:" in result
    assert "無法辨識您想查詢的區域" in result


def test_weather_graph_info_out_reflects_area_override_not_raw_llm_guess():
    """override（例如UI下拉選單指定行政區）贏過LLM解析結果時，寫回`info_out`的
    也應該是override後的值，這樣下一輪繼承到的才是「這輪真正查詢的地點」。"""
    sysmod = _make_fake_sysmod()  # LLM固定猜 area=西屯區
    info_out = {}
    weather_graph.run_weather_graph(
        sysmod, "問句", area_override="北屯區", city_override="臺中市", info_out=info_out,
    )
    assert info_out["area"] == "北屯區"
    assert info_out["city"] == "臺中市"


class _FakeChatbot:
    general_weather_query_keywords = ["天氣", "氣象"]
    weather_phenomenon_keywords = {"下雨": "降雨"}

    def __init__(self, resp_text="Streamlit回應", query_type="attraction"):
        self.resp_text = resp_text
        self.query_type = query_type

    def process_message(self, user_input, override_district=None):
        return (self.resp_text, None, self.query_type)


def test_supervisor_routes_to_advanced_on_weather_keyword():
    state = {"user_input": "今天天氣如何", "chatbot": _FakeChatbot()}
    assert orchestrator.supervisor_node(state)["should_call_advanced"] is True


def test_supervisor_does_not_route_to_advanced_without_keyword():
    state = {"user_input": "推薦景點", "chatbot": _FakeChatbot()}
    assert orchestrator.supervisor_node(state)["should_call_advanced"] is False


def test_orchestrator_uses_advanced_response_when_available():
    chatbot = _FakeChatbot(resp_text="Streamlit備援回應", query_type="attraction")
    result = orchestrator.run_orchestrator(
        user_input="今天天氣如何",
        chatbot=chatbot,
        system_weather_fn=lambda prompt, override: "OpenAI進階回應",
        api_key_available=True,
    )
    assert "OpenAI進階回應" in result["final_display_text"]
    assert "Streamlit備援回應" not in result["final_display_text"]


def test_orchestrator_falls_back_to_lightweight_when_no_api_key():
    chatbot = _FakeChatbot(resp_text="建議關注的旅遊景點：某景點", query_type="attraction")
    result = orchestrator.run_orchestrator(
        user_input="今天天氣如何",
        chatbot=chatbot,
        system_weather_fn=lambda prompt, override: "不該被用到",
        api_key_available=False,
    )
    assert "建議關注的旅遊景點" in result["final_display_text"]
    assert "不該被用到" not in result["final_display_text"]
