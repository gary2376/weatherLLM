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
        analyze_prompt_with_llm=lambda prompt: {
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
        generate_answer_from_user_prompt_and_data=lambda user_prompt, raw_data: "這是最終建議文字",
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

    def spy_generate(user_prompt, raw_data):
        captured["raw_data"] = raw_data
        return "最終建議"

    sysmod = _make_fake_sysmod(generate_answer_from_user_prompt_and_data=spy_generate)
    weather_graph.run_weather_graph(sysmod, "西屯區明天天氣如何")
    assert "某景點" in captured["raw_data"]
    assert "某住宿" in captured["raw_data"]


def test_weather_graph_intent_error_short_circuits_downstream_nodes():
    calls = {"weather": 0, "attraction": 0, "synthesis": 0}

    def boom(prompt):
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


class _FakeChatbot:
    general_weather_query_keywords = ["天氣", "氣象"]
    weather_phenomenon_keywords = {"下雨": "降雨"}

    def __init__(self, resp_text="Streamlit回應", cctv_names=None, query_type="attraction"):
        self.resp_text = resp_text
        self.cctv_names = cctv_names or []
        self.query_type = query_type

    def process_message(self, user_input, override_district=None):
        return (self.resp_text, self.cctv_names, None, self.query_type)


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


def test_orchestrator_appends_cctv_legend_when_present():
    chatbot = _FakeChatbot(resp_text="回應", cctv_names=["某CCTV"], query_type=None)
    result = orchestrator.run_orchestrator(
        user_input="推薦景點",
        chatbot=chatbot,
        system_weather_fn=lambda prompt, override: "",
        api_key_available=False,
    )
    assert "建議關注的監視器位置" in result["final_display_text"]
    assert "某CCTV" in result["final_display_text"]
