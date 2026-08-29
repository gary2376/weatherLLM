"""LangGraph 版節點圖，取代 `system.py::get_weather_data()` 原本一條到底的線性
函式呼叫。見 ROADMAP.md Phase 2：`intent_node → weather_node →
attraction_node/lodging_node（平行）→ synthesis_node`。

節點內部直接呼叫 `system.py` 既有函式（`analyze_prompt_with_llm`/
`fetch_radar_data`/`fetch_forecast_data`/`_get_places_for_area`/
`generate_answer_from_user_prompt_and_data`），不重寫任何天氣/RAG/LLM邏輯，只是
把原本函式內部的線性步驟拆成圖的節點——這樣未來要幫某個節點加超時/重試、或是
把 attraction/lodging 兩個外部查詢改成真的平行執行，才有地方掛。

`system` 用 local import（在 `get_weather_data()` 內部），不是在檔案最上面：
`system.py` 是這個檔案的呼叫端，檔案最上面 import 會變成循環 import。
"""
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END


class WeatherState(TypedDict, total=False):
    user_prompt: str
    area_override: Optional[str]
    city_override: Optional[str]
    api_key: Optional[str]
    info: dict
    query_type: Optional[str]
    area: Optional[str]
    city: str
    raw_data: str
    attractions: list
    lodgings: list
    final_answer: str
    error: str


def _make_intent_node(sysmod):
    def intent_node(state: WeatherState) -> WeatherState:
        try:
            info = sysmod.analyze_prompt_with_llm(state["user_prompt"], api_key=state.get("api_key"))
        except Exception as e:
            return {"error": sysmod._make_safe(f"{sysmod.ERROR_PREFIX} Prompt 分析階段出錯：{e}")}

        area = info.get("area")
        city = info.get("city") or sysmod.DEFAULT_CITY
        if state.get("area_override"):
            area = state["area_override"]
        if state.get("city_override"):
            city = state["city_override"]

        if not area:
            return {"error": sysmod._make_safe(f"{sysmod.ERROR_PREFIX} 無法辨識您想查詢的區域。")}

        return {"info": info, "query_type": info.get("type"), "area": area, "city": city}

    return intent_node


def _make_weather_node(sysmod):
    def weather_node(state: WeatherState) -> WeatherState:
        if state.get("error"):
            return {}
        info = state["info"]
        query_type = state["query_type"]
        area = state["area"]
        city = state["city"]

        if query_type == "radar":
            raw_data = sysmod.fetch_radar_data(area, city)
        elif query_type == "forecast":
            date_str = info.get("date")
            hour = info.get("hour", "06:00")
            if not date_str:
                return {"error": sysmod._make_safe(f"{sysmod.ERROR_PREFIX} 預報查詢需要有效日期。")}
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return {"error": sysmod._make_safe(f"{sysmod.ERROR_PREFIX} 日期格式錯誤 ({date_str})，應為 YYYY-MM-DD。")}

            if hour and isinstance(hour, str) and hour not in ["06:00", "18:00"]:
                try:
                    hour_int = int(hour.split(":")[0])
                    if 0 <= hour_int <= 2:
                        target_date = target_date - timedelta(days=1)
                        hour = "18:00"
                    elif 3 <= hour_int < 17:
                        hour = "06:00"
                    else:
                        hour = "18:00"
                except Exception:
                    hour = "06:00"
            elif not hour:
                hour = "06:00"

            raw_data = sysmod.fetch_forecast_data(area, city, target_date=target_date, target_hour=hour)
        else:
            return {"error": sysmod._make_safe(
                f"{sysmod.ERROR_PREFIX} 無法辨識查詢類型 (應為 'radar' 或 'forecast')。LLM 回應 type: {query_type or '未提供'}"
            )}

        if raw_data.startswith((sysmod.WARN_PREFIX, sysmod.ERROR_PREFIX)):
            return {"error": raw_data}

        try:
            radar_summary = sysmod.get_latest_radar_summary()
            raw_data = f"[RAG - 本機雷達摘要]\n{radar_summary}\n\n" + raw_data
        except Exception:
            pass
        try:
            current_weather_summary = sysmod.fetch_current_weather_open_meteo(area=area, city=city)
            raw_data = f"[RAG - 即時天氣]\n{current_weather_summary}\n\n" + raw_data
        except Exception:
            pass

        return {"raw_data": raw_data}

    return weather_node


def _make_attraction_node(sysmod):
    def attraction_node(state: WeatherState) -> WeatherState:
        if state.get("error"):
            return {}
        # `_get_places_for_area()` 只包住RAG那段的例外（RAG不可用時退回SQL查詢），
        # 退回路徑本身（`get_attractions_for_area()`）沒有防護——DB出錯之類的例外
        # 會直接往上炸穿整個graph，跟Phase2「單一node失敗要優雅退回」的系統邊界
        # 承諾不符。這裡補上，失敗就當作查無資料處理（synthesis_node本來就有
        # 「抱歉沒有景點資料」的空清單分支），不讓整個回覆掛掉。
        try:
            attractions = sysmod._get_places_for_area(
                state["user_prompt"], state["area"], state["city"], "attraction", top_n=5
            )
        except Exception:
            attractions = []
        return {"attractions": attractions}

    return attraction_node


def _make_lodging_node(sysmod):
    def lodging_node(state: WeatherState) -> WeatherState:
        if state.get("error"):
            return {}
        try:
            lodgings = sysmod._get_places_for_area(
                state["user_prompt"], state["area"], state["city"], "lodging", top_n=3
            )
        except Exception:
            lodgings = []
        return {"lodgings": lodgings}

    return lodging_node


def _make_synthesis_node(sysmod):
    def synthesis_node(state: WeatherState) -> WeatherState:
        if state.get("error"):
            return {"final_answer": state["error"]}

        raw_data = state.get("raw_data", "")
        area = state["area"]

        attractions_list = state.get("attractions") or []
        if attractions_list:
            raw_data += "\n\n🏞️ 此區推薦景點：\n"
            for idx, name in enumerate(attractions_list, start=1):
                raw_data += f"  {idx}. {name}\n"
        else:
            raw_data += f"\n\n🏞️ 抱歉，目前沒有「{area}」的景點資料可供推薦。\n"

        lodgings_list = state.get("lodgings") or []
        if lodgings_list:
            raw_data += "\n\n🏨 此區推薦住宿：\n"
            for idx, name in enumerate(lodgings_list, start=1):
                raw_data += f"  {idx}. {name}\n"
        else:
            raw_data += f"\n\n🏨 抱歉，目前沒有「{area}」的住宿資料可供推薦。\n"

        start_time = time.time()
        generated_summary = sysmod.generate_answer_from_user_prompt_and_data(
            state["user_prompt"], raw_data, api_key=state.get("api_key")
        )
        elapsed = time.time() - start_time

        llm_response_with_header = f"回應建議（回應耗時 {elapsed:.2f} 秒）：\n{generated_summary}"
        return {"final_answer": sysmod._make_safe(llm_response_with_header)}

    return synthesis_node


# 系統邊界：graph 最大步驟數上限（見 ROADMAP.md Phase 2 系統邊界項目）。這個圖
# 目前是固定5個節點的線性流程（分支只是平行，不是迴圈），15步是安全上限，不是
# 實際會用到的值——之後如果加 checkpointing 支援多輪對話，這個上限才會是真的
# 有意義的防呆。
MAX_GRAPH_STEPS = 15


def build_graph(sysmod):
    """`sysmod` 由呼叫端傳入 `system` 模組物件，不在這裡 import——避免跟
    `system.py` 之間的循環 import，也方便測試時傳入替身模組。"""
    graph = StateGraph(WeatherState)
    graph.add_node("intent_node", _make_intent_node(sysmod))
    graph.add_node("weather_node", _make_weather_node(sysmod))
    graph.add_node("attraction_node", _make_attraction_node(sysmod))
    graph.add_node("lodging_node", _make_lodging_node(sysmod))
    graph.add_node("synthesis_node", _make_synthesis_node(sysmod))

    graph.set_entry_point("intent_node")
    graph.add_edge("intent_node", "weather_node")
    # attraction/lodging 各自獨立查詢，沒有互相依賴，都接回 synthesis_node——
    # LangGraph 會自動等兩條分支都到齊才進入 synthesis_node。
    graph.add_edge("weather_node", "attraction_node")
    graph.add_edge("weather_node", "lodging_node")
    graph.add_edge("attraction_node", "synthesis_node")
    graph.add_edge("lodging_node", "synthesis_node")
    graph.add_edge("synthesis_node", END)
    return graph.compile()


_COMPILED_GRAPH_CACHE = {}


def _get_graph(sysmod):
    """跟 `orchestrator.py::get_orchestrator_graph()` 一樣快取編譯過的 graph，
    不用每次請求都重新編譯——`sysmod` 在正式環境永遠是同一個 `system` 模組物件
    （`sys.modules[__name__]`），用 `id(sysmod)` 當 key 也順便讓測試用不同的假
    sysmod 替身時各自拿到對應的 graph，不會互相污染。"""
    key = id(sysmod)
    if key not in _COMPILED_GRAPH_CACHE:
        _COMPILED_GRAPH_CACHE[key] = build_graph(sysmod)
    return _COMPILED_GRAPH_CACHE[key]


def run_weather_graph(
    sysmod,
    user_prompt: str,
    area_override: Optional[str] = None,
    city_override: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    graph = _get_graph(sysmod)
    initial_state: WeatherState = {
        "user_prompt": user_prompt,
        "area_override": area_override,
        "city_override": city_override,
        "api_key": api_key,
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": MAX_GRAPH_STEPS})
    return final_state.get("final_answer") or sysmod._make_safe(f"{sysmod.ERROR_PREFIX} graph 執行未產生回覆。")
