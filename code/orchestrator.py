"""LangGraph 版 supervisor + worker 組合，取代 `streamlit_app.py::main()` 裡原本
`should_call_system_py` 關鍵字硬判斷、以及呼叫完兩套系統後手動 if/else 拼接回覆
文字的寫法。見 ROADMAP.md Phase 2。

刻意只把「決策＋資料處理」這些不碰 Streamlit runtime 的部分搬進圖：
`st.spinner`/`st.warning`/`st.session_state` 這些 UI 專屬呼叫，以及 API key
是否可用的判斷（牽涉寫入 `system_module.openai.api_key`/`os.environ` 這種有副作用
的動作），都留在 `main()` 呼叫端——這樣這個圖本身不依賴 Streamlit，可以直接用
pytest 測路由跟組合邏輯，不用開瀏覽器。

兩個 worker：
- lightweight_node：對應原本「Streamlit 內部機器人處理」那段，包成 `WeatherChatbot`
  worker（地圖景點圖釘資料仍由 `WeatherChatbot.process_message()` 產生，這個圖
  不重寫那段邏輯）。
- advanced_node：對應原本「呼叫 system.py」那段，包成 OpenAI 進階建議 worker
  （實際天氣/RAG/LLM 邏輯仍在 `system.py` 的 `get_weather_data()`，這個圖只決定
  「要不要呼叫、呼叫完怎麼處理回傳」）。
"""
from typing import Callable, List, Optional, TypedDict

from langgraph.graph import StateGraph, END


class OrchestratorState(TypedDict, total=False):
    user_input: str
    chatbot: object  # WeatherChatbot instance
    override_district: Optional[str]
    api_key_available: bool
    system_weather_fn: Callable[[str, Optional[str]], str]

    sl_resp_text: str
    sl_attr_df: object
    sl_query_type: Optional[str]

    should_call_advanced: bool
    system_py_main_response: str
    final_display_text: str


def _is_machine_format(content) -> bool:
    """跟原本 `streamlit_app.py` 裡的同名函式邏輯完全一致，搬過來給
    `advanced_node`/`combine_node` 共用。"""
    try:
        if content is None:
            return True
        if not isinstance(content, str):
            return True
        s = content.strip()
        if not s:
            return True
        low = s.lower()
        if low.startswith('[') or low.startswith('{'):
            if 'null' in low or 'none' in low or ',' in s:
                return True
            return True
        if s.startswith('[') and ']' in s:
            return True
        if s.count(',') >= 2 and all(ord(ch) < 128 for ch in s if ch.isalpha()):
            return True
    except Exception:
        return True
    return False


def _sanitize_system_response(content) -> str:
    """跟原本 `streamlit_app.py` 裡的同名函式邏輯完全一致，避免把 raw JSON/NULL
    直接顯示給使用者。"""
    fallback = "抱歉，進階建議系統暫時無法提供可讀回覆，請稍後或調整查詢內容。"
    try:
        if content is None:
            return fallback
        if not isinstance(content, str):
            try:
                import json
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return fallback
        if _is_machine_format(content):
            try:
                import json
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str) and item.strip():
                            return item
                    return fallback
                if isinstance(parsed, dict):
                    for k in ('text', 'content', 'message'):
                        if k in parsed and isinstance(parsed[k], str):
                            return parsed[k]
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                s = content.replace('[', '').replace(']', '').replace('null', '').replace('None', '').strip()
                if s:
                    return s
                return fallback
        return content
    except Exception:
        return fallback


def lightweight_node(state: OrchestratorState) -> OrchestratorState:
    chatbot = state["chatbot"]
    sl_resp_text, sl_attr_df, sl_query_type = chatbot.process_message(
        state["user_input"], state.get("override_district")
    )
    return {
        "sl_resp_text": sl_resp_text,
        "sl_attr_df": sl_attr_df,
        "sl_query_type": sl_query_type,
    }


def supervisor_node(state: OrchestratorState) -> OrchestratorState:
    """路由決策，邏輯沿用 `WeatherChatbot` 身上已有的關鍵字表（不重新發明一套），
    只是從 `main()` 裡的行內判斷搬成獨立可測的節點。"""
    chatbot = state["chatbot"]
    user_input = state["user_input"]
    should_call = False
    if any(kw in user_input for kw in getattr(chatbot, "general_weather_query_keywords", [])):
        should_call = True
    if not should_call and any(
        kw in user_input.lower() for kw in getattr(chatbot, "weather_phenomenon_keywords", {}).keys()
    ):
        should_call = True
    return {"should_call_advanced": should_call}


def advanced_node(state: OrchestratorState) -> OrchestratorState:
    if not state.get("should_call_advanced") or not state.get("api_key_available"):
        return {"system_py_main_response": ""}
    system_weather_fn = state["system_weather_fn"]
    try:
        response = system_weather_fn(state["user_input"], state.get("override_district"))
    except Exception as e:
        response = f"抱歉，OpenAI 進階建議系統暫時無法服務。（{e}）"
    return {"system_py_main_response": _sanitize_system_response(response)}


def combine_node(state: OrchestratorState) -> OrchestratorState:
    system_py_main_response = state.get("system_py_main_response", "")
    sl_query_type = state.get("sl_query_type")

    final_response_elements: List[str] = []

    if system_py_main_response:
        final_response_elements.append(system_py_main_response)

    if not system_py_main_response or "錯誤：" in system_py_main_response or "無法導入" in system_py_main_response:
        safe_sl_resp = _sanitize_system_response(state.get("sl_resp_text")) if state.get("sl_resp_text") is not None else None
        if sl_query_type in ["attraction", "restaurant", "hotel"]:
            if safe_sl_resp and ("建議關注的旅遊景點：" in safe_sl_resp or "為您找到以下位於" in safe_sl_resp):
                if not any("建議關注的旅遊景點：" in str(p) for p in final_response_elements) and \
                   not any("為您找到以下位於" in str(p) for p in final_response_elements):
                    final_response_elements.append(safe_sl_resp)
        else:
            if safe_sl_resp and not final_response_elements:
                final_response_elements.append(safe_sl_resp)

    if not final_response_elements:
        final_response_elements.append("抱歉，目前無法處理您的請求，請再試一次或調整您的問題。")

    final_display_text = "\n".join(filter(None, final_response_elements)).strip()
    if not final_display_text:
        final_display_text = "抱歉，進階建議系統暫時無法提供內容，請稍後或調整查詢。"

    return {"final_display_text": final_display_text}


def build_orchestrator_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("lightweight_node", lightweight_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("advanced_node", advanced_node)
    graph.add_node("combine_node", combine_node)

    graph.set_entry_point("lightweight_node")
    graph.add_edge("lightweight_node", "supervisor_node")
    graph.add_edge("supervisor_node", "advanced_node")
    graph.add_edge("advanced_node", "combine_node")
    graph.add_edge("combine_node", END)
    return graph.compile()


_ORCH_GRAPH = None


def get_orchestrator_graph():
    global _ORCH_GRAPH
    if _ORCH_GRAPH is None:
        _ORCH_GRAPH = build_orchestrator_graph()
    return _ORCH_GRAPH


# 系統邊界：graph 最大步驟數上限（見 ROADMAP.md Phase 2 系統邊界項目）。目前是
# 4個節點的線性流程，20步是安全上限，不是實際會用到的值。
MAX_ORCHESTRATOR_STEPS = 20


def run_orchestrator(
    user_input: str,
    chatbot,
    system_weather_fn: Callable[[str, Optional[str]], str],
    override_district: Optional[str] = None,
    api_key_available: bool = False,
) -> dict:
    graph = get_orchestrator_graph()
    initial_state: OrchestratorState = {
        "user_input": user_input,
        "chatbot": chatbot,
        "override_district": override_district,
        "api_key_available": api_key_available,
        "system_weather_fn": system_weather_fn,
    }
    return graph.invoke(initial_state, config={"recursion_limit": MAX_ORCHESTRATOR_STEPS})
