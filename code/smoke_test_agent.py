"""Agent層多輪對話+guardrail驗收腳本。不是pytest單元測試，累積失敗項目、用
sys.exit(1)回報，避免像早期RAG smoke test犯過的「印✗但exit code還是0」問題。

跑法：python3 code/smoke_test_agent.py
"""
import asyncio
import sys
import uuid

import agent as agent_module
from agent import WeatherAgent

MULTI_TURN_CASES = [
    "南投日月潭有什麼景點",
    "附近有推薦餐廳嗎",
    "有素食的嗎",
]

# 判斷指代消解有沒有成功：第2、3輪的回答內容裡，要能看到跟「日月潭/南投」相關的
# 地名字樣，而不是答非所問(例如跑去講其他縣市)。這是關鍵字啟發式檢查，不是嚴謹NLU
# 驗證，但足夠當回歸測試的最低標準。
CONTEXT_KEYWORDS = ["日月潭", "南投", "魚池"]


async def run_multi_turn_test() -> list:
    failures = []
    thread_id = f"smoke-test-{uuid.uuid4()}"
    print("=== 多輪對話測試(指代消解) ===")
    async with WeatherAgent() as agent:
        for i, user_input in enumerate(MULTI_TURN_CASES, start=1):
            reply, _doc_ids = await agent.ask(thread_id, user_input)
            print(f"\n第{i}輪 使用者：{user_input}")
            print(f"第{i}輪 Agent：{reply}")
            if i >= 2:
                hit = any(kw in reply for kw in CONTEXT_KEYWORDS)
                print(f"  指代消解關鍵字檢查：{'✓' if hit else '✗'}")
                if not hit:
                    failures.append(f"第{i}輪回答沒有出現{CONTEXT_KEYWORDS}任一關鍵字，可能指代消解失敗：{reply[:100]}")
        print(f"\n本次對話累積花費估算：US${agent.cost_so_far(thread_id):.4f}")
    return failures


async def run_multi_need_test() -> list:
    """單輪就要agent同時處理多個需求(景點/住宿/餐廳)——這種query通常會讓LLM一次
    發出多個工具呼叫，LangGraph會併發執行，2026-09-30就是靠這個case才抓到
    hybrid_search.py/bm25_util.py的singleton初始化沒上鎖的競態bug(併發時
    Chroma client初始化壞掉，工具報錯後agent還瞎掰答案)。不能只靠手動測一次，
    這裡固定成回歸測試。"""
    failures = []
    print("\n=== 多需求單輪query測試(併發工具呼叫) ===")
    thread_id = f"smoke-test-multineed-{uuid.uuid4()}"
    async with WeatherAgent() as agent:
        reply, doc_ids = await agent.ask(thread_id, "我想去南投日月潭玩，順便找住宿跟推薦餐廳")
    print(f"回答：\n{reply}")
    hit = any(kw in reply for kw in CONTEXT_KEYWORDS)
    if not hit:
        failures.append(f"多需求query回答沒有出現{CONTEXT_KEYWORDS}任一關鍵字：{reply[:100]}")
    # 資料庫裡真實存在的record關鍵字(見docs/rag_quality/eval_set.json同批語料)，
    # 用來粗略檢查答案是不是真的查到資料庫內容，不是LLM自己編的——不是嚴謹檢查，
    # 但比完全不檢查好。
    grounded = any(kw in reply for kw in ["民宿", "餐廳", "飯店", "酒店"])
    if not grounded:
        failures.append(f"多需求query回答裡沒看到住宿/餐廳類關鍵字，可能查詢失敗或答非所問：{reply[:200]}")
    if not doc_ids:
        failures.append("多需求query沒有回傳任何RAG文件id")
    if hit and grounded and doc_ids:
        print(f"  ✓ 多需求query有正確回應且回傳 {len(doc_ids)} 筆RAG文件id")
    return failures


async def run_weather_test() -> list:
    """真實走planner → weather tool → final，避免只測MCP cache卻漏掉agent routing。"""
    failures = []
    print("\n=== 天氣工具routing測試(真實CWA API) ===")
    thread_id = f"smoke-test-weather-{uuid.uuid4()}"
    async with WeatherAgent() as agent:
        reply, _doc_ids = await agent.ask(thread_id, "台北市現在有沒有生效中的天氣特報？")
    print(f"回答：\n{reply}")
    if not any(keyword in reply for keyword in ["台北", "臺北", "特報"]):
        failures.append(f"天氣query回答不像是weather tool結果：{reply[:200]}")
    else:
        print("  ✓ agent有完成weather tool routing並組成文字回答")
    return failures


async def run_guardrail_test() -> list:
    failures = []
    print("\n=== Guardrail觸發測試 ===")
    thread_id = f"smoke-test-guardrail-{uuid.uuid4()}"
    # 直接把這個thread的累積花費設成超過門檻，不用真的燒錢燒到US$0.08才能測——
    # 這樣測的是guardrail判斷邏輯本身，不是重新驗證token計價公式。
    agent_module._thread_costs[thread_id] = agent_module.COST_THRESHOLD_USD + 1.0
    async with WeatherAgent() as agent:
        reply, _doc_ids = await agent.ask(thread_id, "隨便問一句話測guardrail")
    print(f"guardrail觸發後的回應：{reply}")
    if "安全上限" not in reply:
        failures.append(f"guardrail應該擋下並回傳安全上限訊息，但實際回應是：{reply}")
    else:
        print("  ✓ guardrail正確擋下，沒有繼續呼叫工具/LLM")
    return failures


async def main() -> int:
    failures = await run_multi_turn_test()
    failures += await run_multi_need_test()
    failures += await run_weather_test()
    failures += await run_guardrail_test()

    if failures:
        print(f"\n{'=' * 40}\n{len(failures)}項未通過：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n全部通過")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
