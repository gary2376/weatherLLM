"""量測 analyze_prompt_with_llm() 的意圖解析準確率（type/city/area/date/hour）。
見 ROADMAP.md Phase 5/Phase 6。原本20組（Phase 5先小量驗證方法可行），Phase 6
為了給fine-tuning當訓練/評估資料，擴充到260組，涵蓋全部22縣市＋radar/forecast
兩種type＋相對日期（今天/明天/後天/大後天）＋部分早晚時段＋景點名稱推論行政區——
用gpt-4o-mini，260次呼叫成本仍在幾分錢等級（不是免費，但不到需要另外編預算的
程度），不像20次那樣可以說「可忽略」，跑之前留意一下。

用法：python3 eval/run_intent_eval.py（需要 .env 有 OPENAI_API_KEY）
會呼叫真的 OpenAI API，不是deterministic測試，接的是 nightly-eval.yml（手動/排程
觸發），不進一般push/PR都跑的CI。完全正確率低於 MIN_EXACT_MATCH 門檻時
exit code非0，讓nightly workflow能真的因為準確率退步而標紅。
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

MIN_EXACT_MATCH = 0.85  # 見 ROADMAP.md Phase 6。20組時基準95%(19/20)；擴充到
# 260組後先誠實測出model真弱點是76.5%(199/260)；接著把「大後天日期算術」「冷門
# 行政區縣市歸屬」「縣轄市欄位擺錯」這三類會犯錯的地方，從「讓LLM猜」改成「查
# `district_coords`表+程式碼算」——不是fine-tuning，是把系統裡本來就有正確答案
# 的東西交給查表，最終基準衝到93.8%(244/260)。門檻抓0.85，留緩衝容忍
# temperature=0.2的隨機性，抓的是「比93.8%明顯退步」，不是要求每次都碰到93.8%


def load_cases():
    path = Path(__file__).resolve().parent / "intent_ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expected_date(case: dict) -> str:
    offset = case.get("date_offset_days")
    if offset is None:
        return None
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")


def main():
    import system

    cases = load_cases()
    field_correct = {"type": 0, "city": 0, "area": 0, "date": 0, "hour": 0}
    field_total = {"type": 0, "city": 0, "area": 0, "date": 0, "hour": 0}
    exact_match = 0
    rows = []

    for case in cases:
        try:
            info = system.analyze_prompt_with_llm(case["query"])
        except Exception as e:
            rows.append((case["query"], f"ERROR: {e}", False))
            continue

        checks = {}
        field_total["type"] += 1
        checks["type"] = info.get("type") == case["type"]
        field_total["city"] += 1
        checks["city"] = (info.get("city") or "").replace("台", "臺") == case["city"]
        field_total["area"] += 1
        checks["area"] = info.get("area") == case["area"]

        exp_date = expected_date(case)
        if exp_date:
            field_total["date"] += 1
            checks["date"] = info.get("date") == exp_date
        if case.get("hour"):
            field_total["hour"] += 1
            checks["hour"] = info.get("hour") == case["hour"]

        for k, ok in checks.items():
            if ok:
                field_correct[k] += 1

        all_ok = all(checks.values())
        if all_ok:
            exact_match += 1
        rows.append((case["query"], info, all_ok))

    print(f"=== 共 {len(cases)} 組 ===")
    for query, info, ok in rows:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {query} -> {info}")

    rate = exact_match / len(cases)
    print(f"\n完全正確（所有欄位都對）: {exact_match}/{len(cases)} = {rate:.1%}")
    print("各欄位準確率:")
    for k in ["type", "city", "area", "date", "hour"]:
        if field_total[k]:
            print(f"  {k}: {field_correct[k]}/{field_total[k]} = {field_correct[k]/field_total[k]:.1%}")

    if rate < MIN_EXACT_MATCH:
        print(f"\n完全正確率 {rate:.1%} 低於門檻 {MIN_EXACT_MATCH:.0%}，判定失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
