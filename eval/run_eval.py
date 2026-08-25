"""RAG 檢索品質評估：Hit Rate@k / MRR，對照 eval/ground_truth.json 的24組
「問句→正確id」人工標註（每題only有一個正解id，來自實際 travel.db 資料，不是編的）。

為什麼不是原本 ROADMAP 規劃的 RAGAS：RAGAS 的 context precision/recall 是
LLM-judge 分數，要打LLM API（沒設定 OPENAI_API_KEY，而且這正是改用本地 embedding
模型想避免的持續成本）。我們的 ground truth 是「明確id」而不是「golden answer
text」，直接看id有沒有出現在top-k、出現在第幾名，就是標準、免費、確定性的檢索
評估指標，比硬套一個需要LLM的框架更合適——見 ROADMAP.md Phase 1 決策調整。

用法：
  python3 eval/run_eval.py semantic   # 對照 code/rag.py（純語意檢索）
  python3 eval/run_eval.py hybrid     # 對照 code/hybrid_search.py（BM25+語意 RRF融合）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

TOP_K = 5


def load_ground_truth():
    path = Path(__file__).resolve().parent / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(search_fn, cases, top_k=TOP_K):
    hits = 0
    reciprocal_ranks = []
    rows = []
    for case in cases:
        results = search_fn(
            case["query"],
            city=case.get("city"),
            district=case.get("district"),
            type_=case.get("type"),
            top_k=top_k,
        )
        ids = [r["id"] for r in results]
        # 索引裡的id都是「type:原始id」（見 collector/build_embeddings.py），
        # ground_truth.json 存的是原始id，這裡補回同樣的prefix才能比對
        expected = f"{case['type']}:{case['expected_id']}"
        if expected in ids:
            rank = ids.index(expected) + 1
            hits += 1
            reciprocal_ranks.append(1 / rank)
            rows.append((case["query"], case["note"], "PASS", rank))
        else:
            reciprocal_ranks.append(0.0)
            rows.append((case["query"], case["note"], "FAIL", None))
    n = len(cases)
    hit_rate = hits / n
    mrr = sum(reciprocal_ranks) / n
    return hit_rate, mrr, rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "semantic"
    if mode == "semantic":
        import rag
        search_fn = rag.search
    elif mode == "hybrid":
        import hybrid_search
        search_fn = hybrid_search.search
    else:
        print(f"未知模式：{mode}（用 semantic 或 hybrid）")
        sys.exit(1)

    cases = load_ground_truth()
    hit_rate, mrr, rows = evaluate(search_fn, cases)

    print(f"=== 模式：{mode}，共 {len(cases)} 組，top_k={TOP_K} ===")
    for query, note, status, rank in rows:
        rank_str = f"第{rank}名" if rank else "-"
        print(f"  [{status}] {query}（正解：{note}）{rank_str}")
    print(f"\nHit Rate@{TOP_K}: {hit_rate:.2%}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
