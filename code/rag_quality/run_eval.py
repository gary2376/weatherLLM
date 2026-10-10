"""跑`eval_set.json`：對每筆問題真的呼叫`code/rag.py::query()`做檢索，用
gpt-4o-mini基於檢索到的context生成一個簡短回答(這是eval專用的簡易生成，不是
完整agent)，最後用ragas的Faithfulness/ContextRecall/AnswerRelevancy三個指標
打分。

全程用`DiskCacheBackend`快取LLM呼叫——同一份eval_set沒改的話，重跑不用再花錢。

用法：python3 code/rag_quality/run_eval.py [輸出檔名，預設帶時間戳]
      python3 code/rag_quality/run_eval.py --eval-set eval_set_v2.json [輸出檔名]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
import rag  # noqa: E402

from ragas.cache import DiskCacheBackend
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = "eval_set.json"
CACHE_DIR = BASE_DIR / ".ragas_cache"
MODEL = "gpt-4o-mini"

ANSWER_PROMPT = """根據下面的參考資料，用繁體中文簡短回答使用者的問題(2-3句話即可)，
只根據參考資料回答，不要編造資料裡沒有的資訊。

參考資料：
{context}

問題：{question}
"""


def generate_answer(
    client: OpenAI, cache: DiskCacheBackend, question: str, contexts: list
) -> str:
    prompt = ANSWER_PROMPT.format(context="\n".join(contexts), question=question)
    cache_payload = json.dumps(
        {"kind": "eval_answer", "model": MODEL, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    if cache.has_key(cache_key):
        return cache.get(cache_key)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    cache.set(cache_key, answer)
    return answer


def parse_args():
    parser = argparse.ArgumentParser(description="執行RAG品質評估")
    parser.add_argument(
        "--eval-set",
        default=DEFAULT_EVAL_SET,
        help="Eval set路徑；相對路徑以code/rag_quality為基準",
    )
    parser.add_argument("output", nargs="?", help="輸出檔名；預設使用當日日期")
    return parser.parse_args()


def resolve_eval_set(path_arg: str) -> Path:
    path = Path(path_arg)
    return path if path.is_absolute() else BASE_DIR / path


def main():
    args = parse_args()
    load_dotenv()
    eval_set_path = resolve_eval_set(args.eval_set)
    eval_set = json.loads(eval_set_path.read_text(encoding="utf-8"))

    client = OpenAI()
    ragas_client = AsyncOpenAI()
    cache = DiskCacheBackend(cache_dir=str(CACHE_DIR))
    ragas_llm = llm_factory(MODEL, client=ragas_client, cache=cache, max_tokens=4096)
    ragas_embeddings = OpenAIEmbeddings(client=ragas_client, cache=cache)

    faithfulness = Faithfulness(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    results = []
    for i, item in enumerate(eval_set):
        question = item["question"]
        docs = rag.query(question, top_k=5)
        contexts = [d["text"] for d in docs]

        if not contexts:
            print(f"  [{i + 1}/{len(eval_set)}] 「{question}」檢索不到任何文件，跳過")
            results.append({"id": item["id"], "question": question, "error": "no_context"})
            continue

        answer = generate_answer(client, cache, question, contexts)
        f_score = faithfulness.score(user_input=question, response=answer, retrieved_contexts=contexts)
        r_score = context_recall.score(
            user_input=question,
            retrieved_contexts=contexts,
            reference=item["reference"],
        )
        a_score = answer_relevancy.score(user_input=question, response=answer)

        row = {
            "id": item["id"],
            "question": question,
            "answer": answer,
            "faithfulness": f_score.value,
            "context_recall": r_score.value,
            "answer_relevancy": a_score.value,
        }
        results.append(row)
        print(
            f"  [{i + 1}/{len(eval_set)}] faithfulness={row['faithfulness']:.2f} "
            f"context_recall={row['context_recall']:.2f} answer_relevancy={row['answer_relevancy']:.2f}"
        )

    valid = [r for r in results if "error" not in r]
    summary = {
        "n_total": len(eval_set),
        "n_valid": len(valid),
        "n_no_context": len(eval_set) - len(valid),
        "faithfulness_mean": mean(r["faithfulness"] for r in valid) if valid else None,
        "context_recall_mean": mean(r["context_recall"] for r in valid) if valid else None,
        "answer_relevancy_mean": mean(r["answer_relevancy"] for r in valid) if valid else None,
    }

    default_name = f"results_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    out_name = args.output or default_name
    out_path = BASE_DIR / out_name
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== 總結 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n完整結果存到 {out_path}")


if __name__ == "__main__":
    main()
