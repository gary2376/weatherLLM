"""建立RAG品質評估測試集(一次性)：從已建好的BM25索引語料(跟RAG查詢用的是同一份
docs)依city分層抽樣(每個縣市最多PER_CITY筆，涵蓋全部22縣市，資料稀疏的縣市自然
抽比較少)，每筆文件用gpt-4o-mini生成一句「遊客可能會問的自然語言查詢」，該文件
原文當ContextRecall要用的reference(ground truth)。

存成固定的eval_set.json，之後baseline/比較用的都是同一份，不重新抽樣——不然每次
比較基準都不一樣，比較沒意義。

沒有用ragas.testset.TestsetGenerator：那是ragas 0.4.3的知識圖譜pipeline，對這種
結構簡單、已有metadata的短文件語料來說偏重，這裡直接寫針對性的簡化版，效果一樣
(自動生成問題+已知ground truth)但可控、好debug。

用法：python3 code/rag_quality/build_eval_set.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from ragas.cache import DiskCacheBackend

COLLECTOR_DIR = Path(__file__).resolve().parents[2] / "collector"
sys.path.insert(0, str(COLLECTOR_DIR))
import bm25_util  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "eval_set.json"
CACHE_DIR = Path(__file__).resolve().parent / ".ragas_cache"
PER_CITY = 3
SEED = 42
MODEL = "gpt-4o-mini"

QUESTION_PROMPT = """你是台灣旅遊網站的使用者。根據下面這筆景點/住宿/餐廳資料，
寫一句你會在旅遊查詢欄輸入的自然語言查詢，用來找「這一類」地方(不用完全等於這
一筆，但這筆資料應該要能回答這個查詢)。不要直接照抄資料裡的句子，不用一定要包含
店家全名，可以包含縣市/類型/特色關鍵字。只回傳查詢句本身，不要其他說明文字。

資料類型：{type}
縣市：{city}{district}
內容：{text}
"""


def sample_docs():
    idx = bm25_util.load()
    by_city = {}
    for doc_id, text, metadata in zip(idx["ids"], idx["texts"], idx["metadatas"]):
        city = metadata.get("city") or "未知"
        by_city.setdefault(city, []).append({"id": doc_id, "text": text, "metadata": metadata})

    rng = random.Random(SEED)
    sampled = []
    for city in sorted(by_city):
        docs = by_city[city]
        rng.shuffle(docs)
        sampled.extend(docs[:PER_CITY])
    return sampled


def generate_question(client: OpenAI, cache: DiskCacheBackend, doc: dict) -> str:
    meta = doc["metadata"]
    prompt = QUESTION_PROMPT.format(
        type=meta.get("type", ""),
        city=meta.get("city", ""),
        district=meta.get("district", ""),
        text=doc["text"],
    )
    cache_payload = json.dumps(
        {"kind": "eval_question", "model": MODEL, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    if cache.has_key(cache_key):
        return cache.get(cache_key)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    question = resp.choices[0].message.content.strip()
    cache.set(cache_key, question)
    return question


def main():
    load_dotenv()
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        print(f"沿用既有固定測試集：{OUTPUT_PATH}（{len(existing)}筆）")
        return

    client = OpenAI()
    cache = DiskCacheBackend(cache_dir=str(CACHE_DIR))
    docs = sample_docs()
    print(f"抽樣到{len(docs)}筆文件，開始生成問題...")

    eval_set = []
    for i, doc in enumerate(docs):
        question = generate_question(client, cache, doc)
        eval_set.append(
            {
                "id": doc["id"],
                "question": question,
                "reference": doc["text"],
                "metadata": doc["metadata"],
            }
        )
        print(f"  [{i + 1}/{len(docs)}] {doc['metadata'].get('city')} -> {question}")

    OUTPUT_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成，{len(eval_set)}筆存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
