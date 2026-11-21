"""建立唯一指向性的第二版RAG品質評估測試集。

沿用v1 eval_set.json已抽樣好的66筆文件，只重新生成問題。問題必須包含足以鎖定
單一文件的獨特細節，但不能直接出現店家／景點／住宿全名，避免退化成字串比對。

用法：python3 code/rag_quality/build_eval_set_v2.py
"""
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from ragas.cache import DiskCacheBackend

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "eval_set.json"
OUTPUT_PATH = BASE_DIR / "eval_set_v2.json"
CACHE_DIR = BASE_DIR / ".ragas_cache"
MODEL = "gpt-4o-mini"
MAX_ATTEMPTS = 3

QUESTION_PROMPT = """你是台灣旅遊RAG評估集的出題者。請根據下面這一筆資料，寫一句
自然、合理的繁體中文旅遊查詢。這句查詢必須「唯一指向」這筆資料：包含兩個以上具體
且獨特的細節，例如特殊活動、確切特色、價格、認證、歷史事實、距離、設施或特定服務，
讓熟悉資料的人只憑問題就能鎖定這一筆，而不是同縣市同類型的多筆資料都能回答。

禁止直接寫出完整店家／景點／住宿名稱「{name}」，也不要照抄一整句原文；我們要測語意
檢索，不是名稱字串比對。可以寫縣市或行政區。只回傳一句查詢，不要解釋或加引號。

資料類型：{type}
縣市：{city}{district}
資料：{reference}
"""

REWRITE_PROMPT = """上一句查詢直接包含了禁止出現的完整名稱「{name}」：
{question}

請改寫成不含完整名稱、但仍保留至少兩個足以唯一鎖定該筆資料的獨特細節。只回傳改寫後
的一句繁體中文查詢，不要解釋或加引號。
"""


def cache_key(kind: str, prompt: str) -> str:
    payload = json.dumps(
        {"kind": kind, "model": MODEL, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def complete(client: OpenAI, cache: DiskCacheBackend, kind: str, prompt: str) -> str:
    key = cache_key(kind, prompt)
    if cache.has_key(key):
        return cache.get(key)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = response.choices[0].message.content.strip().strip("\"「」")
    cache.set(key, text)
    return text


def generate_question(client: OpenAI, cache: DiskCacheBackend, item: dict) -> str:
    name = item["reference"].split("、", 1)[0].strip()
    metadata = item["metadata"]
    prompt = QUESTION_PROMPT.format(
        name=name,
        type=metadata.get("type", ""),
        city=metadata.get("city", ""),
        district=metadata.get("district", ""),
        reference=item["reference"],
    )
    question = complete(client, cache, "eval_question_v2", prompt)

    for attempt in range(1, MAX_ATTEMPTS):
        if name not in question:
            return question
        rewrite_prompt = REWRITE_PROMPT.format(name=name, question=question)
        question = complete(
            client,
            cache,
            f"eval_question_v2_rewrite_{attempt}",
            rewrite_prompt,
        )

    if name in question:
        raise ValueError(f"生成問題仍包含完整名稱：{name} -> {question}")
    return question


def main():
    load_dotenv()
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        print(f"沿用既有v2固定測試集：{OUTPUT_PATH}（{len(existing)}筆）")
        return

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    client = OpenAI()
    cache = DiskCacheBackend(cache_dir=str(CACHE_DIR))
    output = []

    print(f"沿用v1的{len(source)}筆文件，開始生成唯一指向性問題...")
    for index, item in enumerate(source, start=1):
        question = generate_question(client, cache, item)
        output.append(
            {
                "id": item["id"],
                "question": question,
                "reference": item["reference"],
                "metadata": item["metadata"],
            }
        )
        print(f"  [{index}/{len(source)}] {item['metadata'].get('city')} -> {question}")

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成，{len(output)}筆存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
