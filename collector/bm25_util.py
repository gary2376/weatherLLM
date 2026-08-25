"""共用的 BM25 lexical 索引工具：jieba 斷詞 + rank_bm25，供
collector/build_embeddings.py（建索引）和 code/hybrid_search.py（查詢）共用同一套
斷詞規則——建索引跟查詢兩邊斷詞方式不一致的話，詞頻表對不上，分數就沒有意義。
"""
import pickle
from pathlib import Path
from typing import List

import jieba
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = Path(__file__).resolve().parents[1] / "Data" / "bm25_index.pkl"


def tokenize(text: str) -> List[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def build_and_save(docs: list) -> int:
    """docs: [{"id": str, "text": str, "metadata": dict}, ...]（跟
    collector/build_embeddings.py 建 Chroma 向量索引用的是同一份 docs，
    確保 BM25 跟語意檢索是同一個候選文件集合，RRF 融合時id才能對得上）。"""
    tokenized = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "ids": [d["id"] for d in docs],
        "metadatas": [d["metadata"] for d in docs],
        "texts": [d["text"] for d in docs],
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    return len(docs)


def load() -> dict:
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)
