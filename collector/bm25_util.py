"""共用的 BM25 lexical 索引工具：jieba 斷詞 + rank_bm25，供
collector/build_embeddings.py（建索引）和 code/hybrid_search.py（查詢）共用同一套
斷詞規則——建索引跟查詢兩邊斷詞方式不一致的話，詞頻表對不上，分數就沒有意義。
"""
import pickle
import threading
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


# 系統邊界（見 ROADMAP.md Phase 4）：`hybrid_search.py::search()` 原本每次查詢
# 都呼叫這個函式，代表每次查詢都要重新從硬碟反序列化整個BM25索引（14000+筆文件
# 的斷詞結果+BM25Okapi物件）——這不是「重複查詢才浪費」，是每一次查詢都在做的
# 浪費。改成lazy singleton，跟 `code/rag.py::_get_collection()` 對Chroma client
# 的快取是同一個模式：索引檔案只在`build_embeddings.py`重建索引時才會變，
# runtime過程中不會變，process生命週期內快取一次是安全的。
_index_cache = None
_index_cache_lock = threading.Lock()


def load() -> dict:
    # Agent層併發呼叫多個工具時可能同時觸發第一次load()——見
    # code/hybrid_search.py::_get_collection()同一個坑的說明，這裡一併上鎖。
    global _index_cache
    if _index_cache is None:
        with _index_cache_lock:
            if _index_cache is None:
                with open(BM25_INDEX_PATH, "rb") as f:
                    _index_cache = pickle.load(f)
    return _index_cache


def reset_cache() -> None:
    """讓下次load()重新讀檔——同一個process內重跑build_embeddings.py後想立刻
    生效時呼叫（見code/hybrid_search.py::clear_cache()）。"""
    global _index_cache
    _index_cache = None
