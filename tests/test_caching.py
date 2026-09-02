"""Phase 4 快取層的 deterministic 測試：驗證重複查詢真的命中快取、TTL 真的會
過期、bm25索引真的只從硬碟載入一次——不需要真的 Data/chroma/Data/bm25_index.pkl
（monkeypatch 掉實際連線/載入索引的那一層），零成本。見 ROADMAP.md Phase 4。
"""
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))

import system  # noqa: E402
import rag  # noqa: E402
import hybrid_search  # noqa: E402
import bm25_util  # noqa: E402


# ---------- system.py 的天氣 TTL 快取 ----------

def test_weather_cache_returns_cached_value_within_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(system.time, "time", lambda: now[0])
    system._weather_cache.clear()
    system._weather_cache_set(("test", 1.0, 2.0), "cached value")
    assert system._weather_cache_get(("test", 1.0, 2.0)) == "cached value"


def test_weather_cache_expires_after_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(system.time, "time", lambda: now[0])
    system._weather_cache.clear()
    system._weather_cache_set(("test", 1.0, 2.0), "cached value")
    now[0] += system.WEATHER_CACHE_TTL_SECONDS + 1
    assert system._weather_cache_get(("test", 1.0, 2.0)) is None


def test_weather_cache_miss_returns_none():
    system._weather_cache.clear()
    assert system._weather_cache_get(("從沒存過的key", 0, 0)) is None


def test_weather_cache_get_on_already_removed_expired_key_does_not_raise(monkeypatch):
    """回歸測試（codex審查抓到的真bug）：`_weather_cache`是跨session共用的
    module-level dict，兩個並發請求同時讀到同一把過期的key時，先到的那個
    刪掉key後，後到的那個原本用`del`會拿KeyError。這裡直接模擬「key已經被
    別的請求先清掉」的狀態，驗證第二次呼叫不會炸掉。"""
    now = [1000.0]
    monkeypatch.setattr(system.time, "time", lambda: now[0])
    system._weather_cache.clear()
    key = ("test", 1.0, 2.0)
    system._weather_cache_set(key, "cached value")
    now[0] += system.WEATHER_CACHE_TTL_SECONDS + 1
    del system._weather_cache[key]  # 模擬另一個並發請求已經搶先清掉了
    assert system._weather_cache_get(key) is None


def test_fetch_current_weather_uses_cache_on_second_call(monkeypatch):
    """驗證的是`fetch_current_weather_open_meteo()`本身有沒有真的走快取，
    不是只測快取工具函式本身。"""
    calls = {"n": 0}

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            calls["n"] += 1
            return {"current_weather": {"temperature": 25, "windspeed": 3, "weathercode": 1, "time": "2026-01-01T00:00"}}

    system._weather_cache.clear()
    monkeypatch.setattr(system.requests, "get", lambda *a, **k: _FakeResp())
    r1 = system.fetch_current_weather_open_meteo(lat=24.1, lon=120.6)
    r2 = system.fetch_current_weather_open_meteo(lat=24.1, lon=120.6)
    assert r1 == r2
    assert calls["n"] == 1  # 第二次應該命中快取，不會真的再打一次 Open-Meteo


# ---------- rag.py / hybrid_search.py 的查詢結果快取 ----------

def test_rag_search_caches_repeated_identical_queries(monkeypatch):
    calls = {"n": 0}

    class _FakeCollection:
        def query(self, query_texts, n_results, where):
            calls["n"] += 1
            return {"ids": [["a1"]], "documents": [["某景點描述"]], "metadatas": [[{"city": "臺中市"}]], "distances": [[0.1]]}

    monkeypatch.setattr(rag, "_get_collection", lambda: _FakeCollection())
    rag._search_cached.cache_clear()

    rag.search("親子景點", city="臺中市", top_k=3)
    rag.search("親子景點", city="臺中市", top_k=3)
    assert calls["n"] == 1


def test_rag_search_weather_fit_list_is_hashable_and_still_cached(monkeypatch):
    """weather_fit傳list時（例如["indoor_ok","mixed"]）本身不可雜湊，
    search()要負責轉成可快取的形式，不能讓lru_cache直接爆TypeError。"""
    calls = {"n": 0}

    class _FakeCollection:
        def query(self, query_texts, n_results, where):
            calls["n"] += 1
            assert where == {"weather_fit": {"$in": ["indoor_ok", "mixed"]}}
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(rag, "_get_collection", lambda: _FakeCollection())
    rag._search_cached.cache_clear()

    rag.search("室內景點", weather_fit=["indoor_ok", "mixed"])
    rag.search("室內景點", weather_fit=["indoor_ok", "mixed"])
    assert calls["n"] == 1


def test_hybrid_search_caches_repeated_identical_queries(monkeypatch):
    calls = {"bm25_load": 0, "rag_search": 0}

    def fake_load():
        calls["bm25_load"] += 1
        return {
            "bm25": SimpleNamespace(get_scores=lambda tokens: [0.5]),
            "ids": ["a1"], "metadatas": [{"city": "臺中市"}], "texts": ["某景點描述"],
        }

    def fake_rag_search(query, city=None, district=None, type_=None, weather_fit=None, top_k=5):
        calls["rag_search"] += 1
        return [{"id": "a1", "text": "某景點描述", "city": "臺中市"}]

    monkeypatch.setattr(hybrid_search.bm25_util, "load", fake_load)
    monkeypatch.setattr(hybrid_search, "rag", SimpleNamespace(search=fake_rag_search))
    hybrid_search._search_cached.cache_clear()

    hybrid_search.search("親子景點", city="臺中市")
    hybrid_search.search("親子景點", city="臺中市")
    assert calls["bm25_load"] == 1
    assert calls["rag_search"] == 1


# ---------- bm25_util.py 的索引檔案 lazy singleton ----------

def test_bm25_util_load_caches_after_first_read(monkeypatch, tmp_path):
    bm25_util._index_cache = None
    fake_path = tmp_path / "fake_bm25.pkl"
    payload = {"bm25": None, "ids": [], "metadatas": [], "texts": []}
    with open(fake_path, "wb") as f:
        pickle.dump(payload, f)
    monkeypatch.setattr(bm25_util, "BM25_INDEX_PATH", fake_path)

    result1 = bm25_util.load()
    fake_path.unlink()  # 如果真的有快取，刪掉檔案也不該影響第二次呼叫
    result2 = bm25_util.load()
    assert result1 is result2
