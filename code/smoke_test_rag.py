"""手動驗收腳本，不是pytest單元測試，但會累積失敗項目、結尾用exit code回報，
避免像原本那樣「印出✗但exit code還是0」造成自動化執行時的假陽性(Codex複核抓到)。
三部分：
1) CITY_CASES：全台22個縣市各跑一筆查詢，檢查偵測到的city跟結果的city是否都對得
   上——刻意用「台」而非「臺」測`_normalize_tai()`、用「馬祖」測連江縣別名表。
2) THEMATIC_CASES：不帶縣市的主題性查詢，只能機器檢查「有沒有回傳結果」，語意
   相不相關要人工看印出來的內容。
3) 邊界case：cache一致性/確實命中、clear_cache()、空查詢、資料最稀疏的連江縣、
   一次講兩個縣市。

跑法：python3 code/smoke_test_rag.py
"""
import sys

import hybrid_search
import rag
from hybrid_search import detect_cities

CITY_CASES = [
    ("台北信義區文青咖啡廳", ["臺北市"]),
    ("新北登山步道景點", ["新北市"]),
    ("桃園親子景點推薦", ["桃園市"]),
    ("台中文創園區景點", ["臺中市"]),
    ("台南孔廟古蹟景點", ["臺南市"]),
    ("高雄駁二藝術特區", ["高雄市"]),
    ("基隆廟口夜市小吃", ["基隆市"]),
    ("新竹市科學園區附近住宿", ["新竹市"]),
    ("新竹縣內灣老街景點", ["新竹縣"]),
    ("苗栗客家桐花公園景點", ["苗栗縣"]),
    ("彰化鹿港老街景點", ["彰化縣"]),
    ("南投日月潭景點", ["南投縣"]),
    ("雲林農業體驗景點", ["雲林縣"]),
    ("嘉義市區美食小吃", ["嘉義市"]),
    ("嘉義縣阿里山森林景點", ["嘉義縣"]),
    ("屏東墾丁海邊景點", ["屏東縣"]),
    ("宜蘭礁溪溫泉民宿", ["宜蘭縣"]),
    ("花蓮太魯閣峽谷景點", ["花蓮縣"]),
    ("台東池上稻田景點", ["臺東縣"]),
    ("澎湖跨海大橋景點", ["澎湖縣"]),
    ("金門古厝民宿", ["金門縣"]),
    ("馬祖藍眼淚景點", ["連江縣"]),  # 連江縣的俗名，靠_KNOWN_ALIASES補上，會通過
]

THEMATIC_CASES = [
    "適合帶小孩的親子景點",
    "背包客青年旅館推薦",
    "素食友善餐廳",
    "適合雨天的室內景點",
]


def run_city_cases():
    failures = []
    total_hit, total_n = 0, 0
    for q, expect_cities in CITY_CASES:
        detected = detect_cities(q)
        detected_ok = set(detected or []) == set(expect_cities)
        docs = rag.query(q, top_k=5)
        hit = sum(1 for d in docs if d["metadata"].get("city") in expect_cities)
        total_hit += hit
        total_n += len(docs)
        ok = detected_ok and len(docs) > 0 and hit == len(docs)
        mark = "✓" if ok else ("△" if hit > 0 else "✗")
        print(f"{mark} {q}｜預期{expect_cities}｜偵測到city={detected}｜命中{hit}/{len(docs)}")
        if not ok:
            failures.append(f"{q}：預期city={expect_cities}，實際偵測={detected}，命中{hit}/{len(docs)}")
    if total_n == 0:
        print("\n⚠️ 全部查詢都回傳0筆結果，無法算命中率")
        failures.append("全部city查詢皆回傳0筆")
    else:
        print(f"\n總命中率：{total_hit}/{total_n}（{total_hit/total_n:.0%}）")
    return failures


def run_thematic_cases():
    print("\n=== 主題性查詢（無city過濾，人工看語意相關性）===")
    failures = []
    for q in THEMATIC_CASES:
        docs = rag.query(q, top_k=5)
        print(f"\n查詢：{q}")
        if not docs:
            print("  (無結果)")
            failures.append(f"{q}：回傳0筆")
            continue
        for d in docs:
            meta = d["metadata"]
            print(f"  [{meta.get('type')}] {meta.get('city')}{meta.get('district')} — {d['text'][:40]}")
    return failures


def run_edge_cases():
    print("\n=== 邊界case ===")
    failures = []

    q = "台北文青咖啡廳"
    hybrid_search.clear_cache()
    r1 = hybrid_search.search(q)
    info_after_first = hybrid_search.search.cache_info()
    r2 = hybrid_search.search(q)
    info_after_second = hybrid_search.search.cache_info()
    cache_hit_registered = (info_after_second.hits - info_after_first.hits) == 1
    print(
        f"cache結果一致：{'✓' if r1 == r2 else '✗'}｜"
        f"cache_info().hits真的+1(不是只是結果剛好一樣)：{'✓' if cache_hit_registered else '✗'}"
    )
    if r1 != r2 or not cache_hit_registered:
        failures.append("cache一致性或cache命中計數檢查失敗")

    before = hybrid_search.search.cache_info().currsize
    hybrid_search.clear_cache()
    after = hybrid_search.search.cache_info().currsize
    print(f"clear_cache()：清前currsize={before}，清後currsize={after}")
    if after != 0:
        failures.append(f"clear_cache()沒清乾淨，清後currsize={after}")

    for q in ["", "   "]:
        try:
            docs = rag.query(q)
            print(f"空查詢{q!r}沒crash，回傳{len(docs)}筆(語意上算不算合理，留給agent/UI層擋輸入決定)")
        except Exception as e:
            print(f"空查詢{q!r}丟出例外：{e!r}")

    try:
        docs = rag.query("連江縣景點", top_k=5)
        print(f"連江縣(全台資料量最稀疏)查詢沒crash，回傳{len(docs)}筆")
        if not docs:
            failures.append("連江縣查詢回傳0筆")
    except Exception as e:
        print(f"✗ 連江縣查詢crash了：{e!r}")
        failures.append(f"連江縣查詢crash：{e!r}")

    q = "台北到台南自由行景點"
    expect = {"臺北市", "臺南市"}
    try:
        detected = detect_cities(q)
        docs = rag.query(q, top_k=5)
        detected_ok = set(detected or []) == expect
        cities_ok = bool(docs) and all(d["metadata"].get("city") in expect for d in docs)
        print(f"多縣市查詢偵測到{detected}，回傳{len(docs)}筆，city都在範圍內：{'✓' if cities_ok else '✗'}")
        if not detected_ok or not cities_ok:
            failures.append(f"多縣市查詢檢查失敗：detected={detected}，cities_ok={cities_ok}，docs數={len(docs)}")
    except Exception as e:
        print(f"✗ 多縣市查詢crash了：{e!r}")
        failures.append(f"多縣市查詢crash：{e!r}")

    return failures


if __name__ == "__main__":
    all_failures = run_city_cases() + run_thematic_cases() + run_edge_cases()
    if all_failures:
        print(f"\n{'=' * 40}\n{len(all_failures)}項未通過：")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n全部通過")
