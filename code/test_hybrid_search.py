"""ponytail自我檢查：RRF融合排序、縣市偵測這兩塊純邏輯對不對，不需要真正的
BM25/Chroma資料跑得動。"""
from hybrid_search import detect_cities, rrf_fuse


def test_rrf_fuse_prefers_doc_ranked_high_in_both():
    # 原本用"a","b"完全平手的分數斷言"fused[0] in (a,b)"，這種寫法測不出加總邏輯
    # 錯誤(Codex複核抓到)——換成"d"在兩份排名都墊底、明確該排最後，斷言精確位置。
    fused = rrf_fuse([["a", "b", "c", "d"], ["b", "a", "c", "d"]])
    assert fused[-1] == "d"
    assert set(fused[:2]) == {"a", "b"}  # a、b互相平手排前兩名，順序不特別斷言


def test_rrf_fuse_exact_scores_no_ties():
    # 完全不平手的case，直接核對精確順序，確保加總公式沒有off-by-one。
    fused = rrf_fuse([["a", "b", "c"]], k=1)
    assert fused == ["a", "b", "c"]


def test_rrf_fuse_doc_missing_from_one_ranking_still_included():
    fused = rrf_fuse([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}


def test_rrf_fuse_empty_rankings_returns_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_detect_cities_single_match():
    assert detect_cities("高雄的海邊景點") == ["高雄市"]


def test_detect_cities_ambiguous_bare_name_returns_both():
    matched = detect_cities("新竹室內親子景點")
    assert set(matched) == {"新竹市", "新竹縣"}


def test_detect_cities_exact_full_name_not_widened():
    """Codex複核抓到的真bug：query明確寫「新竹市」時，原本的邏輯還是會因為比對用
    去掉市/縣的裸名而錯誤放寬成新竹市+新竹縣都算。寫出完整縣市名時只該精確匹配那個。"""
    assert detect_cities("新竹市科學園區附近住宿") == ["新竹市"]
    assert detect_cities("新竹縣內灣老街景點") == ["新竹縣"]
    assert detect_cities("嘉義市區美食小吃") == ["嘉義市"]
    assert detect_cities("嘉義縣阿里山森林景點") == ["嘉義縣"]


def test_detect_cities_no_match_returns_none():
    assert detect_cities("室內景點推薦") is None


def test_detect_cities_handles_tai_variant_character():
    """全台22縣市手動驗收發現的坑：CITY_ZH用正式「臺」字，但口語打「台」，
    兩字字面不同，比對前沒正規化就會漏偵測（見hybrid_search.py裡_normalize_tai）。"""
    assert detect_cities("台北信義區文青咖啡廳") == ["臺北市"]
    assert detect_cities("台中文創園區景點") == ["臺中市"]
    assert detect_cities("台南孔廟古蹟景點") == ["臺南市"]
    assert detect_cities("台東池上稻田景點") == ["臺東縣"]


def test_detect_cities_known_alias():
    """馬祖是連江縣的俗名，字面上完全不同，靠手動別名表補上。"""
    assert detect_cities("馬祖藍眼淚景點") == ["連江縣"]


def test_detect_cities_multi_city_query():
    matched = detect_cities("台北到台南自由行景點")
    assert set(matched) == {"臺北市", "臺南市"}


if __name__ == "__main__":
    test_rrf_fuse_prefers_doc_ranked_high_in_both()
    test_rrf_fuse_exact_scores_no_ties()
    test_rrf_fuse_doc_missing_from_one_ranking_still_included()
    test_rrf_fuse_empty_rankings_returns_empty()
    test_detect_cities_single_match()
    test_detect_cities_ambiguous_bare_name_returns_both()
    test_detect_cities_exact_full_name_not_widened()
    test_detect_cities_no_match_returns_none()
    test_detect_cities_handles_tai_variant_character()
    test_detect_cities_known_alias()
    test_detect_cities_multi_city_query()
    print("ok")
