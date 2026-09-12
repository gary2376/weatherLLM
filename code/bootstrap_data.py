"""首次啟動時如果`Data/`資料夾不存在（例如全新clone的Streamlit Community Cloud
部署——`Data/`整個被`.gitignore`排除、不進git版控，因為`travel.db`/Chroma索引/
BM25索引加起來將近200MB，是可以重新產生的衍生資料，不是原始碼，不適合塞進repo），
從GitHub Release下載已經打包好的資料包並解壓縮。

本機開發不會走到這裡：`Data/`已經存在且內容完整的話，`ensure_data_present()`
直接跳過，不會每次啟動都重新下載。
"""
import os
import shutil
import sqlite3
import tarfile
import tempfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

# 見 collector/build_embeddings.py / derive_district_coords.py 的產出物：
# travel.db（SQLite，含attractions/lodgings/restaurants/transit_stations/
# district_coords）、chroma/（語意檢索向量索引）、bm25_index.pkl（BM25 lexical
# 索引）、cctv_enhanced.csv（CCTV圖釘資料）。這個URL對應repo release頁面的
# `data-v1` tag，重新產生資料庫後要記得建新的release、更新這個URL。
DATA_RELEASE_URL = (
    "https://github.com/gary2376/weatherLLM/releases/download/data-v1/weatherllm-data-v1.tar.gz"
)

REQUIRED_PATHS = ["travel.db", "bm25_index.pkl", "cctv_enhanced.csv", "chroma"]

# 真正的travel.db有38MB+，小於這個門檻代表下載/解壓縮不完整或是空的
MIN_TRAVEL_DB_BYTES = 1_000_000


def _log(msg: str) -> None:
    # 容器環境裡stdout對pipe是block-buffered，不flush的話print()可能一直卡在
    # buffer裡，Streamlit Cloud的log面板看不到——之前排查線上bug時就是卡在這裡
    print(msg, flush=True)


def _travel_db_has_data(db_path: Path) -> bool:
    """線上部署真的發生過的bug：只檢查檔案存不存在不夠——`sqlite3.connect()`
    對一個不存在的路徑會靜靜建立一個空的新資料庫，不會報錯。如果任何原因
    （例如兩個Streamlit session同時觸發下載互相干擾）讓一個空/不完整的
    `travel.db`出現在`Data/`底下，只看`.exists()`的舊版`data_exists()`會永遠
    誤判成「資料已經在」，之後永遠不會重新下載，卡死在壞掉的狀態。這裡除了
    看檔案大小，還真的開起來查一次`attractions`表存不存在，兩層防護。

    注意：連線用`sqlite3.connect(str(db_path))`，不要改成URI模式
    （`file:{path}?mode=ro`）——單斜線的URI寫法在這台機器的sqlite build上對
    絕對路徑會直接丟`unable to open database file`（正確寫法要三個斜線
    `file:///...`），曾經讓一份確認正常的資料庫被誤判成損毀，害我一開始以為
    是暫時性的檔案系統時機問題去查了半天。這裡existence已經檢查過，用預設
    的可寫模式開檔不會有意外建立空db的風險。"""
    if not db_path.exists() or db_path.stat().st_size < MIN_TRAVEL_DB_BYTES:
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("SELECT 1 FROM attractions LIMIT 1")
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def data_exists() -> bool:
    if not _travel_db_has_data(DATA_DIR / "travel.db"):
        return False
    return all((DATA_DIR / p).exists() for p in REQUIRED_PATHS if p != "travel.db")


def ensure_data_present(url: str = DATA_RELEASE_URL, timeout: int = 300) -> None:
    """`Data/`資料不完整時，從GitHub Release下載資料包解壓縮進去；資料已經存在
    且驗證完整就直接跳過。失敗時直接讓例外往上拋——呼叫端（`streamlit_app.py`）
    沒有這批資料的話，天氣查詢以外的每個功能本來就會壞掉，靜靜吞掉例外只會讓
    使用者看到更難懂的後續錯誤（找不到資料庫表之類），不如在啟動當下就講清楚
    是資料下載失敗。

    下載+解壓縮全部在一個獨立的暫存目錄裡完成，驗證過內容完整後才用
    `os.replace()`把整個目錄換到`Data/`——線上部署真的發生過兩個Streamlit
    session同時觸發下載、各自寫入同一個`Data/`資料夾互相干擾、拼出一個損毀的
    `travel.db`的race condition，改成先在別處完整組好再一次換過去，把「寫入
    中」這個危險視窗從整段下載+解壓縮時間（可能一兩分鐘）縮到`os.replace()`
    那一個原子操作，大幅降低撞在一起的機率（不是理論上完全消除，但對這種
    低流量個人專案已經足夠，不需要再上檔案鎖這種更重的機制）。

    這裡的`print()`不是裝飾——`st.spinner()`的文字只在瀏覽器端看得到，
    Streamlit Cloud後台的「Manage app」log面板完全看不到，之前排查線上
    bug時就是卡在不知道這個function到底有沒有被呼叫到、跑到哪一步就停了。
    `print()`會直接進stdout，Streamlit Cloud的log面板看得到。"""
    if data_exists():
        _log("[bootstrap_data] Data/ 已存在且驗證通過，跳過下載")
        return

    _log(f"[bootstrap_data] Data/ 缺少或驗證失敗，開始從 {url} 下載")
    with tempfile.TemporaryDirectory(dir=DATA_DIR.parent) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        tar_path = tmp_dir / "bundle.tar.gz"
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())  # 確保下載內容真的落到磁碟，不是還停留在OS的write buffer裡
        _log(f"[bootstrap_data] 下載完成，tar大小={tar_path.stat().st_size} bytes")

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir)
        _log("[bootstrap_data] 解壓縮完成，開始驗證 travel.db")

        if not _travel_db_has_data(extract_dir / "travel.db"):
            raise RuntimeError(
                f"從 {url} 下載的資料包解壓縮後 travel.db 仍然不完整（查不到 attractions 表），"
                "可能是下載中途被截斷，或release資產本身有問題"
            )

        # 別的process可能在我們下載的這段時間已經搶先完成了，那就用現成的，
        # 不要用自己這份去覆蓋一個可能已經是完整的資料
        if data_exists():
            _log("[bootstrap_data] 另一個session已經先完成下載，直接使用現成的")
            return
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR, ignore_errors=True)
        os.replace(extract_dir, DATA_DIR)
        _log("[bootstrap_data] 已將解壓縮結果換入 Data/，完成")
