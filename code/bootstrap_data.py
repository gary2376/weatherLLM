"""首次啟動時若 Data/ 不存在(例如全新 clone 的 Streamlit Community Cloud 部署)，
從 GitHub Release 下載預先建好的資料包並解壓縮。

Data/ 被 .gitignore 排除(資料庫與索引加起來約 165MB，是可重新產生的衍生資料，
不適合放進 repo)。本機開發已經有完整的 Data/ 時，ensure_data_present() 直接略過。

安全與一致性：
- 下載後必須通過 SHA-256 比對(格式不對或沒設定一律拒絕，不會靜默略過)。
- 只允許一般檔案與資料夾，最上層名稱限定三個產物；擋路徑穿越、符號/硬連結、
  特殊檔(FIFO、裝置)，並限制成員數與解壓縮大小。
- 先完整解壓縮並驗證到同檔案系統的暫存區，才逐項換上；失敗會還原舊資料。
- 行程內用 lock、跨行程用檔案 lock，避免兩個 session 同時安裝。

資料包內容與授權見 DATA_LICENSE.md。重新產生資料後要建新的 release 並更新下方的
DATA_URL 與 DATA_SHA256。
"""
import hashlib
import os
import re
import shutil
import sqlite3
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

import requests

try:  # 檔案 lock 只在 POSIX 可用；沒有時退回行程內 lock
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

DATA_URL = os.getenv(
    "WEATHERLLM_DATA_URL",
    "https://github.com/gary2376/weatherLLM/releases/download/data-v1/weatherllm-data-v1.tar.gz",
)
DATA_SHA256 = os.getenv(
    "WEATHERLLM_DATA_SHA256",
    "fd80aaae97e34faa7348c59a76b43d61092ad46dde286a8dc29929bea2a5415c",
)

REQUIRED = ("travel.db", "bm25_index.pkl", "chroma")
MIN_DB_BYTES = 1_000_000
MIN_BM25_BYTES = 100_000
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
DOWNLOAD_DEADLINE_SECONDS = 600

_process_lock = threading.Lock()


class BootstrapDataError(RuntimeError):
    """取得資料失敗(下載、校驗、解壓縮或安裝)。訊息不含內部路徑細節以外的敏感內容。"""


def data_present(data_dir: Path = DATA_DIR) -> bool:
    db, bm25, chroma = (data_dir / n for n in REQUIRED)
    return (
        not (data_dir / ".bootstrap-old").exists()  # 殘留表示上次安裝被中斷，資料可能新舊混合
        and db.is_file()
        and db.stat().st_size > MIN_DB_BYTES
        and bm25.is_file()
        and bm25.stat().st_size > MIN_BM25_BYTES
        and (chroma / "chroma.sqlite3").is_file()
    )


def _download(url: str, dest: Path) -> None:
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS
    size = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise BootstrapDataError("資料包超過大小上限，已中止。")
                if time.monotonic() > deadline:
                    raise BootstrapDataError("下載逾時，已中止。")
                f.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_members(tar: tarfile.TarFile) -> None:
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise BootstrapDataError("資料包成員數量超過上限。")
    total = 0
    for m in members:
        parts = Path(m.name).parts
        if (
            not parts
            or parts[0] not in REQUIRED
            or m.name.startswith("/")
            or ".." in parts
            or not (m.isreg() or m.isdir())
        ):
            raise BootstrapDataError(f"資料包含不允許的路徑或類型：{m.name}")
        if m.isreg():
            if m.size > MAX_FILE_BYTES:
                raise BootstrapDataError("資料包內有檔案超過大小上限。")
            total += m.size
    if total > MAX_TOTAL_BYTES:
        raise BootstrapDataError("資料包解壓縮後超過大小上限。")


def _extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        _validate_members(tar)
        try:
            tar.extractall(target, filter="data")
        except TypeError:  # 舊版 Python 沒有 filter 參數；上面已逐一驗證成員
            tar.extractall(target)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _install(staging: Path, data_dir: Path) -> None:
    """把暫存區裡的三個產物換上去；失敗就移除新放的、還原舊資料。
    只有成功或完整還原後才刪除備份；還原也失敗時保留備份供人工復原。"""
    backup = data_dir / ".bootstrap-old"
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir()
    existed = {n for n in REQUIRED if (data_dir / n).exists()}
    moved, placed = [], []
    try:
        for name in REQUIRED:
            live = data_dir / name
            if name in existed:
                os.replace(live, backup / name)
                moved.append(name)
            os.replace(staging / name, live)
            placed.append(name)
    except Exception:
        for name in placed:
            _remove(data_dir / name)
        for name in moved:  # 這裡若再失敗，備份留在原地，例外往上丟
            os.replace(backup / name, data_dir / name)
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _staging_ok(staging: Path) -> bool:
    """暫存區除了檔案齊全，還要 travel.db 真的是可讀、有內容的 SQLite。"""
    if not data_present(staging):
        return False
    try:
        with sqlite3.connect(f"file:{staging / 'travel.db'}?mode=ro", uri=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM attractions").fetchone()[0] > 0
    except sqlite3.Error:
        return False


def ensure_data_present(
    data_dir: Path = DATA_DIR,
    url: Optional[str] = None,
    sha256: Optional[str] = None,
) -> bool:
    """資料已齊全回傳 False(沒做事)；下載並安裝完成回傳 True；失敗丟 BootstrapDataError。"""
    try:
        return _ensure(data_dir, url, sha256)
    except BootstrapDataError:
        raise
    except (requests.RequestException, tarfile.TarError, sqlite3.Error, OSError) as exc:
        raise BootstrapDataError(f"取得資料失敗：{type(exc).__name__}") from exc


def _ensure(data_dir: Path, url: Optional[str], sha256: Optional[str]) -> bool:
    if data_present(data_dir):
        return False
    expected = (DATA_SHA256 if sha256 is None else sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise BootstrapDataError("未設定有效的 SHA-256，拒絕下載。")
    url = url or DATA_URL
    with _process_lock:
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / ".bootstrap.lock", "w") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            if data_present(data_dir):  # 等鎖期間別的 session 可能已經裝好了
                return False
            work = data_dir / ".bootstrap-work"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir()
            try:
                archive = work / "data.tar.gz"
                staging = work / "out"
                staging.mkdir()
                _download(url, archive)
                if _sha256(archive) != expected:
                    raise BootstrapDataError("資料包的 SHA-256 與預期不符，已中止。")
                _extract(archive, staging)
                if not _staging_ok(staging):
                    raise BootstrapDataError("資料包內容不完整或資料庫無法讀取，已中止。")
                _install(staging, data_dir)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    if not data_present(data_dir):
        raise BootstrapDataError("安裝後資料仍不完整。")
    return True


if __name__ == "__main__":
    print("downloaded" if ensure_data_present() else "data already present")
