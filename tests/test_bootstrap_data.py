"""`code/bootstrap_data.py`的deterministic測試：monkeypatch掉`requests.get`跟
`DATA_DIR`，不需要真的打GitHub Release網址。見 ROADMAP.md「資料版本化」項目。
"""
import io
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import bootstrap_data  # noqa: E402


def _make_real_sqlite_db(path: Path, with_attractions_table: bool = True) -> None:
    """建一個真的SQLite檔案（不是隨便寫字串），因為`_travel_db_has_data()`會
    真的用`sqlite3.connect()`打開來查`attractions`表存不存在，用假的文字檔案
    測不出這條邏輯有沒有正確運作。"""
    conn = sqlite3.connect(path)
    if with_attractions_table:
        conn.execute("CREATE TABLE attractions (id TEXT)")
        conn.execute("INSERT INTO attractions VALUES ('A1')")
    else:
        conn.execute("CREATE TABLE something_else (id TEXT)")
    conn.commit()
    # 補到超過MIN_TRAVEL_DB_BYTES門檻，避免因為「檔案太小」被判定不完整，
    # 這裡要測的是「有沒有attractions表」這條邏輯，不是檔案大小門檻本身
    conn.execute(f"CREATE TABLE padding (blob TEXT)")
    conn.execute(f"INSERT INTO padding VALUES ('{'x' * (bootstrap_data.MIN_TRAVEL_DB_BYTES + 1000)}')")
    conn.commit()
    conn.close()


def test_travel_db_has_data_false_when_file_missing(tmp_path):
    assert bootstrap_data._travel_db_has_data(tmp_path / "nope.db") is False


def test_travel_db_has_data_false_when_too_small(tmp_path):
    """回歸測試（線上部署真的踩到的bug）：`sqlite3.connect()`對不存在的路徑會
    靜靜建立一個空資料庫，不會報錯——單純檢查`.exists()`會被這個空檔案騙過去。
    這裡確認小於門檻的檔案（包含這種意外產生的空db）會被正確判定為不完整。"""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)  # 模擬「意外建立的空db」
    conn.close()
    assert bootstrap_data._travel_db_has_data(db_path) is False


def test_travel_db_has_data_false_when_attractions_table_missing(tmp_path):
    db_path = tmp_path / "wrong.db"
    _make_real_sqlite_db(db_path, with_attractions_table=False)
    assert bootstrap_data._travel_db_has_data(db_path) is False


def test_travel_db_has_data_true_when_valid(tmp_path):
    db_path = tmp_path / "good.db"
    _make_real_sqlite_db(db_path, with_attractions_table=True)
    assert bootstrap_data._travel_db_has_data(db_path) is True


def test_data_exists_false_when_directory_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path / "does_not_exist")
    assert bootstrap_data.data_exists() is False


def test_data_exists_false_when_other_files_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path)
    _make_real_sqlite_db(tmp_path / "travel.db")
    # 缺 bm25_index.pkl / cctv_enhanced.csv / chroma/，不算完整
    assert bootstrap_data.data_exists() is False


def test_data_exists_true_when_all_required_paths_present_and_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path)
    _make_real_sqlite_db(tmp_path / "travel.db")
    for name in ["bm25_index.pkl", "cctv_enhanced.csv"]:
        (tmp_path / name).write_text("x")
    (tmp_path / "chroma").mkdir()
    assert bootstrap_data.data_exists() is True


def _fake_tar_bytes(valid_db: bool = True) -> bytes:
    """建一個最小的假tarball，內容跟真實資料包同樣結構，travel.db是真的SQLite
    檔案（`ensure_data_present()`會真的驗證內容），其餘三項是佔位檔案。"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "travel.db"
        _make_real_sqlite_db(db_path, with_attractions_table=valid_db)
        (tmp_path / "bm25_index.pkl").write_text("fake-bm25")
        (tmp_path / "cctv_enhanced.csv").write_text("fake-cctv")
        (tmp_path / "chroma").mkdir()
        (tmp_path / "chroma" / "chroma.sqlite3").write_text("fake-chroma")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for item in tmp_path.iterdir():
                tar.add(item, arcname=item.name)
        return buf.getvalue()


def test_ensure_data_present_skips_download_when_data_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path)
    _make_real_sqlite_db(tmp_path / "travel.db")
    for name in ["bm25_index.pkl", "cctv_enhanced.csv"]:
        (tmp_path / name).write_text("existing")
    (tmp_path / "chroma").mkdir()

    called = {"get": False}
    monkeypatch.setattr(bootstrap_data.requests, "get", lambda *a, **k: called.update(get=True))

    bootstrap_data.ensure_data_present()
    assert called["get"] is False  # 資料已經存在，不該真的發request


def test_ensure_data_present_downloads_and_extracts_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path / "Data")
    tar_bytes = _fake_tar_bytes(valid_db=True)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield tar_bytes

    monkeypatch.setattr(bootstrap_data.requests, "get", lambda *a, **k: FakeResponse())

    assert bootstrap_data.data_exists() is False
    bootstrap_data.ensure_data_present()
    assert bootstrap_data.data_exists() is True
    assert (bootstrap_data.DATA_DIR / "bm25_index.pkl").read_text() == "fake-bm25"


def test_ensure_data_present_raises_when_downloaded_db_is_invalid(tmp_path, monkeypatch):
    """回歸測試：下載回來的資料包本身`travel.db`不完整（例如下載中途被截斷）
    時，要明確報錯，不能悄悄留下一個壞掉的`Data/`資料夾讓後續查詢用一堆難懂
    的`no such table`錯誤收場。"""
    monkeypatch.setattr(bootstrap_data, "DATA_DIR", tmp_path / "Data")
    tar_bytes = _fake_tar_bytes(valid_db=False)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield tar_bytes

    monkeypatch.setattr(bootstrap_data.requests, "get", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="不完整"):
        bootstrap_data.ensure_data_present()
    assert not (tmp_path / "Data").exists()  # 失敗時不該留下半殘的Data/資料夾
