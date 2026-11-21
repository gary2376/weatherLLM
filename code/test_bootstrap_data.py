"""bootstrap_data 與 reranker 開關的單元測試(全部本機模擬，不實際下載、不載入模型)。"""
import hashlib
import io
import os
import tarfile
import threading
from pathlib import Path

import pytest

import bootstrap_data as bd
import rag


def _archive(path: Path, members: dict[str, bytes], extra=None) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tar.addfile(info)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sqlite_bytes(rows: int = 1) -> bytes:
    import sqlite3, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE attractions (id INTEGER, pad BLOB)")
        for i in range(rows):
            conn.execute("INSERT INTO attractions VALUES (?, ?)", (i, b"x" * (bd.MIN_DB_BYTES + 10)))
        conn.commit()
        conn.close()
        return path.read_bytes()


GOOD = {
    "travel.db": _sqlite_bytes(),
    "bm25_index.pkl": b"b" * (bd.MIN_BM25_BYTES + 10),
    "chroma/chroma.sqlite3": b"c",
}


def _serve(monkeypatch, archive: Path, counter=None):
    def fake(url, dest):
        if counter is not None:
            counter.append(1)
        dest.write_bytes(archive.read_bytes())
    monkeypatch.setattr(bd, "_download", fake)


def _write_data(d: Path, tag: bytes = b"old") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "travel.db").write_bytes(tag * (bd.MIN_DB_BYTES // len(tag) + 1))
    (d / "bm25_index.pkl").write_bytes(tag * (bd.MIN_BM25_BYTES // len(tag) + 1))
    (d / "chroma").mkdir(exist_ok=True)
    (d / "chroma" / "chroma.sqlite3").write_bytes(tag)


def test_present_data_is_not_downloaded(tmp_path, monkeypatch):
    _write_data(tmp_path)
    monkeypatch.setattr(bd, "_download", lambda *a: pytest.fail("不該下載"))
    assert bd.ensure_data_present(tmp_path, url="http://x", sha256="0" * 64) is False


def test_missing_data_is_downloaded_verified_and_installed(tmp_path, monkeypatch):
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    data = tmp_path / "Data"
    assert bd.ensure_data_present(data, url="http://x", sha256=sha) is True
    assert bd.data_present(data)
    assert not (data / ".bootstrap-work").exists() and not (data / ".bootstrap-old").exists()


@pytest.mark.parametrize("bad_sha", ["", "not-hex", "0" * 63, "0" * 64])
def test_missing_invalid_or_wrong_sha_is_refused(tmp_path, monkeypatch, bad_sha):
    src = tmp_path / "src.tar.gz"
    _archive(src, GOOD)
    _serve(monkeypatch, src)
    data = tmp_path / "Data"
    with pytest.raises(bd.BootstrapDataError):
        bd.ensure_data_present(data, url="http://x", sha256=bad_sha)
    assert not bd.data_present(data)


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs.txt", "other/file.txt"])
def test_unexpected_paths_are_rejected(tmp_path, monkeypatch, evil):
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, {**GOOD, evil: b"bad"})
    _serve(monkeypatch, src)
    with pytest.raises(bd.BootstrapDataError, match="不允許"):
        bd.ensure_data_present(tmp_path / "Data", url="http://x", sha256=sha)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_special_members_are_rejected(tmp_path, monkeypatch, kind):
    info = tarfile.TarInfo("chroma/special")
    info.type = kind
    info.linkname = "/etc/passwd"
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD, extra=[info])
    _serve(monkeypatch, src)
    with pytest.raises(bd.BootstrapDataError, match="不允許"):
        bd.ensure_data_present(tmp_path / "Data", url="http://x", sha256=sha)


def test_size_limits_are_enforced(tmp_path, monkeypatch):
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    monkeypatch.setattr(bd, "MAX_FILE_BYTES", 10)
    with pytest.raises(bd.BootstrapDataError, match="大小上限"):
        bd.ensure_data_present(tmp_path / "Data", url="http://x", sha256=sha)


def test_incomplete_archive_is_rejected_and_old_data_untouched(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    _write_data(data)
    (data / "bm25_index.pkl").write_bytes(b"")  # partial 舊資料 -> 觸發下載
    before = (data / "travel.db").read_bytes()
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, {"travel.db": GOOD["travel.db"]})  # 缺 bm25 與 chroma
    _serve(monkeypatch, src)
    with pytest.raises(bd.BootstrapDataError, match="不完整"):
        bd.ensure_data_present(data, url="http://x", sha256=sha)
    assert (data / "travel.db").read_bytes() == before


def test_partial_existing_data_triggers_download(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    _write_data(data)
    (data / "bm25_index.pkl").write_bytes(b"")
    assert not bd.data_present(data)
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    assert bd.ensure_data_present(data, url="http://x", sha256=sha) is True


def test_install_failure_rolls_back_to_old_data(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    _write_data(data, b"old")
    (data / "chroma" / "chroma.sqlite3").unlink()  # 讓 present=False 以觸發安裝
    old_db = (data / "travel.db").read_bytes()
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 4:  # 第二個產物換上去時失敗
            raise OSError("disk full")
        return real_replace(a, b)

    monkeypatch.setattr(bd.os, "replace", flaky)
    with pytest.raises(bd.BootstrapDataError):
        bd.ensure_data_present(data, url="http://x", sha256=sha)
    monkeypatch.setattr(bd.os, "replace", real_replace)
    assert (data / "travel.db").read_bytes() == old_db
    assert (data / "bm25_index.pkl").exists() and (data / "chroma").is_dir()
    assert not (data / ".bootstrap-old").exists()


def test_corrupt_archive_becomes_bootstrap_error(tmp_path, monkeypatch):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not a tar")
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    _serve(monkeypatch, bad)
    with pytest.raises(bd.BootstrapDataError):
        bd.ensure_data_present(tmp_path / "Data", url="http://x", sha256=sha)


def test_concurrent_sessions_download_only_once(tmp_path, monkeypatch):
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    calls: list = []
    _serve(monkeypatch, src, calls)
    data = tmp_path / "Data"
    results, errors = [], []

    def run():
        try:
            results.append(bd.ensure_data_present(data, url="http://x", sha256=sha))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(calls) == 1 and results.count(True) == 1 and bd.data_present(data)


def _stub_rag(monkeypatch):
    docs = {f"d{i}": {"id": f"d{i}", "text": f"t{i}", "metadata": {}} for i in range(8)}
    monkeypatch.setattr(rag.hybrid_search, "search", lambda q: tuple(docs))
    monkeypatch.setattr(rag, "_get_doc_lookup", lambda: docs)


def test_lite_mode_skips_reranker_and_keeps_fused_order(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setenv("WEATHERLLM_RERANK", "0")
    monkeypatch.setattr(rag.reranker_fn, "rerank_indices", lambda *a, **k: pytest.fail("不該呼叫 reranker"))
    assert [d["id"] for d in rag.query("q", top_k=3)] == ["d0", "d1", "d2"]


def test_default_mode_still_uses_reranker(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.delenv("WEATHERLLM_RERANK", raising=False)
    monkeypatch.setattr(rag.reranker_fn, "rerank_indices", lambda q, texts, top_k: [4, 2])
    assert [d["id"] for d in rag.query("q", top_k=2)] == ["d4", "d2"]


def test_unreadable_database_in_bundle_is_rejected(tmp_path, monkeypatch):
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, {**GOOD, "travel.db": b"x" * (bd.MIN_DB_BYTES + 10)})  # 不是 SQLite
    _serve(monkeypatch, src)
    with pytest.raises(bd.BootstrapDataError, match="不完整|無法讀取"):
        bd.ensure_data_present(tmp_path / "Data", url="http://x", sha256=sha)


def test_interrupted_install_leftover_forces_reinstall(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    _write_data(data, b"mixed")  # 看起來齊全，但上次安裝被中斷
    (data / ".bootstrap-old").mkdir()
    assert not bd.data_present(data)
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    assert bd.ensure_data_present(data, url="http://x", sha256=sha) is True
    assert not (data / ".bootstrap-old").exists()
    assert (data / "travel.db").read_bytes() == GOOD["travel.db"]


def test_failed_rollback_keeps_backup_for_manual_recovery(tmp_path, monkeypatch):
    data = tmp_path / "Data"
    _write_data(data, b"old")
    (data / "chroma" / "chroma.sqlite3").unlink()
    src = tmp_path / "src.tar.gz"
    sha = _archive(src, GOOD)
    _serve(monkeypatch, src)
    real_replace = os.replace
    calls = {"n": 0}

    def always_fail_after_two(a, b):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError("io error")
        return real_replace(a, b)

    monkeypatch.setattr(bd.os, "replace", always_fail_after_two)
    with pytest.raises(bd.BootstrapDataError):
        bd.ensure_data_present(data, url="http://x", sha256=sha)
    monkeypatch.setattr(bd.os, "replace", real_replace)
    assert (data / ".bootstrap-old").exists()  # 還原失敗時舊資料沒被刪掉


def test_filesystem_errors_become_bootstrap_error(tmp_path, monkeypatch):
    blocker = tmp_path / "Data"
    blocker.write_text("i am a file, not a directory")
    with pytest.raises(bd.BootstrapDataError):
        bd.ensure_data_present(blocker, url="http://x", sha256="0" * 64)
