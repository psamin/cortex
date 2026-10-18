import zipfile

import pytest

from cortex.services.import_service import extract_archive, find_dataset_root


def test_extract_finds_nested_dataset_root(tmp_path):
    arc = tmp_path / "demos.zip"
    with zipfile.ZipFile(arc, "w") as z:
        z.writestr("my_so101/meta/info.json", '{"total_episodes": 3}')
        z.writestr("my_so101/data/chunk-000/file-000.parquet", b"")
    root = extract_archive(arc, tmp_path / "out")
    assert root == tmp_path / "out" / "my_so101"
    assert find_dataset_root(tmp_path / "out") == root


def test_extract_refuses_path_traversal_and_non_datasets(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        extract_archive(evil, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("notes.txt", "x")
    with pytest.raises(ValueError, match="no LeRobot dataset"):
        extract_archive(empty, tmp_path / "out2")
    (tmp_path / "plain.txt").write_text("x")
    with pytest.raises(ValueError, match=".zip or .tar.gz"):
        extract_archive(tmp_path / "plain.txt", tmp_path / "out3")


def _dataset_zip(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ds/meta/info.json", '{"total_episodes": 1}')
        z.writestr("ds/data/chunk-000/file-000.parquet", b"x" * 4096)
    return path


def test_save_upload_streams_to_disk_and_hashes(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from cortex.services.import_service import save_upload

    arc = _dataset_zip(tmp_path / "demos.zip")
    ctx = SimpleNamespace(local=lambda key: tmp_path / "data" / key)
    with arc.open("rb") as f:
        src = save_upload(ctx, "prj_1", "../../demos.zip", f)
    assert src["sha256"] == hashlib.sha256(arc.read_bytes()).hexdigest()
    assert src["path"].endswith("ds") and (tmp_path / "data" / "projects" / "prj_1").exists()


def test_upload_and_unpacked_size_limits(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cortex.services import import_service

    arc = _dataset_zip(tmp_path / "demos.zip")
    ctx = SimpleNamespace(local=lambda key: tmp_path / "data" / key)
    monkeypatch.setattr(import_service, "MAX_UPLOAD_BYTES", 100)
    with arc.open("rb") as f, pytest.raises(ValueError, match="larger than 2 GB"):
        import_service.save_upload(ctx, "prj_1", "demos.zip", f)
    assert not any((tmp_path / "data" / "projects" / "prj_1" / "imports").iterdir())  # partial upload removed

    monkeypatch.setattr(import_service, "MAX_EXTRACTED_BYTES", 1000)  # the archive unpacks to > 4 KB
    with pytest.raises(ValueError, match="unpacks to"):
        import_service.extract_archive(arc, tmp_path / "out")
    assert not (tmp_path / "out" / "ds").exists()
