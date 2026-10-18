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
