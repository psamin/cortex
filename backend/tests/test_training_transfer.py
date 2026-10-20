"""#346: cluster copies work from a Windows backend (no C:\\ path ever reaches rsync/scp; scp when there's no rsync)."""
import asyncio

import pytest

from cortex.services import training_service as ts

REAL_PREFLIGHT = ts.preflight_transfer_tools


def _record(calls):
    async def sh(*args, timeout=600, cwd=None, input=None):
        calls.append({"args": list(args), "cwd": cwd, "input": input})
        return 0, "12345\n" if any("sbatch --parsable" in a for a in args) else ""
    return sh


def _no_drive_letters(calls):
    for c in calls:
        for a in c["args"]:
            assert not (len(a) > 2 and a[1] == ":" and a[0].isalpha()), f"absolute drive path passed to {c['args'][0]}: {a}"


@pytest.mark.parametrize("have_rsync", [True, False])
async def test_dataset_copy_uses_relative_local_paths(tmp_path, monkeypatch, have_rsync):
    calls = []
    monkeypatch.setattr(ts, "_sh", _record(calls))
    monkeypatch.setattr(ts.shutil, "which", lambda t: (None if t == "rsync" and not have_rsync else f"/usr/bin/{t}"))
    ds = tmp_path / "projects" / "p" / "datasets" / "d" / "lerobot"
    ds.mkdir(parents=True)
    await ts._push_dir("futurama", ds, "~/cortex/jobs/tj_1/dataset")
    await ts._pull_dir("futurama", "~/cortex/jobs/tj_1/output/pretrained_model", tmp_path / "out" / "pretrained_model")
    _no_drive_letters(calls)
    copy = [c for c in calls if c["args"][0] in ("rsync", "scp")]
    assert [c["args"][0] for c in copy] == (["rsync", "rsync"] if have_rsync else ["scp", "scp"])
    assert copy[0]["cwd"] == ds.parent and "lerobot" in " ".join(copy[0]["args"])
    assert copy[1]["cwd"] == tmp_path / "out"
    if not have_rsync:  # scp copies into an existing directory, so the stale remote copy is removed first
        assert any("rm -rf ~/cortex/jobs/tj_1/dataset" in " ".join(c["args"]) for c in calls)


async def test_small_files_go_through_ssh_stdin_with_unix_line_endings(monkeypatch):
    calls = []
    monkeypatch.setattr(ts, "_sh", _record(calls))
    await ts._ship_text("futurama", "~/j/train_act.sbatch", "#!/bin/bash\r\necho hi\r\n", "job script")
    (c,) = calls
    assert c["args"][0] == "ssh" and c["args"][-1] == "cat > ~/j/train_act.sbatch"
    assert c["input"] == b"#!/bin/bash\necho hi\n"


def test_preflight_names_the_missing_tool(monkeypatch):
    monkeypatch.setattr(ts.shutil, "which", lambda t: None)
    with pytest.raises(ValueError, match="ssh"):
        REAL_PREFLIGHT()
    monkeypatch.setattr(ts.shutil, "which", lambda t: None if t in ("rsync", "scp") else "/usr/bin/ssh")
    with pytest.raises(ValueError, match="rsync or scp"):
        REAL_PREFLIGHT()
    monkeypatch.setattr(ts.shutil, "which", lambda t: None if t == "rsync" else f"/x/{t}")
    REAL_PREFLIGHT()  # scp alone is enough


async def test_real_subprocess_receives_stdin_and_cwd(tmp_path):
    rc, out = await ts._sh("sh", "-c", "pwd; cat", cwd=tmp_path, input=b"payload")
    assert rc == 0 and out.splitlines() == [str(tmp_path), "payload"]
    assert asyncio.get_running_loop()  # ran on the test loop


def test_remote_paths_are_quoted_and_rm_only_hits_a_job_dir():
    assert ts._remote_q("~/cortex/jobs/tj_1/dataset") == "~/cortex/jobs/tj_1/dataset"
    assert ts._remote_q("~/my cortex/jobs/tj_1") == "~/'my cortex/jobs/tj_1'"
    assert ts._remote_q("/srv/x; rm -rf /") == "'/srv/x; rm -rf /'"
    assert ts._safe_job_dir("~/cortex/jobs/tj_1/dataset")
    for bad in ("~/", "", "~/cortex", "/jobs", "~/cortex/jobs/../..", "~/x/y"):
        with pytest.raises(ValueError, match="not a job directory"):
            ts._safe_job_dir(bad)
