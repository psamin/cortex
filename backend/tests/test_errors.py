import asyncio
import json

import pytest

from cortex.errors import INTERNAL, UserError, WorkerFailed, describe, friendly
from cortex.services.agent import tidy_job_rows
from cortex.services.jobs import EventBus, JobManager


def raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    raise AssertionError("did not raise")


def cortex_value_error():
    import tempfile
    from pathlib import Path

    from cortex.services.import_service import find_dataset_root

    find_dataset_root(Path(tempfile.mkdtemp()))  # raises ValueError on purpose, inside Cortex code


NO_DATASET = "no LeRobot dataset found (expected a folder containing meta/info.json)"


def test_library_errors_are_internal_and_cortex_messages_pass_through():
    lib = raised(lambda: json.loads("{not json"))  # a ValueError raised inside a library
    assert describe(lib) == (INTERNAL, describe(lib)[1]) and "JSONDecodeError" in describe(lib)[1]
    assert describe(raised(cortex_value_error)) == (NO_DATASET, None)
    assert describe(UserError("upload a .zip or .tar.gz")) == ("upload a .zip or .tar.gz", None)
    assert describe(RuntimeError("boom"))[0] == INTERNAL
    msg, detail = describe(ConnectionError("Could not connect on port '/dev/tty.usbmodem1'. Make sure..."))
    assert msg.startswith("Couldn't reach the arm on /dev/tty.usbmodem1") and detail


def test_worker_failures_keep_the_workers_verdict():
    assert describe(WorkerFailed("not a manipulation dataset", user_facing=True)) == ("not a manipulation dataset", None)
    msg, detail = describe(WorkerFailed("ValueError: The truth value of a DataFrame is ambiguous."))
    assert msg == INTERNAL and detail == "ValueError: The truth value of a DataFrame is ambiguous."


def test_stored_text_is_cleaned_too():
    raw = "RuntimeError: ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool()."
    assert friendly(raw) == (INTERNAL, raw)
    assert friendly("interrupted by a backend restart")[0] == "It stopped because Cortex restarted. Run it again."
    assert friendly("no LeRobot dataset found (expected a folder containing meta/info.json)") == \
        ("no LeRobot dataset found (expected a folder containing meta/info.json)", None)
    assert friendly("RuntimeError: UserError: bad archive") == ("bad archive", None)


def test_old_chat_rows_are_rewritten_and_superseded_failures_marked():
    rows = [{"kind": "job_done", "job_kind": "import", "status": "failed",
             "text": "Demonstration import failed: RuntimeError: ValueError: The truth value of a DataFrame is "
                     "ambiguous."},
            {"kind": "job_done", "job_kind": "train", "status": "failed", "text": "Training failed: x",
             "detail": None},
            {"kind": "job_done", "job_kind": "import", "status": "succeeded", "text": "Demonstration import finished"}]
    out = tidy_job_rows(rows)
    assert out[0]["text"] == f"Demonstration import failed: {INTERNAL}"
    assert "DataFrame" in out[0]["detail"] and out[0]["superseded"] is True
    assert out[1]["superseded"] is False  # no later training succeeded


@pytest.mark.parametrize("fn,expected", [(lambda: json.loads("{bad"), INTERNAL),
                                         (cortex_value_error, NO_DATASET)])
def test_failed_jobs_carry_a_readable_error(tmp_path, fn, expected):
    async def run():
        jm = JobManager(EventBus())

        async def job_fn(job):
            fn()
        j = jm.submit("dataset", "prj_1", job_fn)
        await asyncio.sleep(0.05)
        return j
    j = asyncio.run(run())
    assert j.status == "failed" and j.error == expected
    assert (j.error_detail is None) == (expected != INTERNAL)
