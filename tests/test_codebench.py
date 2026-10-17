"""Track B: isolation, validation, stale-snapshot invalidation, and reuse (scripted LLM)."""
import json
import os

from dreamcache.codebench.agent import run_task
from dreamcache.codebench.speculator import OpPredictor, Speculator
from dreamcache.codebench.tasks import TEMPLATES, make_task
from dreamcache.codebench.tools import execute, fingerprint, snapshot_hash
from dreamcache.types import Cost


class ScriptedClient:
    def __init__(self, fix):
        self.ops = [{"op": "run_tests"}, {"op": "read_file", "file": "pkg/core.py"},
                    {"op": "edit", "function": "clamp", "code": fix}, {"op": "run_tests"}, {"op": "finish"}]
        self.i = 0

    def chat(self, msgs, schema, temperature, seed):
        d = self.ops[min(self.i, len(self.ops) - 1)]
        self.i += 1
        return json.dumps(d), Cost(llm_calls=1, llm_seconds=0.0)


def test_task_has_failing_test_and_fix_passes(tmp_path):
    ws = str(tmp_path / "ws")
    meta = make_task(ws, "clamp", 0, "eval")
    assert execute("run_tests", ws)["rc"] != 0
    fp = fingerprint()
    rows = [{"prev_op": "edit", "last_ok": "ok", "family": "clamp", "op": "run_tests"}] * 5
    sp = Speculator("top1", OpPredictor("personalized").fit(rows), fp=fp)
    r = run_task(ScriptedClient(TEMPLATES["clamp"][0]), ws, meta, sp)
    st = sp.finalize()
    assert r["success"]
    # the post-edit run_tests was predicted from history and reused after validation
    assert any(s["op"] == "run_tests" and s["fate"].startswith("reused") for s in r["steps"]), r["steps"]
    assert st["launched"] >= 1


def test_stale_snapshot_never_reused(tmp_path):
    ws = str(tmp_path / "ws")
    make_task(ws, "mean", 0, "eval")
    sp = Speculator("all", None, fp="fp")
    snap0 = snapshot_hash(ws)
    sp.on_new_snapshot(ws, snap0, "start", "na", "mean")
    with open(os.path.join(ws, "pkg", "core.py"), "a") as f:   # workspace changes afterwards
        f.write("\n# changed\n")
    snap1 = snapshot_hash(ws)
    assert snap1 != snap0
    _, info = sp.request("run_tests", ws, snap1)
    assert not info["reused"]          # speculative result for snap0 is not valid for snap1
    st = sp.finalize()
    assert st.get("unused", 0) >= 1    # counted as wasted speculative work
