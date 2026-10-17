"""LLM coding agent for Track B (real tool calls; the LLM picks every next operation).

Timing model: each step = LLM think (critical path, GPU) + tool execution (critical
path unless its result was pre-executed and validated). In `replay` mode the LLM
response cache returns the reference run's response and the agent SLEEPS for the
reference call's recorded server time, so all scheduling variants see identical
decisions and identical think durations (paired design); only tool scheduling differs.
"""
from __future__ import annotations

import json
import time

from dreamcache.codebench.speculator import OPS
from dreamcache.codebench.tools import READ_ONLY, edit_function, execute, read_file, snapshot_hash

SYSTEM = ("You are a coding agent fixing a bug in a small Python package (pkg/core.py, tests in "
          "tests/test_core.py). Tools: read_file(file), edit(function, code) which replaces the WHOLE "
          "function definition in pkg/core.py, run_tests, lint, deps_check, finish. Typical workflow: "
          "run tests, read the code, edit the buggy function, re-run tests, finish when tests pass. "
          "Respond only with JSON.")
SCHEMA = {"type": "object", "properties": {"op": {"type": "string", "enum": OPS}, "file": {"type": "string"},
                                           "function": {"type": "string"}, "code": {"type": "string"}},
          "required": ["op"]}


def run_task(client, ws, meta, speculator, max_steps=12, replay=False):
    hist, steps = [], []
    prev_op, last_ok = "start", "na"
    t_task = time.perf_counter()
    for step in range(max_steps):
        snap = snapshot_hash(ws)
        t_spec = time.perf_counter()
        speculator.on_new_snapshot(ws, snap, prev_op, last_ok, meta["family"])
        sched_s = time.perf_counter() - t_spec
        obs = "\n".join(f"[{h['op']}] {h['out']}" for h in hist[-4:]) or "(no actions yet)"
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Task: tests are failing; find and fix the bug.\nStep {step + 1}/{max_steps}.\n"
                                            f"Recent tool outputs:\n{obs}\nChoose the next tool call."}]
        t0 = time.perf_counter()
        content, cost = client.chat(msgs, SCHEMA, 0.0, 0)
        if replay and cost.cache_hits:
            time.sleep(max(0.0, cost.llm_seconds - (time.perf_counter() - t0)))
        think_s = time.perf_counter() - t0
        try:
            d = json.loads(content)
        except ValueError:
            d = {"op": "run_tests"}
        op = d.get("op") if d.get("op") in OPS else "run_tests"
        t1 = time.perf_counter()
        info = {"critical_s": 0.0, "reused": False, "fate": "local"}
        if op in READ_ONLY:
            res, info = speculator.request(op, ws, snap)
            out, ok = res["out"], "ok" if res["rc"] == 0 else "fail"
        elif op == "read_file":
            out, ok = read_file(ws, d.get("file") or "pkg/core.py"), "ok"
        elif op == "edit":
            out = edit_function(ws, d.get("function", ""), d.get("code", ""))
            ok = "ok" if out.startswith("edited") else "fail"
        else:
            out, ok = "finished", "ok"
        tool_s = time.perf_counter() - t1
        steps.append({"step": step, "op": op, "think_s": think_s, "tool_s": tool_s, "sched_s": sched_s,
                      "t_think": t0 - t_task, "t_tool": t1 - t_task, "t_end": time.perf_counter() - t_task,
                      "fate": info["fate"], "llm_seconds": cost.llm_seconds, "gen_tokens": cost.gen_tokens,
                      "snapshot": snap, "prev_op": prev_op, "last_ok": last_ok, "family": meta["family"]})
        hist.append({"op": op, "out": out[-600:]})
        prev_op, last_ok = op, ok
        if op == "finish":
            break
    wall = time.perf_counter() - t_task
    final = execute("run_tests", ws)   # correctness check, outside the timed region
    return {"task": meta["task"], "family": meta["family"], "success": final["rc"] == 0, "wall_s": wall,
            "t0": t_task,
            "steps": steps, "n_steps": len(steps)}
