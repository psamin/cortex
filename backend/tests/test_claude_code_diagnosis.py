"""Claude Code failure diagnosis against a fake `claude` executable (a Python script that answers
like `claude -p --output-format json`). No real Claude Code call is made."""

import asyncio
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, InfraChoice
from cortex.services import agent_orchestrator as orch
from cortex.services import analysis_service
from cortex.services import claude_code_agent as cc
from cortex.services import claude_code_diagnosis as ccd

FRAMES = [np.full((96, 128, 3), 40 * i, np.uint8) for i in range(6)]
TEL = {"outcome": "failure", "condition": "shifted_object", "cube_max_height_m": 0.0}

# Answers only if the contact sheet named in the prompt exists in its cwd (what Read would open).
FAKE = r'''
import json, os, re, sys
prompt = sys.argv[sys.argv.index("-p") + 1]
image = re.search(r"\./(\S+\.jpg)", prompt).group(1)
assert os.path.getsize(image) > 0, image
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "crash":
    sys.stderr.write("auth required"); sys.exit(1)
if mode == "hang":
    import time; time.sleep(30)
reply = {"hypotheses": [{"category": "grasp_miss", "confidence": 0.8, "rationale": "jaws close beside the cube",
                         "frame_indices": [2, 3]}, {"category": "made_up", "confidence": 0.9, "rationale": "x"}]}
print(json.dumps({"type": "result", "subtype": "max_turns" if mode == "turns" else "success",
                  "is_error": False, "result": "Here you go:\n" + json.dumps(reply)}))
'''


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE)
    monkeypatch.setattr(ccd, "command", lambda prompt, model=None: [sys.executable, str(script), "-p", prompt])
    return script


def run(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", mode)
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    return asyncio.run(ccd.diagnose("ev_1/br_1/seed1000", FRAMES, "pick up the cube", TEL, workdir, "target00"))


def test_hypotheses_come_back_as_unverified_diagnoses(fake_claude, tmp_path, monkeypatch):
    d = run("ok", tmp_path, monkeypatch)
    assert d.error is None and d.model == "claude-code" and d.status == "hypothesis"
    assert [h.category for h in d.hypotheses] == ["grasp_miss"]  # off-taxonomy answers are dropped
    assert d.hypotheses[0].frame_indices == [2, 3]


@pytest.mark.parametrize("mode,match", [("crash", "exited 1: auth required"), ("turns", "max_turns")])
def test_failures_are_recorded_not_invented(fake_claude, tmp_path, monkeypatch, mode, match):
    d = run(mode, tmp_path, monkeypatch)
    assert d.hypotheses == [] and match in d.error


def test_a_hung_run_times_out(fake_claude, tmp_path, monkeypatch):
    monkeypatch.setattr(ccd, "TIMEOUT_S", 1)
    d = run("hang", tmp_path, monkeypatch)
    assert d.hypotheses == [] and "did not answer within 1 s" in d.error


def test_the_command_allows_only_read_and_no_mcp():
    cmd = ccd.command("look", model="sonnet")
    assert cmd[1:3] == ["-p", "look"]
    assert cmd[cmd.index("--allowedTools") + 1] == "Read"
    assert "--strict-mcp-config" in cmd and "--mcp-config" not in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json" and cmd[-2:] == ["--model", "sonnet"]


def route(monkeypatch, planner, cli_ok, model=None, engine="auto"):
    monkeypatch.setattr(cc, "available", lambda: None if cli_ok else "the `claude` CLI is not on PATH")
    ctx = SimpleNamespace(settings=SimpleNamespace(agent_engine=engine), model=model, vlm_source="OpenRouter")
    return ccd.diagnoser(ctx, SimpleNamespace(infra=InfraChoice(planner=planner)))[0]


def test_routing_follows_the_projects_planner_choice(monkeypatch):
    m = object()
    assert route(monkeypatch, "auto", True, m) == "claude_code"
    assert route(monkeypatch, "auto", False, m) == "model"
    assert route(monkeypatch, "auto", False) is None
    assert route(monkeypatch, "model", True, m) == "model"
    assert route(monkeypatch, "model", True) is None
    assert route(monkeypatch, "claude_code", False, m) is None  # chosen but missing: skipped, not substituted
    assert route(monkeypatch, "auto", True, m, engine="model") == "model"


async def test_analysis_says_why_diagnosis_was_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "available", lambda: "the `claude` CLI is not on PATH")
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    p.infra = InfraChoice(planner="claude_code")
    job = await analysis_service.start_analysis(ctx, p)
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job.id]), 10)
    an = await analysis_service.latest_analysis(ctx, p.id)
    assert an.vlm_provider is None
    assert "VLM skipped: Claude Code was chosen but the `claude` CLI is not on PATH" in an.summary
