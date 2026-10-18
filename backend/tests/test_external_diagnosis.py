"""Failure diagnosis by the MCP client's own model: inputs, strict validation of what comes back,
provenance on the stored hypotheses, and pick-up by the normal cluster pipeline. Keyframe targets
are stubbed (no simulator run); everything else is the real backend."""

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.errors import UserError
from cortex.models import Analysis, CreateProjectRequest
from cortex.services import agent_orchestrator as orch
from cortex.services import analysis_service
from cortex.services import external_diagnosis as ext

FRAMES = [np.full((96, 128, 3), 40 * i, np.uint8) for i in range(6)]
REFS = ["ev_1/br_dim/seed1000", "ev_1/br_dim/seed1001"]
GOOD = {"hypotheses": [{"category": "poor_visibility", "confidence": 0.7, "rationale": "cube is in deep shadow",
                        "frame_indices": [0, 1]}]}


@pytest.fixture
def stub_targets(monkeypatch):
    async def targets(ctx, eps, failures, max_vlm):
        return [(r, FRAMES, {"outcome": "failure", "condition": "dim", "cube_max_height_m": 0.0})
                for r in REFS][:max_vlm]

    monkeypatch.setattr(analysis_service, "vlm_targets", targets)


async def setup(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the box"))
    return ctx, p


def submit(p, an_id, answers, route="mcp_sampling", model="claude-sonnet-test", client="inspector 1.0"):
    return ext.SubmitArgs(project_id=p.id, analysis_id=an_id, route=route, model=model, client=client,
                          answers=answers)


async def test_inputs_carry_the_contact_sheet_telemetry_and_fixed_prompt(tmp_path, stub_targets):
    ctx, p = await setup(tmp_path)
    out = await orch.call_tool(ctx, "get_diagnosis_inputs", {"project_id": p.id}, "mcp")
    assert [t["ref_id"] for t in out["targets"]] == REFS and set(out["taxonomy"]) == set(ext.vlm.TAXONOMY)
    t = out["targets"][0]
    assert base64.b64decode(t["image_b64"])[:2] == b"\xff\xd8" and t["mime_type"] == "image/jpeg"
    assert '"cube_max_height_m": 0.0' in t["prompt"] and "- grasp_miss:" in t["prompt"]
    assert t["prompt"] == ext.prompt_for(p.spec.goal, t["telemetry"])  # same prompt as the model route
    an = await ctx.repo.get("analyses", out["analysis_id"], Analysis)
    assert an.status == "running" and [d.error for d in an.diagnoses] == [ext.PENDING] * 2
    assert await analysis_service.latest_analysis(ctx, p.id) is None  # nothing published until answers arrive


async def test_sampling_answers_are_stored_as_unverified_hypotheses_with_provenance(tmp_path, stub_targets):
    ctx, p = await setup(tmp_path)
    an_id = (await ext.get_inputs(ctx, p, 4, "mcp"))["analysis_id"]
    reply = "Sure, here is my analysis:\n```json\n" + json.dumps(GOOD) + "\n```"
    out = await ext.submit(ctx, submit(p, an_id, [ext.TargetAnswer(ref_id=REFS[0], reply_text=reply)]), "mcp")

    an = await analysis_service.latest_analysis(ctx, p.id)
    assert an.id == an_id and an.status == "succeeded"
    assert an.vlm_provider == "MCP sampling: claude-sonnet-test (client-reported)"
    d0, d1 = an.diagnoses
    assert (d0.route, d0.model, d0.client, d0.status) == ("mcp_sampling", "claude-sonnet-test", "inspector 1.0",
                                                          "hypothesis")
    assert [(h.category, h.confidence, h.frame_indices) for h in d0.hypotheses] == [("poor_visibility", 0.7, [0, 1])]
    assert d1.hypotheses == [] and d1.error == ext.UNANSWERED  # recorded, never filled in

    clusters = await analysis_service.current_clusters(ctx, p.id)
    vlm = [c for c in clusters if c.source == "vlm"]
    assert [(c.category, c.episode_ids, c.verified_by_human) for c in vlm] == [
        ("vlm:poor_visibility", [REFS[0]], False)]
    assert vlm[0].uncertainty == pytest.approx(0.3)
    assert out["vlm_clusters"][0]["id"] == vlm[0].id and "unverified" in out["note"]


async def test_agent_route_takes_structured_hypotheses_and_client_errors(tmp_path, stub_targets):
    ctx, p = await setup(tmp_path)
    an_id = (await ext.get_inputs(ctx, p, 4, "mcp"))["analysis_id"]
    await ext.submit(ctx, submit(p, an_id, [
        ext.TargetAnswer(ref_id=REFS[0], hypotheses=GOOD["hypotheses"]),
        ext.TargetAnswer(ref_id=REFS[1], error="user declined the sampling request"),
    ], route="mcp_agent", client=None), "mcp")
    an = await ctx.repo.get("analyses", an_id, Analysis)
    assert [d.route for d in an.diagnoses] == ["mcp_agent", "mcp_agent"]
    assert an.diagnoses[1].error == "user declined the sampling request" and an.diagnoses[1].hypotheses == []


@pytest.mark.parametrize("bad,match", [
    ({"hypotheses": [{"category": "made_up", "confidence": 0.9, "rationale": "x"}]}, "not in the failure taxonomy"),
    ({"hypotheses": [{"category": "grasp_miss", "confidence": 1.5, "rationale": "x"}]}, "less than or equal to 1"),
    ({"hypotheses": [{"category": "grasp_miss", "confidence": 0.5, "rationale": "x", "frame_indices": [9]}]},
     "outside the 6 keyframes"),
    ({"hypotheses": []}, "at least 1 item"),
    ({"hypotheses": [{"category": "grasp_miss", "confidence": 0.5}]}, "rationale: Field required"),
    ("I think the gripper missed.", "no JSON object"),
    ("{hypotheses: [grasp_miss]}", "not valid JSON"),
])
async def test_bad_model_output_is_rejected_and_nothing_is_stored(tmp_path, stub_targets, bad, match):
    ctx, p = await setup(tmp_path)
    an_id = (await ext.get_inputs(ctx, p, 4, "mcp"))["analysis_id"]
    text = bad if isinstance(bad, str) else json.dumps(bad)
    answers = [ext.TargetAnswer(ref_id=REFS[0], reply_text=json.dumps(GOOD)),  # one good answer is not enough
               ext.TargetAnswer(ref_id=REFS[1], reply_text=text)]
    with pytest.raises(UserError, match=match) as e:
        await ext.submit(ctx, submit(p, an_id, answers), "mcp")
    assert "nothing was stored" in str(e.value) and REFS[1] in str(e.value)
    an = await ctx.repo.get("analyses", an_id, Analysis)
    assert an.status == "running" and all(d.error == ext.PENDING for d in an.diagnoses)
    assert await analysis_service.current_clusters(ctx, p.id) == []


async def test_answers_must_match_a_pending_analysis(tmp_path, stub_targets):
    ctx, p = await setup(tmp_path)
    an_id = (await ext.get_inputs(ctx, p, 4, "mcp"))["analysis_id"]
    with pytest.raises(UserError, match="not a target"):
        await ext.submit(ctx, submit(p, an_id, [ext.TargetAnswer(ref_id="ep_invented", reply_text="{}")]), "mcp")
    with pytest.raises(UserError, match="answered twice"):
        await ext.submit(ctx, submit(p, an_id, [ext.TargetAnswer(ref_id=REFS[0], reply_text=json.dumps(GOOD))] * 2),
                         "mcp")
    with pytest.raises(KeyError):
        await ext.submit(ctx, submit(p.model_copy(update={"id": "prj_other"}), an_id,
                                     [ext.TargetAnswer(ref_id=REFS[0], reply_text=json.dumps(GOOD))]), "mcp")
    ok = [ext.TargetAnswer(ref_id=REFS[0], reply_text=json.dumps(GOOD))]
    await ext.submit(ctx, submit(p, an_id, ok), "mcp")
    with pytest.raises(UserError, match="not waiting for hypotheses"):  # no second write over stored hypotheses
        await ext.submit(ctx, submit(p, an_id, ok), "mcp")


def test_answer_shape_and_provenance_are_required():
    with pytest.raises(ValueError, match="exactly one of"):
        ext.TargetAnswer(ref_id="x", reply_text="{}", error="also failed")
    with pytest.raises(ValueError, match="exactly one of"):
        ext.TargetAnswer(ref_id="x")
    base = {"project_id": "p", "analysis_id": "a", "answers": [{"ref_id": "x", "error": "e"}]}
    with pytest.raises(ValueError, match="model"):
        ext.SubmitArgs(**base, route="mcp_sampling")
    with pytest.raises(ValueError, match="route"):
        ext.SubmitArgs(**base, route="openrouter", model="m")  # only the MCP routes come in this way


async def test_no_targets_completes_the_analysis_without_vlm(tmp_path):
    ctx, p = await setup(tmp_path)
    out = await ext.get_inputs(ctx, p, 4, "mcp")
    assert out["targets"] == []
    an = await analysis_service.latest_analysis(ctx, p.id)
    assert an.id == out["analysis_id"] and an.vlm_provider is None and an.diagnoses == []


def test_tools_are_for_mcp_clients_not_the_studio_agent(tmp_path, monkeypatch, stub_targets):
    assert not orch.TOOLS["get_diagnosis_inputs"].agent_callable
    assert not orch.TOOLS["submit_failure_hypotheses"].agent_callable
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        pid = c.post("/api/tools/create_project", json={"task_prompt": "pick up the cube"}).json()["ref_ids"][0]
        an_id = c.post("/api/tools/get_diagnosis_inputs", json={"project_id": pid}).json()["analysis_id"]
        bad = {"project_id": pid, "analysis_id": an_id, "route": "mcp_sampling", "model": "m",
               "answers": [{"ref_id": REFS[0], "reply_text": json.dumps({"hypotheses": [{"category": "nope"}]})}]}
        r = c.post("/api/tools/submit_failure_hypotheses", json=bad)
        assert r.status_code == 400 and "nothing was stored" in r.json()["detail"]
        r = c.post("/api/tools/submit_failure_hypotheses", json={**bad, "route": "insforge"})
        assert r.status_code == 422
        r = c.post("/api/tools/submit_failure_hypotheses", json={**bad, "answers": [
            {"ref_id": REFS[0], "reply_text": json.dumps(GOOD)}]})
        assert r.status_code == 200 and r.json()["diagnoses"][0]["route"] == "mcp_sampling"
        r = c.post("/api/tools/get_diagnosis_inputs", json={"project_id": pid}, headers={"x-cortex-actor": "agent"})
        assert r.status_code == 403 and r.json()["code"] == "not_agent_callable"
    settings.cache_clear()
