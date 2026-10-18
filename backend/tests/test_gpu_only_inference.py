"""#290: a trained policy never runs on this Mac. Learned-policy inference refuses anywhere but a CUDA GPU, with a
plain sentence every caller shows as-is; the scripted oracle (no network) is unaffected. Checkpoints here are fakes
(a config.json): the refusal comes before any weights are read."""

import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.errors import UserError
from cortex.models import Checkpoint, CreateProjectRequest, EvaluationRun, JobStatus, SimulationBranch
from cortex.services import agent_orchestrator as orch
from cortex.sim import policies

GPU_ONLY = "Trained policies run on the GPU cluster, not this Mac: start the GPU server from the Inference tab"


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.delenv(policies.CPU_FOR_TESTS, raising=False)
    monkeypatch.setattr(policies, "has_cuda", lambda: False)


def fake_checkpoint(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"type": "act", "input_features": {}}))
    return d


def test_the_message_is_the_one_people_see():
    assert policies.GPU_ONLY == GPU_ONLY


def test_a_trained_policy_refuses_without_a_cuda_gpu_before_reading_its_files(tmp_path, no_gpu):
    with pytest.raises(UserError, match=GPU_ONLY):
        policies.LeRobotPolicy(tmp_path / "not_even_there")
    with pytest.raises(UserError, match=GPU_ONLY):
        policies.load_policy("checkpoint", str(fake_checkpoint(tmp_path / "ck")))


def test_cpu_is_refused_even_on_a_gpu_machine(monkeypatch):
    monkeypatch.delenv(policies.CPU_FOR_TESTS, raising=False)
    monkeypatch.setattr(policies, "has_cuda", lambda: True)
    assert policies.inference_device() == "cuda" and policies.inference_device("cuda:1") == "cuda:1"
    for device in ("cpu", "mps"):
        with pytest.raises(UserError, match=GPU_ONLY):
            policies.inference_device(device)


def test_this_mac_never_counts_as_a_gpu(monkeypatch):
    monkeypatch.setattr(policies.sys, "platform", "darwin")
    assert policies.has_cuda() is False


def test_only_the_explicit_test_override_allows_cpu(monkeypatch, no_gpu):
    monkeypatch.setenv(policies.CPU_FOR_TESTS, "1")
    assert policies.inference_device() == "cpu"


def test_the_scripted_oracle_needs_no_gpu(no_gpu):
    assert isinstance(policies.load_policy("scripted_expert", None), policies.ExpertPolicy)


def test_the_real_arm_refuses_a_policy_before_touching_the_arm(tmp_path, no_gpu):
    from cortex.hardware import arm_worker as aw

    with pytest.raises(UserError, match=GPU_ONLY):
        aw.PolicySource(str(fake_checkpoint(tmp_path / "ck")), ["front"])


async def test_an_evaluation_of_a_checkpoint_fails_with_the_reason_not_a_traceback(tmp_path, monkeypatch):
    """Real sim workers: each refuses at policy load (before building the scene), and the job, every branch and the
    agent's tool summary carry the plain sentence."""
    monkeypatch.delenv(policies.CPU_FOR_TESTS, raising=False)  # the workers inherit this environment
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ck = Checkpoint(project_id=p.id, training_job_id=None, policy_type="act", step=1,
                    path=str(fake_checkpoint(tmp_path / "ck")))
    await ctx.repo.save("checkpoints", ck)
    out = await orch.call_tool(ctx, "launch_simulation_evaluation",
                               {"project_id": p.id, "policy": ck.id, "episodes_per_branch": 1, "wait_s": 120}, "mcp")
    run = await ctx.repo.get("evaluation_runs", out["ref_ids"][0], EvaluationRun)
    branches = [await ctx.repo.get("simulation_branches", b, SimulationBranch) for b in run.branch_ids]
    job = ctx.jobs.get(run.job_id)
    assert run.status == JobStatus.failed and job.status == "failed"
    assert job.error == GPU_ONLY and job.error_detail is None
    assert [b.error for b in branches] == [GPU_ONLY] * len(branches)
    assert GPU_ONLY in out["summary"]


async def test_an_rl_round_says_why_its_re_evaluation_has_no_result(tmp_path):
    from types import SimpleNamespace

    from cortex.models import BranchSpec, Condition
    from cortex.services import rl_service

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ids = []
    for ck in ("ck_old", "ck_new"):  # both re-evaluations failed: no GPU here
        run = EvaluationRun(project_id=p.id, policy={"kind": "checkpoint", "checkpoint_id": ck},
                            status=JobStatus.failed)
        b = SimulationBranch(evaluation_id=run.id, project_id=p.id, seeds=[1], status=JobStatus.failed,
                             spec=BranchSpec(name="baseline", condition=Condition()), error=GPU_ONLY)
        run.branch_ids = [b.id]
        await ctx.repo.save("evaluation_runs", run)
        await ctx.repo.save("simulation_branches", b, project_id=p.id)
        ids.append(run.id)
    rnd = rl_service.RLRound(project_id=p.id, sandbox_id="rl_x", weakness_key="baseline", condition="center/normal",
                             skill_gain=0.5, status="reevaluating", evaluation_before=ids[0], evaluation_after=ids[1])
    await rl_service._realise(ctx, p, SimpleNamespace(weakness=SimpleNamespace(key="baseline")), rnd, "mcp")
    assert rnd.status == "retrain_proposed" and GPU_ONLY in rnd.note
