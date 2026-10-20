"""RL 4/5, closing the loop: sandbox demos -> a capped dataset version -> a proposed (never approved) retrain ->
paired re-evaluation of every condition -> predicted vs realised uplift -> the refitted transfer curve.

The dataset build runs the real LeRobot writer on small synthetic episodes; the two re-evaluations are recorded
results (no ACT checkpoint is trained here), so the paired statistics are checked on known outcomes."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex.models import (
    BranchResult,
    BranchSpec,
    Checkpoint,
    Condition,
    DatasetEpisodeLink,
    DatasetVersion,
    Episode,
    EpisodeResult,
    EpisodeSource,
    EpisodeStatus,
    EvaluationRun,
    JobStatus,
    PolicyRef,
    SimulationBranch,
)
from cortex.rl import SandboxBudget, Weakness, propose
from cortex.rl.proposals import MIN_ROUNDS, transfer_fit
from cortex.rl.rounds import MAX_TARGETED_SHARE, needs_human_demos, paired_compare, select_targeted
from cortex.rl.types import SandboxResult
from cortex.services.simulation_runner import wilson


# ------------------------------------------------------------------ pure helpers
def ep(i: int, planner: str = "scripted_ik_oracle", cond: Condition = Condition()) -> Episode:
    return Episode(id=f"ep_{planner[:3]}{i}", project_id="prj", index=i, source=EpisodeSource.sim_scripted,
                   condition=cond, seed=i, num_frames=60, fps=30, success=True, status=EpisodeStatus.accepted,
                   provenance={"planner": planner})


def test_targeted_demos_are_capped_on_the_train_split():
    base = [ep(i) for i in range(20)]
    new = [ep(i, "rl_sandbox:rl_a") for i in range(30)]
    chosen, info = select_targeted(base, new, eval_fraction=0.2, split_seed=0)
    assert 0 < len(chosen) < 30 and info["added"] == len(chosen) and info["offered"] == 30
    assert info["targeted_share"] <= MAX_TARGETED_SHARE and info["cap"] == pytest.approx(0.4)
    assert chosen == new[:len(chosen)]
    # earlier rounds' targeted demos already in the dataset count towards the same cap: round 1 left little room
    more = [ep(i, "rl_sandbox:rl_b") for i in range(100, 110)]  # distinct ids from round 1's
    again, info2 = select_targeted(base + chosen, more, 0.2, 0)
    assert len(again) < len(more) and info2["targeted_share"] <= MAX_TARGETED_SHARE
    assert info2["targeted_train_episodes"] >= info["targeted_train_episodes"]
    assert select_targeted([], new[:3], 0.2, 0)[0] == []  # nothing to dilute into: never an all-targeted dataset


def branch(name: str, outcomes: list[bool], seed0: int = 900_000, offset=(0.0, 0.0)) -> SimulationBranch:
    eps = [EpisodeResult(seed=seed0 + i, success=s, reward=float(s), steps=200,
                         failure_mode=None if s else "grasp_failed") for i, s in enumerate(outcomes)]
    k = sum(outcomes)
    res = BranchResult(n=len(eps), successes=k, success_rate=k / len(eps), ci95=wilson(k, len(eps)), mean_reward=0.5,
                       episodes=eps)
    return SimulationBranch(evaluation_id="ev", spec=BranchSpec(name=name, condition=Condition(), cube_offset=offset),
                            seeds=[e.seed for e in eps], result=res)


def test_paired_comparison_flags_improvements_and_regressions():
    T, F = True, False
    old = [branch("baseline", [T] * 9 + [F]), branch("shifted_object", [T] * 3 + [F] * 7), branch("dim", [T] * 6 + [F] * 4)]
    new = [branch("baseline", [T] * 2 + [F] * 8), branch("shifted_object", [T] * 9 + [F]), branch("dim", [F] + [T] * 6 + [F] * 3)]
    c = {x.name: x for x in paired_compare(old, new, target_name="shifted_object")}
    assert c["shifted_object"].is_target and c["shifted_object"].verdict == "improved"
    assert c["shifted_object"].delta == pytest.approx(0.6) and c["shifted_object"].discordant == (0, 6)
    assert c["baseline"].verdict == "regressed" and c["baseline"].p_value < 0.05
    assert c["dim"].verdict == "no_significant_change" and not c["dim"].possible_regression
    assert all(x.kind == "measured" and x.n == 10 for x in c.values())
    # seeds are matched, never imputed: an unmatched seed is left out
    c2 = paired_compare([branch("baseline", [T] * 4)], [branch("baseline", [T] * 3, seed0=900_001)])
    assert c2[0].n == 3 and c2[0].seeds == [900_001, 900_002, 900_003]


def test_round_stopping_rule_needs_two_failed_rounds_in_a_row():
    assert not needs_human_demos(["no_significant_change"])
    assert needs_human_demos(["no_significant_change", "regressed"])
    assert not needs_human_demos(["no_significant_change", "improved"])
    assert not needs_human_demos(["no_significant_change", "improved", "no_significant_change"])


def test_transfer_curve_switches_from_prior_to_learned():
    assert transfer_fit([])["source"] == "prior"
    hist = [(0.3, 0.15), (0.4, 0.1), (0.2, 0.1)]
    assert transfer_fit(hist[:MIN_ROUNDS - 1])["source"] == "prior"
    fit = transfer_fit(hist)
    assert fit["source"] == "learned" and fit["rounds_used"] == 3 and fit["kind"] == "estimate"
    assert fit["t_mean"] == pytest.approx(np.mean([0.5, 0.25, 0.5]), abs=0.01)
    assert fit["t_ci90"][0] <= fit["t_mean"] <= fit["t_ci90"][1]


# ------------------------------------------------------------------ the service, end to end over the registry
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def ctx():
    from cortex.api import app as appmod

    return appmod.CTX


def tool(c, name, status=200, **args):
    r = c.post(f"/api/tools/{name}", json=args, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == status, r.text
    return r.json()


def write_npz(path, n: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    act = np.linspace(np.zeros(6), np.r_[np.full(5, 20.0), 40.0], n).astype(np.float32)
    np.savez_compressed(path / "episode.npz",
                        front=rng.integers(60, 180, (n, 96, 128, 3), dtype=np.uint8),
                        wrist=rng.integers(60, 180, (n, 96, 128, 3), dtype=np.uint8),
                        state=act, action=act, reward=np.zeros(n, np.float32),
                        timestamp=(np.arange(n) / 30).astype(np.float32),
                        action_low=np.r_[np.full(5, -180.0), 0].astype(np.float32),
                        action_high=np.r_[np.full(5, 180.0), 100].astype(np.float32))


def seed_project(c, n_base: int = 10, n_demos: int = 8):
    """A project with a trained checkpoint ck_old evaluated in sim (weak on shifted_object), its dataset version,
    and a finished sandbox with n_demos recorded demos and its skill_measured round."""
    from cortex.services.rl_service import RLRound, demo_dir

    pid = tool(c, "create_project", task_prompt="Pick up the red cube and place it in the green box")["project"]["id"]
    run = EvaluationRun(project_id=pid, policy=PolicyRef(kind="checkpoint", checkpoint_id="ck_old"),
                        status=JobStatus.succeeded)
    T, F = True, False
    branches = [branch("baseline", [T] * 9 + [F], 1000), branch("shifted_object", [T] * 3 + [F] * 7, 1000, (0.02, 0))]
    for b in branches:
        b.evaluation_id = run.id
    run.branch_ids = [b.id for b in branches]
    w = Weakness(key="shifted_object", branch_id=branches[1].id, evaluation_id=run.id, checkpoint_id="ck_old",
                 condition=Condition(), cube_offset=(0.02, 0.0), n=10, successes=3, success_rate=0.3,
                 ci95=wilson(3, 10), target=0.8, gap=0.5, dominant_failure="grasp_failed", score=0.3)
    sb = propose(pid, [w], SandboxBudget(demos=n_demos))[0]
    sb.status = "succeeded"
    sb.result = SandboxResult(heldout_seeds=list(range(5)), base_success=0.2, base_ci95=wilson(1, 5),
                              tuned_success=0.8, tuned_ci95=wilson(4, 5), params={},
                              demo_episode_ids=[f"{k:04d}" for k in range(n_demos)])
    base_eps = []
    for i in range(n_base):
        e = ep(i)
        e.project_id = pid
        e.artifacts = {"episode": f"projects/{pid}/episodes/{e.id}/episode.npz"}
        write_npz(ctx().local(f"projects/{pid}/episodes/{e.id}"), seed=i)
        base_eps.append(e)
    for k in range(n_demos):
        d = ctx().local(demo_dir(sb, f"{k:04d}"))
        write_npz(d, seed=100 + k)
        (d / "result.json").write_text(json.dumps({
            "seed": 700_001 + k, "success": True, "num_frames": 60, "fps": 30, "dart": k % 4 == 3,
            "condition": Condition().model_dump(),
            "provenance": {"planner": f"rl_sandbox:{sb.id}", "mode": "privileged", "method": "cem_skill",
                           "dart": k % 4 == 3, "params": {}}}))
    ds = DatasetVersion(project_id=pid, name="v1", links=[
        DatasetEpisodeLink(episode_id=e.id, split="train", lerobot_episode_index=i) for i, e in enumerate(base_eps)])
    rnd = RLRound(project_id=pid, sandbox_id=sb.id, weakness_key=w.key, condition=w.condition.key, skill_gain=0.6,
                  predicted_policy_uplift=0.12)

    async def save():
        r = ctx().repo
        await r.save("evaluation_runs", run)
        for b in branches:
            await r.save("simulation_branches", b, project_id=pid)
        for e in base_eps:
            await r.save("episodes", e)
        await r.save("dataset_versions", ds, status="ready")
        await r.save("rl_sandboxes", sb)
        await r.save("rl_rounds", rnd)
        await r.save("checkpoints", Checkpoint(id="ck_old", project_id=pid, training_job_id=None, policy_type="act",
                                               step=20_000, path="ck_old"))
    c.portal.call(save)
    return pid, sb, ds


def fake_evaluations(monkeypatch):
    """Recorded paired results instead of running ACT: ck_old is the seeded policy, the new checkpoint fixes
    shifted_object (3 -> 9 of 10) and regresses on baseline (9 -> 2)."""
    from cortex.services import rl_service

    launched = []
    T, F = True, False
    outcome = {"ck_old": {"baseline": [T] * 9 + [F], "shifted_object": [T] * 3 + [F] * 7},
               "new": {"baseline": [T] * 2 + [F] * 8, "shifted_object": [T] * 9 + [F]}}

    async def launch(c, project, req, actor="gui"):
        ck = req.policy.checkpoint_id
        run = EvaluationRun(project_id=project.id, policy=req.policy, status=JobStatus.succeeded, name=req.name or "")
        bs = []
        for spec in req.branches:
            got = outcome["ck_old" if ck == "ck_old" else "new"][spec.name][:req.episodes_per_branch]
            b = branch(spec.name, got, req.base_seed, tuple(spec.cube_offset))
            b.evaluation_id, b.project_id = run.id, project.id
            bs.append(b)
        run.branch_ids = [b.id for b in bs]
        await c.repo.save("evaluation_runs", run)
        for b in bs:
            await c.repo.save("simulation_branches", b, project_id=project.id)
        launched.append(req)
        return run

    monkeypatch.setattr(rl_service.simulation_runner, "launch_evaluation", launch)
    return launched


def test_close_the_loop_end_to_end(client, monkeypatch):
    from cortex.models import TrainingJob
    from cortex.services import dataset_service
    from cortex.services.recording_service import list_episodes

    pid, sb, base_ds = seed_project(client)
    launched = fake_evaluations(monkeypatch)

    # 1) demos -> a capped new dataset version -> 2) the retrain is proposed, never approved
    out = tool(client, "close_rl_round", project_id=pid, sandbox_id=sb.id, wait_s=240)
    rnd = out["round"]
    assert rnd["status"] == "retrain_proposed", rnd
    assert out["requires_approval"] and rnd["waiting_on"]["kind"] == "human_approval"
    t = rnd["targeted"]
    assert t["offered"] == 8 and 0 < t["added"] < 8 and t["targeted_share"] <= 0.4  # capped on the train split
    eps = client.portal.call(list_episodes, ctx(), pid)
    rl_eps = [e for e in eps if e.provenance.get("sandbox_id") == sb.id]
    assert len(rl_eps) == 8 and all(e.provenance["planner"] == f"rl_sandbox:{sb.id}" for e in rl_eps)
    assert {e.id for e in rl_eps} == set(rnd["demo_episode_ids"])
    versions = client.portal.call(dataset_service.list_versions, ctx(), pid)
    new_ds = versions[-1]
    assert new_ds.id == rnd["dataset_version_id"] and new_ds.parent_id == base_ds.id == rnd["parent_dataset_version_id"]
    in_ds = {ln.episode_id for ln in new_ds.links}
    assert {ln.episode_id for ln in base_ds.links} <= in_ds and len(in_ds & {e.id for e in rl_eps}) == t["added"]
    lineage = json.loads((ctx().local(new_ds.lerobot_root) / "meta" / "cortex_lineage.json").read_text())
    from_rl = [s for s in lineage["source_episodes"] if s["rl_sandbox_id"] == sb.id]
    assert len(from_rl) == t["added"] and all(s["planner"] == f"rl_sandbox:{sb.id}" for s in from_rl)
    tj = client.portal.call(ctx().repo.get, "training_jobs", rnd["training_job_id"], TrainingJob)
    assert tj.status == "proposed" and tj.dataset_version_id == new_ds.id  # agents can't approve training

    # waiting on the human: calling again changes nothing and launches nothing
    again = tool(client, "close_rl_round", project_id=pid, sandbox_id=sb.id)["round"]
    assert again["status"] == "retrain_proposed" and not launched

    # 3) a human approved it and the checkpoint was registered: paired re-evaluation of every condition
    client.portal.call(ctx().repo.save, "checkpoints", Checkpoint(
        id="ck_new", project_id=pid, training_job_id=tj.id, policy_type="act", step=20_000, path="ck_new"))
    out = tool(client, "close_rl_round", project_id=pid, sandbox_id=sb.id)
    rnd = out["round"]
    assert [r.policy.checkpoint_id for r in launched] == ["ck_old", "ck_new"]
    assert launched[0].base_seed == launched[1].base_seed == rnd["heldout_base_seed"] >= 900_000  # matched seeds
    assert {s.name for s in launched[0].branches} == {"baseline", "shifted_object"}  # all conditions
    assert launched[0].episodes_per_branch == rnd["episodes_per_condition"] == 10

    # 4) predicted vs realised logged; regressions flagged; the transfer curve refitted (still the prior: 1 round)
    assert rnd["status"] == "realised", rnd
    assert rnd["realised_uplift"] == pytest.approx(0.6) and rnd["predicted_policy_uplift"] == pytest.approx(0.12)
    assert rnd["predicted_vs_realised"]["realised_uplift"]["kind"] == "measured"
    assert rnd["predicted_vs_realised"]["predicted_policy_uplift"]["kind"] == "estimate"
    assert rnd["regressions"] == ["baseline"] and rnd["p_value"] < 0.05 and not rnd["needs_human_demos"]
    assert rnd["transfer"]["source"] == "prior" and rnd["transfer"]["rounds_used"] == 1
    assert rnd["checkpoint_before"] == "ck_old" and rnd["checkpoint_after"] == "ck_new"
    rounds = client.get(f"/api/projects/{pid}/rl/rounds").json()
    assert rounds["rounds"][0]["status"] == "realised" and rounds["transfer"]["source"] == "prior"

    # 5) after three realised rounds the estimates use the learned transfer curve
    from cortex.services.rl_service import RLRound

    for g, u in ((0.4, 0.1), (0.3, 0.15)):
        client.portal.call(ctx().repo.save, "rl_rounds", RLRound(
            project_id=pid, sandbox_id="rl_old", weakness_key="dim", condition="center/dim", skill_gain=g,
            realised_uplift=u, status="realised"))
    listed = tool(client, "get_rl_rounds", project_id=pid)
    assert listed["transfer"]["source"] == "learned" and listed["transfer"]["rounds_used"] == 3


def test_close_round_refuses_unfinished_sandboxes_and_respects_the_cap(client):
    pid, sb, _ = seed_project(client, n_base=2, n_demos=4)
    # 2 base episodes: the cap leaves room for at most one demo
    out = tool(client, "close_rl_round", project_id=pid, sandbox_id=sb.id)["round"]
    assert out["targeted"]["added"] <= 1 and out["targeted"]["targeted_share"] <= 0.4
    other = propose(pid, [sb.weakness])[0]
    other.status = "running"
    client.portal.call(ctx().repo.save, "rl_sandboxes", other)
    r = tool(client, "close_rl_round", 400, project_id=pid, sandbox_id=other.id)
    assert "a round starts once the sandbox has finished" in r["detail"]
