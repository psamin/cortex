import pytest

from cortex.models import (
    BranchResult,
    BranchSpec,
    Condition,
    EpisodeResult,
    EvaluationRun,
    FailureCluster,
    PolicyRef,
    SimulationBranch,
)
from cortex.rl import NoPolicyEvaluation, estimate, find_weaknesses, propose
from cortex.services.simulation_runner import wilson


def branch(ev, name, succ, fails, cond=Condition(), offset=(0.0, 0.0)):
    eps = [EpisodeResult(seed=i, success=True, reward=1.0, steps=200) for i in range(succ)]
    eps += [EpisodeResult(seed=100 + i, success=False, reward=0.2, steps=300, failure_mode=m)
            for i, m in enumerate(fails)]
    n = len(eps)
    res = BranchResult(n=n, successes=succ, success_rate=succ / n, ci95=wilson(succ, n), mean_reward=0.5, episodes=eps)
    return SimulationBranch(evaluation_id=ev.id, spec=BranchSpec(name=name, condition=cond, cube_offset=offset),
                            seeds=[e.seed for e in eps], result=res)


def evaluation(kind="checkpoint", ck="ck_1", branches=()):
    run = EvaluationRun(project_id="prj_x", policy=PolicyRef(kind=kind, checkpoint_id=ck if kind == "checkpoint" else None))
    return {"run": run, "branches": [b(run) for b in branches]}


def test_weaknesses_rank_by_gap_and_confidence_and_name_failure_modes():
    ev = evaluation(branches=[
        lambda r: branch(r, "baseline", 9, ["grasp_failed"]),
        lambda r: branch(r, "shifted_object", 2, ["grasp_failed"] * 6 + ["dropped_before_target"] * 2,
                         offset=(0.04, 0.06)),
        lambda r: branch(r, "changed_lighting", 6, ["released_outside_target"] * 4,
                         cond=Condition(lighting="dim")),
    ])
    ws = find_weaknesses([ev], [], target=0.8)
    assert [w.key for w in ws] == ["shifted_object", "changed_lighting"]  # baseline (0.9) passes the target
    top = ws[0]
    assert top.dominant_failure == "grasp_failed" and top.failure_modes == {"grasp_failed": 6,
                                                                           "dropped_before_target": 2}
    assert top.gap == pytest.approx(0.6) and top.ci95[0] < top.success_rate < top.ci95[1]


def test_only_learned_checkpoints_count():
    expert = evaluation(kind="scripted_expert", branches=[lambda r: branch(r, "baseline", 1, ["grasp_failed"] * 9)])
    with pytest.raises(NoPolicyEvaluation):
        find_weaknesses([expert], [])


def test_vlm_hypotheses_attach_but_stay_labelled():
    ev = evaluation(branches=[lambda r: branch(r, "changed_lighting", 3, ["grasp_failed"] * 7,
                                                cond=Condition(lighting="dim"))])
    cl = FailureCluster(project_id="prj_x", analysis_id="an_1", category="vlm:occlusion", source="vlm",
                        episode_ids=[], conditions=["center/dim"], evidence=["gripper shadow hides the cube"],
                        uncertainty=0.5)
    w = find_weaknesses([ev], [cl])[0]
    assert w.vlm_hypotheses == ["gripper shadow hides the cube"]


def test_proposals_focus_on_the_failure_mode_and_estimates_are_labelled():
    ev = evaluation(branches=[
        lambda r: branch(r, "shifted_object", 2, ["grasp_failed"] * 8, offset=(0.04, 0.06)),
        lambda r: branch(r, "changed_lighting", 5, ["released_outside_target"] * 5, cond=Condition(lighting="dim")),
    ])
    sbs = propose("prj_x", find_weaknesses([ev], []))
    grasp, place = sbs
    assert {p.name for p in grasp.params} >= {"grasp_dx", "grasp_dy", "yaw_offset"}
    assert {p.name for p in place.params} >= {"release_dx", "release_dy"}
    assert grasp.reward_weights["lifted"] > place.reward_weights["lifted"]
    assert grasp.env.cube_offset == (0.04, 0.06) and grasp.status == "proposed"

    est = estimate(grasp, pilot_successes=6, pilot_n=10, sim_steps_per_s=600)
    assert est.kind == "estimate" and est.assumptions
    lo, hi = est.policy_uplift
    assert 0 < lo < hi <= 1 - grasp.weakness.success_rate
    assert est.policy_success_after[0] == pytest.approx(0.2 + lo, abs=1e-3)
    assert est.cpu_minutes > 0 and est.usd == 0.0 and est.wall_minutes == pytest.approx(est.cpu_minutes / 2, abs=0.1)


def test_no_uplift_claimed_when_the_sandbox_skill_is_no_better_than_the_policy():
    ev = evaluation(branches=[lambda r: branch(r, "baseline", 7, ["grasp_failed"] * 3)])
    sb = propose("prj_x", find_weaknesses([ev], []))[0]
    est = estimate(sb, pilot_successes=0, pilot_n=10, sim_steps_per_s=600)
    assert est.policy_uplift[0] == 0.0  # a skill at 0-21% after RL cannot lift a 70% policy
