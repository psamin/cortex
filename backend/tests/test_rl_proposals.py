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
from cortex.rl import NoPolicyEvaluation, Pilot, estimate, find_weaknesses, propose
from cortex.rl.learner import cem
from cortex.rl.stats import mcnemar_exact, start_xy
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

    pilot = Pilot(n=10, successes=3, wilson_95=wilson(3, 10), cem_iterations=2, cem_best_successes=9,
                  cem_best_n=10)
    est = estimate(grasp, pilot, sim_steps_per_s=600)
    assert est.kind == "estimate" and est.assumptions
    assert est.predicted_sandbox_success.source == "measured" and est.predicted_policy_uplift.source == "prior"
    lo, hi = est.predicted_policy_uplift.ci90
    assert 0 <= lo < est.predicted_policy_uplift.mean < hi <= 1
    assert est.predicted_policy_success.mean > grasp.weakness.success_rate
    assert 0 < est.p_improve_ge_delta <= 1 and est.predicted_regression_elsewhere.mean <= 0
    assert est.cost.cpu_hours > 0 and est.cost.usd == 0.0 and est.uplift_per_hour > 0


def test_transfer_is_learned_after_three_rounds():
    ev = evaluation(branches=[lambda r: branch(r, "shifted_object", 2, ["grasp_failed"] * 8, offset=(0.04, 0.06))])
    sb = propose("prj_x", find_weaknesses([ev], []))[0]
    pilot = Pilot(n=10, successes=3, wilson_95=wilson(3, 10), cem_iterations=2, cem_best_successes=9, cem_best_n=10)
    est = estimate(sb, pilot, 600, history=[(0.4, 0.1), (0.3, 0.06), (0.5, 0.15)])
    assert est.predicted_policy_uplift.source == "learned"
    assert any("learned" in a for a in est.assumptions)


def test_no_uplift_claimed_when_the_sandbox_skill_is_no_better_than_the_policy():
    ev = evaluation(branches=[lambda r: branch(r, "baseline", 7, ["grasp_failed"] * 3)])
    sb = propose("prj_x", find_weaknesses([ev], []))[0]
    est = estimate(sb, Pilot(n=10, successes=0, wilson_95=wilson(0, 10)), sim_steps_per_s=600)
    assert est.predicted_policy_uplift.ci90[0] == 0.0  # a skill near 0% cannot reliably lift a 70% policy
    assert est.p_improve_ge_delta < 0.5


def test_ranking_uses_the_confident_gap_and_clusters_starts():
    ev = evaluation(branches=[
        lambda r: branch(r, "few_trials", 0, ["grasp_failed"] * 2),  # 0/2: big gap, huge CI
        lambda r: branch(r, "many_trials", 10, ["grasp_failed"] * 20),  # 10/30: confident gap
    ])
    ws = find_weaknesses([ev], [])
    assert ws[0].key == "many_trials" and ws[0].gap_at_upper > 0
    cl = ws[0].start_clusters
    assert cl and cl[0].count == max(c.count for c in cl) and sum(c.count for c in cl) <= 20
    x, y = start_xy("center", (0.0, 0.0), 7)
    assert abs(x - 0.22) <= 0.015 and abs(y + 0.04) <= 0.015


def test_cem_climbs_a_toy_objective_and_stops_on_target():
    from cortex.rl.proposals import PARAMS

    params = [PARAMS["grasp_dx"], PARAMS["grasp_dy"]]

    def evaluate(p, seeds):  # success iff close to (0.006, -0.004)
        ok = abs(p["grasp_dx"] - 0.006) < 0.003 and abs(p["grasp_dy"] + 0.004) < 0.003
        d = abs(p["grasp_dx"] - 0.006) + abs(p["grasp_dy"] + 0.004)  # metres
        return [(1.0 if ok else max(0.0, 1 - d / 0.024), ok) for _ in seeds]  # normalized like real returns

    best, hist, reason = cem(params, evaluate, iterations=15, population=32, episodes=2, seed_base=0, target=0.8)
    assert reason in ("target", "converged") and len(hist) < 15 and hist[-1].success_rate > 0.5
    assert abs(best["grasp_dx"] - 0.006) < 0.003 and abs(best["grasp_dy"] + 0.004) < 0.003


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 9) < 0.05 < mcnemar_exact(3, 6)
