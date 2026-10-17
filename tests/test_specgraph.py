"""Prefetch validation/invalidation/fallback tests and predictor sanity (CPU, no LLM)."""
from dreamcache.environments.minigrid_adapter import make_env
from dreamcache.environments.observations import features
from dreamcache.specgraph.predictors import (ContextPredictor, FrequencyPredictor, PersonalizedPredictor,
                                             evaluate)
from dreamcache.specgraph.prefetch import PrefetchingPolicy
from dreamcache.specgraph.scheduler import SchedConfig, Scheduler
from dreamcache.types import Cost, Decision

DYN = "MiniGrid-Dynamic-Obstacles-8x8-v0"


class Actor:
    def __init__(self):
        self.calls = 0

    def decide(self, obs, history):
        self.calls += 1
        a = 1 if features(obs).front_blocked or len(history) % 4 == 3 else 2
        return Decision(a, {0: .1, 1: .3, 2: .6}, Cost(llm_calls=1, llm_seconds=0.01, wall_seconds=0.01))


class Oracle:  # perfect predictor for the stand-in actor
    def __init__(self, a):
        self.a = a

    def predict(self, row):
        return {0: 0.0, 1: 1.0, 2: 0.0} if self.a == 1 else {0: 0.0, 1: 0.0, 2: 1.0}


def run(policy, predictor_fn, steps=10, fp="fp"):
    env, obs = make_env(DYN, 5)
    actor = Actor()
    ref = Actor()
    pf = PrefetchingPolicy(actor, [0, 1, 2], None, Scheduler(SchedConfig(policy=policy, max_inflight=2), [0, 1, 2]),
                           DYN, fp)
    hist = []
    for t in range(steps):
        pf.predictor = predictor_fn(env, obs, hist, ref)
        pf.speculate(env, obs, hist, t)
        d = pf.decide(obs, hist)
        assert d.action == ref.decide(obs, hist).action  # lossless
        obs, r, term, trunc, _ = env.step(d.action)
        hist.append(d.action)
        if term or trunc:
            break
    return pf.finalize(), hist


def _perfect(env, obs, hist, ref):
    # predicts the action the actor WILL take at the current step (prefetch target = next state)
    return Oracle(ref.decide(obs, hist).action)


def test_perfect_predictor_reuses_and_is_lossless():
    st, hist = run("top1", _perfect)
    assert st["reused"] + st["reused_late"] >= len(hist) - 2


def test_wrong_predictor_invalidates_and_falls_back():
    def wrong(env, obs, hist, ref):
        return Oracle(1 if ref.decide(obs, hist).action == 2 else 2)
    st, hist = run("top1", wrong)
    assert st["reused"] + st["reused_late"] == 0 and st["unused"] >= 1


def test_no_speculation_launches_nothing():
    st, _ = run("none", _perfect)
    assert st["launched"] == 0


def test_stale_fingerprint_is_not_reused():
    env, obs = make_env(DYN, 5)
    actor = Actor()
    pf = PrefetchingPolicy(actor, [0, 1, 2], Oracle(2), Scheduler(SchedConfig(policy="top1"), [0, 1, 2]), DYN, "v1")
    pf.speculate(env, obs, [], 0)
    for r in pf.spec.values():
        r["fingerprint"] = "OLD-MODEL"
    obs2, *_ = env.step(2)
    n0 = actor.calls
    pf.cur_t = 1
    pf.decide(obs2, [2])
    assert actor.calls >= n0 + 1  # recomputed, not reused
    pf.finalize()


def test_predictors_fit_and_score():
    rows = [{"env": "e", "context": f"c{i % 3}", "prev_op": i % 2, "op": (i % 3 == 0) + 1} for i in range(90)]
    for P in (FrequencyPredictor, ContextPredictor, PersonalizedPredictor):
        m = evaluate(P([0, 1, 2]).fit(rows), rows)
        assert 0 <= m["top1"] <= 1 and m["nll"] > 0
    assert evaluate(ContextPredictor([0, 1, 2]).fit(rows), rows)["top1"] > \
        evaluate(FrequencyPredictor([0, 1, 2]).fit(rows), rows)["top1"]


def test_runner_prefetch_with_surrogate(tmp_path):
    from dreamcache.experiments.runner import run_episode
    rows = tmp_path / "rows.jsonl"
    rows.write_text("")
    m = {"name": "B5", "agent": {"trigger": "never"},
         "prefetch": {"predictor": "personalized", "rows": str(rows), "workflow": "w", "sched": {"policy": "top1"}}}
    r = run_episode(DYN, 1, m, {"kind": "surrogate"}, max_steps=8)
    assert r["extra"]["prefetch"]["launched"] >= 1


def test_b7_graph_ranked_candidates(tmp_path):
    import json as _json
    from dreamcache.experiments.runner import run_episode
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(_json.dumps({"workflow": "w", "env": DYN, "context": "x", "prev_op": -1, "op": 2})
                              for _ in range(5)))
    m = {"name": "B7", "agent": {"trigger": "always", "spec": {"k": 2, "candidates": "graph"}},
         "graph_rank": {"predictor": "personalized", "rows": str(rows), "workflow": "w"}}
    r = run_episode(DYN, 1, m, {"kind": "surrogate"}, max_steps=5)
    assert any("graph_probs" in rec for rec in r["records"])
