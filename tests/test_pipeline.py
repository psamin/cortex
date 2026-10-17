import math

from dreamcache.agents.agent import Agent, AgentConfig
from dreamcache.detection.difficulty import DetectorConfig, difficulty
from dreamcache.detection.disagreement import disagreement
from dreamcache.detection.uncertainty import normalized_entropy
from dreamcache.environments.cloning import state_fingerprint
from dreamcache.environments.minigrid_adapter import make_env
from dreamcache.experiments.runner import run_episode
from dreamcache.policies.ollama_policy import parse_reply
from dreamcache.policies.surrogate import SurrogatePolicy
from dreamcache.speculation.executor import SpecConfig

DYN, DK = "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-DoorKey-8x8-v0"


def test_entropy_and_disagreement_bounds():
    assert normalized_entropy({0: 1.0, 1: 0.0, 2: 0.0}) == 0
    assert math.isclose(normalized_entropy({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}), 1.0)
    assert disagreement([2, 2, 2, 2], 3) == 0
    assert math.isclose(disagreement([0, 1, 2], 3), 1.0)
    assert difficulty({"entropy": 1, "disagreement": 1, "risk": 1}, DetectorConfig()) == 1


def test_parse_reply_recovers_and_flags():
    s, a, bad = parse_reply('junk {"scores": {"left": 1, "right": 2, "forward": 7}, "action": "forward"}', [0, 1, 2])
    assert a == 2 and not bad and s[2] == 7
    s, a, bad = parse_reply("not json", [0, 1, 2])
    assert bad


def test_sim_bfs_solves_doorkey_and_dynamic():
    for env_id, seed in [(DK, 0), (DK, 1), (DYN, 0)]:
        r = run_episode(env_id, seed, {"name": "p", "kind": "planner_sim"}, {})
        assert r["steps"] > 0
    assert run_episode(DK, 0, {"name": "p", "kind": "planner_sim"}, {})["success"]


def test_speculation_leaves_real_env_untouched():
    env, obs = make_env(DYN, 1)
    before = state_fingerprint(env)
    cfg = AgentConfig(trigger="always", spec=SpecConfig(k=3, horizon=5, samples=2))
    agent = Agent(SurrogatePolicy([0, 1, 2], seed=0), [0, 1, 2], "dynamic", cfg, seed=0)
    a, rec = agent.act(env, obs, [], 0)
    assert state_fingerprint(env) == before
    assert rec["triggered"] and len(rec["candidate_results"]) == 3
    assert rec["cost"]["sim_steps"] > 0


def test_episode_methods_run_with_surrogate():
    fs = {"kind": "surrogate"}
    for m in [{"name": "greedy", "agent": {"trigger": "never"}},
              {"name": "sel", "agent": {"trigger": "threshold", "threshold": 0.4}},
              {"name": "rand", "agent": {"trigger": "random", "rate": 0.3}},
              {"name": "oracle", "agent": {"trigger": "oracle", "oracle_samples": 2}},
              {"name": "bok", "agent": {"trigger": "always", "resolve": "norollout"}}]:
        r = run_episode(DYN, 2, m, fs, max_steps=15)
        assert r["steps"] >= 1


def test_new_methods_run_with_surrogate():
    fs = {"kind": "surrogate", "big_model": "surrogate"}
    for m in [{"name": "verify", "agent": {"trigger": "always", "resolve": "verify"}},
              {"name": "cascade", "agent": {"trigger": "threshold", "threshold": 0.3, "resolve": "cascade"}},
              {"name": "vote", "agent": {"trigger": "always", "resolve": "vote", "vote_n": 3}},
              {"name": "mcts", "agent": {"trigger": "always", "resolve": "mcts", "mcts_iterations": 12}},
              {"name": "draft", "agent": {"mode": "spec_draft", "plan_len": 3}}]:
        r = run_episode(DYN, 3, m, fs, max_steps=10)
        assert r["steps"] >= 1, m


def test_mcts_does_not_mutate_env():
    from dreamcache.speculation.evaluator import EvalConfig
    from dreamcache.speculation.mcts_search import mcts_decide
    env, _ = make_env(DYN, 4)
    before = state_fingerprint(env)
    a, stats, cost = mcts_decide(env, [0, 1, 2], 20, 6, EvalConfig(), seed=0)
    assert state_fingerprint(env) == before and a in (0, 1, 2) and cost.sim_steps > 0


def test_lossless_spec_actions_match_actor():
    """Executed actions must be the actor's (big model's) own decisions."""
    from dreamcache.agents.lossless import LosslessSpecAgent
    from dreamcache.environments.observations import features
    from dreamcache.types import Cost, Decision

    class Actor:  # deterministic stand-in actor (a pure function of the observation)
        def decide(self, obs, history):
            a = 1 if features(obs).front_blocked or len(history) % 4 == 3 else 2
            return Decision(a, {0: 0.2, 1: 0.3, 2: 0.5}, Cost(llm_calls=1))

    env, obs = make_env(DYN, 5)
    ag = LosslessSpecAgent(Actor(), None, [0, 1, 2], guessers=["obs_heuristic"])
    ref = Actor()
    hist = []
    for t in range(12):
        a, rec = ag.act(env, obs, hist, t)
        assert a == ref.decide(obs, hist).action
        obs, r, term, trunc, _ = env.step(a)
        hist.append(a)
        if term or trunc:
            break
    st = ag.finalize()
    assert st["hits"] + st["misses"] == len(hist)


def test_lossless_draft_equals_big_greedy():
    """With the surrogate big model being deterministic per observation, executed actions
    must equal what the big model alone would choose on the same states."""
    from dreamcache.environments.observations import features
    from dreamcache.types import Cost, Decision

    class Big:
        def decide(self, obs, history):
            return Decision(1 if features(obs).front_blocked else 2, {0: .1, 1: .3, 2: .6}, Cost(llm_calls=1))

    class Small(Big):
        def draft_plan(self, obs, history, length):
            return [2, 2, 0, 2][:length], Cost(llm_calls=1)

    env, obs = make_env(DYN, 6)
    ag = Agent(Small(), [0, 1, 2], "dynamic", AgentConfig(mode="spec_draft_lossless", plan_len=4), seed=0, big=Big())
    hist = []
    for t in range(15):
        a, rec = ag.act(env, obs, hist, t)
        assert a == Big().decide(obs, hist).action
        obs, r, term, trunc, _ = env.step(a)
        hist.append(a)
        if term or trunc:
            break
