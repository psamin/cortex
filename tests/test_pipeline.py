import math

from dreamcache.agents.agent import Agent, AgentConfig
from dreamcache.detection.difficulty import DetectorConfig, difficulty
from dreamcache.detection.disagreement import disagreement
from dreamcache.detection.uncertainty import normalized_entropy
from dreamcache.environments.cloning import state_fingerprint
from dreamcache.environments.minigrid_adapter import make_env
from dreamcache.experiments.runner import run_episode
from dreamcache.policies.heuristic import SimBFSPolicy
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
