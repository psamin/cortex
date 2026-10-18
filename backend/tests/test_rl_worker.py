"""RL sandbox worker (#80): ADR widening, DART noise, and a tiny real-model run (pilot + run, ~2-3 minutes)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cortex.models import Condition
from cortex.rl import SandboxBudget, SandboxEnv, Weakness, propose
from cortex.rl.sandbox_env import DART_LEGS, DART_PHASES, dart_perturb, widen
from cortex.sim.pick_place_skill import Target

BACKEND = Path(__file__).resolve().parents[1]


def test_adr_widens_only_above_threshold_and_caps():
    env = SandboxEnv(condition=Condition(), offset_jitter_m=0.02, yaw_jitter_rad=0.4, light_scales=(0.8, 1.2))
    assert not widen(env, 0.8) and env.widened == 0  # must *exceed* widen_at
    assert widen(env, 0.9)
    assert env.offset_jitter_m == pytest.approx(0.03) and env.yaw_jitter_rad == pytest.approx(0.6)
    assert env.light_scales == pytest.approx((0.7, 1.3)) and env.widened == 1
    while widen(env, 1.0):
        pass
    assert env.offset_jitter_m == pytest.approx(env.max_offset_jitter_m) and env.widened == 3
    assert not widen(env, 1.0)  # at the cap: nothing left to widen


class FakeTuned:  # rl.skill.TunedSkill's shape: a fixed list of (q, hold) waypoints
    def __init__(self):
        self.waypoints = [(np.r_[np.full(5, float(i)), 40.0], 4) for i in range(8)]


def test_dart_perturbs_free_space_waypoints_only():
    sk = FakeTuned()
    before = [q.copy() for q, _ in sk.waypoints]
    dart_perturb(sk, np.random.default_rng(0), sigma_deg=3.0)
    for i, (q, hold) in enumerate(sk.waypoints):
        assert q[5] == before[i][5] and hold == 4  # gripper and timing untouched
        assert (not np.allclose(q, before[i])) == (i in DART_LEGS)


def test_dart_perturbs_pick_place_skill_targets_by_phase():
    class Gen:
        moves = iter([Target(np.zeros(5), 40.0, phase=p) for p in ("hover", "approach", "close", "carry", "lower")])

    sk = Gen()
    dart_perturb(sk, np.random.default_rng(1))
    out = list(sk.moves)
    assert [bool(np.any(t.q != 0)) for t in out] == [t.phase in DART_PHASES for t in out]


def tiny_sandbox():
    w = Weakness(key="shifted_object", branch_id="br_1", evaluation_id="ev_1", checkpoint_id="ck_1",
                 condition=Condition(), cube_offset=(0.02, 0.0), n=10, successes=3, success_rate=0.3,
                 ci95=(0.11, 0.6), target=0.8, gap=0.5, dominant_failure="grasp_failed", score=0.3)
    b = SandboxBudget(iterations=2, population=4, episodes_per_candidate=1, pilot_episodes=4, pilot_cem_iterations=1,
                      heldout_episodes=5, demos=1, dart_noise_frac=0.0)
    return propose("prj_x", [w], b)[0]


def run_worker(tmp: Path, cfg: dict) -> list[dict]:
    p = tmp / f"{cfg['mode']}.json"
    p.write_text(json.dumps(cfg))
    out = subprocess.run([sys.executable, "-m", "cortex.workers.rl_worker", str(p)], cwd=BACKEND,
                         capture_output=True, text=True, timeout=900)
    events = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    assert out.returncode == 0, out.stderr[-2000:]
    return events


def test_tiny_real_sandbox_pilot_and_run(tmp_path):
    sb = tiny_sandbox()
    base = {"sandbox": sb.model_dump(mode="json"), "seed_base": 1, "max_steps": 300}
    ev = run_worker(tmp_path, {**base, "mode": "pilot", "out_dir": str(tmp_path / "pilot")})
    pilot = next(e for e in ev if e["type"] == "rl_pilot")
    assert pilot["pilot"]["n"] == 4 and pilot["pilot"]["cem_iterations"] == 1 and pilot["pilot"]["cem_best_n"] == 4
    assert pilot["sim_steps_per_s"] > pilot["record_steps_per_s"] > 0 and pilot["steps_per_episode"] > 0
    assert ev[-1] == {"type": "result", "ok": True}

    ev = run_worker(tmp_path, {**base, "mode": "run", "out_dir": str(tmp_path / "run")})
    types = [e["type"] for e in ev]
    phases = [e["phase"] for e in ev if e["type"] == "rl_phase"]
    assert phases == ["train", "heldout", "demos"]
    assert types.count("rl_iteration") == 2 and types[-2:] == ["rl_result", "result"]
    assert types.count("rl_episode") >= 2 * 4 + 2 * 5 + 1  # train + matched held-out + at least one demo attempt
    stats = next(e for e in ev if e["type"] == "sim_stats")
    assert len(stats["joints"]) == 6 and len(stats["tcp"]) == 3 and "return_so_far" in stats
    it = next(e for e in ev if e["type"] == "rl_iteration")
    assert {"elite_success_rate", "best_params", "ci95", "env", "widened"} <= set(it)

    res = ev[-2]["result"]
    assert res["kind"] == "measured" and len(res["heldout_seeds"]) == 5
    assert 0 <= res["p_value"] <= 1 and res["stop_reason"] in {"budget", "target", "plateau", "converged"}
    held = [e for e in ev if e["type"] == "rl_episode" and e["phase"] == "heldout"]
    assert [e["seed"] for e in held[:5]] == [e["seed"] for e in held[5:]] == res["heldout_seeds"]  # matched seeds
    assert (tmp_path / "run" / "live.jpg").exists()
    if res["demo_episode_ids"]:
        d = tmp_path / "run" / "demos" / res["demo_episode_ids"][0]
        npz = np.load(d / "episode.npz")
        assert {"front", "wrist", "state", "action", "reward", "timestamp", "action_low", "action_high"} <= set(npz)
        assert npz["action"].shape[1] == 6 and len(npz["front"]) == len(npz["action"])
        meta = json.loads((d / "result.json").read_text())
        assert meta["success"] and meta["provenance"]["planner"] == f"rl_sandbox:{sb.id}"
