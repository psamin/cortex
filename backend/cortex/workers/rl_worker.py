"""RL sandbox worker process (#80): pilot, CEM training with ADR, held-out evaluation, demo generation.

    python -m cortex.workers.rl_worker <config.json>

cfg: {mode: "pilot" | "run", sandbox: RLSandbox json, out_dir, live_dir?, scene?, seed_base?, max_steps?,
      stop_file?, live?}
Emits JSON lines on stdout, like the other workers:
  rl_phase      {phase}                                   pilot | train | heldout | demos
  rl_episode    {phase, seed, success, return, failure_mode, steps}
  rl_iteration  IterationStats (CEM, after ADR)
  sim_stats     {phase, skill_phase, step, joints, tcp, return_so_far} every STATS_EVERY steps
  rl_pilot      {pilot: Pilot, sim_steps_per_s}           (mode "pilot")
  rl_result     {result: SandboxResult, sim_steps_per_s}  (mode "run")
  result        {ok}
The latest frame goes to live_dir/live.jpg (throttled). Demos are saved in sim_worker's record format
(episode.npz, front.mp4, thumb.jpg, result.json) under out_dir/demos/<n>/ with provenance rl_sandbox:<id>.
Touching ``stop_file`` stops the sandbox at the next check; it still evaluates what it has.
Everything the skill does reads privileged sim state, so its demos are labelled oracle demos.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..errors import worker_error
from ..rl.learner import cem
from ..rl.sandbox_env import Physics, act_labelled, dart_perturb, draw, episode_return, widen
from ..rl.stats import mcnemar_exact
from ..rl.types import IterationStats, Pilot, RLSandbox, SandboxResult
from ..services.simulation_runner import wilson
from ..sim.so101 import FPS, PickPlaceEnv, configure_gl
from ..sim import env_factory
from ..sim.viewer import CAM_EVERY
from .sim_worker import Preview, emit, failure_mode, rest_height, write_hd, write_mp4

STATS_EVERY = 15  # control steps between sim_stats events
DEFAULT_MAX_STEPS = 300  # the cube task's episode length
LIVE_EVERY_S = 0.5  # live.jpg throttle
SETTLE_STEPS = 10
UNSTABLE_Z, UNSTABLE_DIST = 0.5, 1.0  # metres: the object left the workspace, i.e. the physics blew up
PILOT_SEEDS, TRAIN_SEEDS, HELDOUT_SEEDS, DEMO_SEEDS = 10_000, 100_000, 500_000, 700_000


def physics_step(env: PickPlaceEnv, action: np.ndarray, render: bool):
    """``env.step`` without the observation-camera renders (~85% of a step's cost) unless pixels are recorded.
    Uses ``PickPlaceEnv.step(render=False)`` from #99 when present; until then, the same render-free step here."""
    if render:
        return env.step(action)
    if "render" in inspect.signature(env.step).parameters:
        return env.step(action, render=False)
    w = env.world  # TODO(#80): drop this branch once #99 is on main
    w.data.ctrl[w.actuators] = w.to_ctrl(np.asarray(action, dtype=np.float32))
    mujoco.mj_step(w.model, w.data, nstep=w.substeps)
    env.env._steps += 1
    env.steps += 1
    env.max_cube_z = max(env.max_cube_z, float(env.cube_pos()[2]))
    success = env.in_box()
    return None, env.reward(), success, env.steps >= env.episode_steps, {"is_success": success}


def make_env(scene: dict[str, Any] | None, max_steps: int) -> PickPlaceEnv:
    return env_factory.make_env(scene, max_steps)


def step_limit(scene: dict[str, Any] | None, cfg: dict[str, Any]) -> int:
    """Control steps per rollout: the cube task's 300, or the intake scene's longer limit (side grasps need
    more than 500 steps; cutting them at 300 scored base and tuned skills ~0, #152)."""
    return env_factory.max_steps(scene, int(cfg.get("max_steps", DEFAULT_MAX_STEPS)))


def make_policy(scene: dict[str, Any] | None, params: dict[str, float]):
    if not scene:
        from ..rl.skill import TunedPolicy

        return TunedPolicy(params)
    from ..sim.pick_place_skill import PickPlacePolicy

    obj = scene.get("object", {})
    return PickPlacePolicy(params, height=obj.get("height"), grasp_width=obj.get("grasp_width"))


class Runner:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.sb = RLSandbox.model_validate(cfg["sandbox"])
        self.scene = cfg.get("scene")
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.live_dir = Path(cfg.get("live_dir") or self.out)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.seed_base = int(cfg.get("seed_base", 0))
        self.max_steps = step_limit(self.scene, cfg)
        self.stop_file = Path(cfg["stop_file"]) if cfg.get("stop_file") else None
        self.env = make_env(self.scene, self.max_steps)
        self.env.reset(self.seed_base, draw(self.sb.env, self.seed_base).var)
        self.physics = Physics(self.env)
        self.preview = (Preview(self.env, self.live_dir, live=True, policy_cams=True)
                        if cfg.get("live", True) else None)
        self.last_live = 0.0
        self.phase = "pilot"
        self.sim_steps, self.sim_s, self.episodes = 0, 0.0, 0

    # ------------------------------------------------------------------ one episode
    def stopped(self) -> bool:
        return bool(self.stop_file and self.stop_file.exists())

    def rollout(self, params: dict[str, float], seed: int, record: bool = False, dart: bool = False) -> dict:
        d = draw(self.sb.env, seed)
        env = self.env
        env.reset(seed, d.var)
        self.physics.apply(d)
        obs = env._obs()
        policy = make_policy(self.scene, params)
        policy.reset(env, obs, seed)
        if dart:
            dart_perturb(policy.ex, np.random.default_rng(seed + 1))
        buf: dict[str, Any] = {"front": [], "wrist": [], "state": [], "action": [], "reward": [], "hd": {}}
        video: list[np.ndarray] = []
        success, settle, t0 = False, 0, time.time()
        for step in range(self.max_steps):
            # DART: the perturbed command runs; the label is the clean skill's command (#152)
            action, label = act_labelled(policy.ex) if dart else (policy.act(obs),) * 2
            if record:
                buf["front"].append(obs["pixels"]["front"])
                buf["wrist"].append(obs["pixels"]["wrist"])
                buf["state"].append(obs["agent_pos"])
                buf["action"].append(label)
            nxt, reward, success, _, _ = physics_step(env, action, render=record)
            obs = nxt if nxt is not None else obs
            if record:
                buf["reward"].append(reward)
                if self.preview:
                    video.append(self.preview.r.render(env.world.data))
                    if step % CAM_EVERY == 0:
                        self.preview.grab_cams(buf["hd"])
            if step % STATS_EVERY == 0:
                self._stats(step, policy, success)
            if policy.ex.done:
                settle += 1
                if settle >= SETTLE_STEPS:
                    break
        steps = step + 1
        self.sim_steps += steps
        self.episodes += 1
        self.sim_s += time.time() - t0
        dist = float(np.linalg.norm(env.cube_pos()[:2] - env.box_pos()[:2]))
        rest = rest_height(env)
        ep = {"seed": seed, "success": bool(success), "steps": steps, "max_cube_z": env.max_cube_z,
              "rest_z": rest, "final_cube_box_dist": dist,
              "failure_mode": failure_mode(success, env.max_cube_z, steps, self.max_steps, dist, rest)}
        if env.max_cube_z > UNSTABLE_Z or dist > UNSTABLE_DIST:  # never reward a physics blow-up as a "lift"
            ep.update({"success": False, "failure_mode": "sim_unstable", "return": 0.0})
        else:
            ep["return"] = round(episode_return(ep, self.sb.reward_weights)[0], 4)
        emit(type="rl_episode", sandbox_id=self.sb.id, phase=self.phase, **ep)
        if record:
            ep["buf"], ep["video"] = buf, video
        return ep

    def _stats(self, step: int, policy, success: bool) -> None:
        env = self.env
        so_far = {"success": success, "max_cube_z": env.max_cube_z, "rest_z": rest_height(env),
                  "final_cube_box_dist": float(np.linalg.norm(env.cube_pos()[:2] - env.box_pos()[:2]))}
        emit(type="sim_stats", sandbox_id=self.sb.id, phase=self.phase, skill_phase=getattr(policy.ex, "phase", None),
             step=step, joints=[round(float(v), 2) for v in env.world.joint_state()],
             tcp=[round(float(v), 4) for v in env.world.data.site_xpos[env.site]],
             return_so_far=round(episode_return(so_far, self.sb.reward_weights)[0], 4))
        if self.preview and time.time() - self.last_live >= LIVE_EVERY_S:
            self.preview.grab()
            self.last_live = time.time()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        emit(type="rl_phase", sandbox_id=self.sb.id, phase=phase)

    def throughput(self) -> float:
        return round(self.sim_steps / max(self.sim_s, 1e-6), 1)

    def record_throughput(self, n: int = 3) -> float:
        """Steps/s when recording a demo: the render-free step plus the observation and preview renders (measured)."""
        t0 = time.time()
        for _ in range(n):
            self.env._obs()
            if self.preview:
                self.preview.r.render(self.env.world.data)
        per_step = 1 / max(self.throughput(), 1e-6) + (time.time() - t0) / n
        return round(1 / per_step, 2)

    def evaluate(self, params: dict[str, float], seeds: list[int]) -> list[tuple[float, bool]]:
        return [(ep["return"], ep["success"]) for ep in (self.rollout(params, s) for s in seeds)]

    def train(self, iterations: int, adr: bool, seed_base: int) -> tuple[dict[str, float], list[IterationStats], str]:
        b = self.sb.budget

        def after(st: IterationStats) -> bool:
            widened = adr and widen(self.sb.env, st.elite_success_rate or 0.0)
            st.env = self.sb.env.model_copy()
            emit(type="rl_iteration", sandbox_id=self.sb.id, widened=widened, **st.model_dump(mode="json"))
            return widened

        return cem(self.sb.params, self.evaluate, iterations, b.population, b.episodes_per_candidate,
                   seed_base, target=self.sb.weakness.target, after_iteration=after, should_stop=self.stopped,
                   rng_seed=self.seed_base, init={p.name: p.init for p in self.sb.params})

    # ------------------------------------------------------------------ modes
    def pilot(self) -> dict[str, Any]:
        b = self.sb.budget
        self.set_phase("pilot")
        seeds = [self.seed_base + PILOT_SEEDS + i for i in range(b.pilot_episodes)]
        base = [self.rollout({}, s)["success"] for s in seeds]
        pilot = Pilot(n=len(base), successes=sum(base), wilson_95=wilson(sum(base), len(base)))
        if b.pilot_cem_iterations and not self.stopped():
            best, _, _ = self.train(b.pilot_cem_iterations, adr=False, seed_base=self.seed_base + PILOT_SEEDS * 2)
            fresh = [self.rollout(best, s + 50_000)["success"] for s in seeds]  # unbiased re-measure of the best
            pilot.cem_iterations, pilot.cem_best_successes, pilot.cem_best_n = b.pilot_cem_iterations, sum(fresh), len(
                fresh)
        out = {"pilot": pilot.model_dump(mode="json"), "sim_steps_per_s": self.throughput(),
               "record_steps_per_s": self.record_throughput(),
               "steps_per_episode": round(self.sim_steps / max(self.episodes, 1))}
        emit(type="rl_pilot", sandbox_id=self.sb.id, **out)
        return out

    def run(self) -> dict[str, Any]:
        b = self.sb.budget
        self.set_phase("train")
        best, history, reason = self.train(b.iterations, adr=True, seed_base=self.seed_base + TRAIN_SEEDS)
        if self.stopped():
            reason = "stopped"
        self.set_phase("heldout")  # matched seeds: the same scenes for base and tuned
        seeds = [self.seed_base + HELDOUT_SEEDS + i for i in range(b.heldout_episodes)]
        base = [self.rollout({}, s)["success"] for s in seeds]
        tuned = [self.rollout(best, s)["success"] for s in seeds]
        only_b = sum(x and not y for x, y in zip(base, tuned))
        only_t = sum(y and not x for x, y in zip(base, tuned))
        demo_ids = [] if reason == "stopped" else self.demos(best, b.demos, b.dart_noise_frac)
        res = SandboxResult(heldout_seeds=seeds, base_success=sum(base) / len(seeds),
                            base_ci95=wilson(sum(base), len(seeds)), tuned_success=sum(tuned) / len(seeds),
                            tuned_ci95=wilson(sum(tuned), len(seeds)), p_value=mcnemar_exact(only_b, only_t),
                            discordant=(only_b, only_t), params=best, final_env=self.sb.env, stop_reason=reason,
                            demo_episode_ids=demo_ids)
        out = {"result": res.model_dump(mode="json"), "progress": [h.model_dump(mode="json") for h in history],
               "sim_steps_per_s": self.throughput()}
        emit(type="rl_result", sandbox_id=self.sb.id, **out)
        return out

    def demos(self, params: dict[str, float], n: int, dart_frac: float) -> list[str]:
        """Tuned-skill rollouts on fresh seeds; a ``dart_frac`` share with perturbed waypoints. Successes only."""
        if n == 0:
            return []
        from PIL import Image

        self.set_phase("demos")
        kept: list[str] = []
        n_dart = int(round(n * dart_frac))
        for k in range(n * 3):  # attempts cap: give up after 3x the requested demos
            if len(kept) >= n or self.stopped():
                break
            dart = len(kept) >= n - n_dart  # the last n_dart demos are the noised ones
            seed = self.seed_base + DEMO_SEEDS + k
            ep = self.rollout(params, seed, record=True, dart=dart)
            buf, video = ep.pop("buf"), ep.pop("video")
            if not ep["success"]:
                continue
            d = self.out / "demos" / f"{len(kept):04d}"
            d.mkdir(parents=True, exist_ok=True)
            m = len(buf["action"])
            env = self.env
            np.savez_compressed(
                d / "episode.npz", front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
                state=np.stack(buf["state"]).astype(np.float32), action=np.stack(buf["action"]).astype(np.float32),
                reward=np.asarray(buf["reward"], np.float32), timestamp=(np.arange(m) / FPS).astype(np.float32),
                action_low=np.r_[np.rad2deg(env.world.lo[:5]), 0].astype(np.float32),
                action_high=np.r_[np.rad2deg(env.world.hi[:5]), 100].astype(np.float32))
            write_hd(buf["hd"], d)
            if video:
                write_mp4(video, d / "front.mp4")
                Image.fromarray(video[min(len(video) - 1, 45)]).save(d / "thumb.jpg", quality=85)
            meta = {**ep, "num_frames": m, "fps": FPS, "out_dir": str(d), "dart": dart, "params": params,
                    "condition": self.sb.env.condition.model_dump(mode="json"),
                    "provenance": {"planner": f"rl_sandbox:{self.sb.id}", "mode": "privileged",
                                   "method": self.sb.method, "dart": dart, "params": params,
                                   "action_labels": "clean skill command" if dart else "executed command"}}
            (d / "result.json").write_text(json.dumps(meta))
            kept.append(d.name)
        return kept

    def close(self) -> None:
        if self.preview:
            self.preview.close()
        self.env.close()


def main() -> None:
    configure_gl()
    cfg = json.loads(Path(sys.argv[1]).read_text())
    try:
        r = Runner(cfg)
        try:
            res = r.pilot() if cfg["mode"] == "pilot" else r.run()
        finally:
            r.close()
        (Path(cfg["out_dir"]) / "result.json").write_text(json.dumps(res))
        emit(type="result", ok=True)
    except Exception as e:  # report, then fail the process
        emit(type="result", ok=False, **worker_error(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
