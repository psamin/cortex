"""Record oracle demos and evaluate checkpoints on the hand-built sim tasks (issue #161).

    python -m cortex.sim.tasks.run demos <task> <dataset_root> --episodes 200 [--examples <dir>] [--shard i --shards n]
    python -m cortex.sim.tasks.run merge <task> <dataset_root> <shard_root> [<shard_root> ...]
    python -m cortex.sim.tasks.run evaluate <task> <checkpoint_dir> <out_dir> --episodes 50 [--device cuda]

Demos are recorded with DART-style exploration (``--noise-deg``): smooth noise is added to the arm command
that is *executed*, while the expert's clean command is what gets *recorded*. The oracles command
absolute joint targets, so from every nudged state the label points back to the plan, and the policy
learns to recover instead of only ever seeing the perfect path. Episodes the noise breaks are dropped.

Recording is parallel: n shard processes take seeds shard, shard + n, ... and ``merge`` joins their
datasets into one (the rendering, not the physics, is the bottleneck, so this scales with processes).

Demos use seeds 0, 1, ... and keep only the oracle's successful episodes (a LeRobotDataset v3 with
front + wrist video, 6-D state and action). Evaluation runs the checkpoint closed loop on seeds from
100000 up, which training never saw, and writes metrics.json plus videos. ``--examples`` also renders
the first few demos with the presentation camera, plus the policy's own camera views as insets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..so101 import FPS, configure_gl
from . import make

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EVAL_SEED0 = 100_000
SETTLE = 30  # control steps held after the motion ends, so the verifier sees things at rest
GRIP_HALF = 30.0  # gripper command above this is open (expert.GRIP_OPEN = 60, GRIP_CLOSED = 0)


class Nudge:
    """Smooth (Ornstein-Uhlenbeck) noise on the five arm joints, in degrees: full size while the gripper is
    open (reaching), 30% while it is closed (holding / turning), so contact phases are not wrecked."""

    def __init__(self, rng: np.random.Generator, deg: float, theta: float = 0.15):
        self.rng, self.deg, self.theta = rng, deg, theta
        self.x = np.zeros(5)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        self.x += -self.theta * self.x + self.deg * np.sqrt(2 * self.theta) * self.rng.standard_normal(5)
        out = np.array(action, dtype=np.float32, copy=True)
        out[:5] += (1.0 if out[5] > GRIP_HALF else 0.3) * self.x
        return out


def _size(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x")
    return int(h), int(w)


class Presenter:
    """The studio's presentation camera aimed at the task, with the policy's two cameras inset."""

    def __init__(self, env):
        from ..viewer import DemoCamera

        self.env, self.cam = env, DemoCamera(env.world.model)
        view = getattr(env, "view", {})  # a task may aim the camera: lookat, distance, azimuth, elevation
        if "lookat" in view:
            self.cam.cam.lookat[:] = view["lookat"]
        for k in ("distance", "azimuth", "elevation"):
            if k in view:
                setattr(self.cam.cam, k, view[k])

    def frame(self, obs: dict[str, Any]) -> np.ndarray:
        from PIL import Image

        img = self.cam.render(self.env.world.data)
        H, W = img.shape[:2]
        h, w = H // 4, W // 4
        for i, name in enumerate(("front", "wrist")):
            inset = np.asarray(Image.fromarray(obs["pixels"][name]).resize((w, h)))
            y, x = H - (2 - i) * (h + 12), W - w - 12
            img[y - 2:y + h + 2, x - 2:x + w + 2] = 235
            img[y:y + h, x:x + w] = inset
        return img

    def close(self) -> None:
        self.cam.close()


def episode(env, policy, seed: int, record: bool, presenter: Presenter | None = None,
            noise_deg: float = 0.0) -> dict[str, Any]:
    obs = env.reset(seed)
    nudge = Nudge(np.random.default_rng(seed + 7919), noise_deg) if noise_deg > 0 else None
    policy.reset(env, obs, seed)
    render = record or presenter is not None or getattr(policy, "needs_pixels", True)
    rows: list[tuple] = []
    video: list[np.ndarray] = []
    success, held, t0 = False, 0, time.time()
    for _ in range(env.max_steps):
        action = np.asarray(policy.act(obs), dtype=np.float32)
        if record:
            rows.append((obs["pixels"]["front"], obs["pixels"]["wrist"],
                         np.asarray(obs["agent_pos"], np.float32), action))
        if presenter is not None:
            video.append(presenter.frame(obs))
        nxt, _, success, _, _ = env.step(nudge(action) if nudge else action, render=render)
        obs = nxt if nxt is not None else obs
        # an expert demo runs its whole motion (back out, return home); a learned policy stops on success
        finished = policy.done if hasattr(policy, "done") else success
        held = held + 1 if finished else 0
        if held >= SETTLE:
            break
    return {"seed": seed, "success": bool(success), "steps": env.steps, "wall_s": round(time.time() - t0, 2),
            **env.metrics(), "rows": rows, "video": video}


def demos(args: argparse.Namespace) -> None:
    from lerobot.datasets import LeRobotDataset

    from ...workers.sim_worker import write_mp4

    env, oracle = make(args.task, *_size(args.size))
    root = Path(args.root)
    img = {"dtype": "video", "shape": (*_size(args.size), 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=f"cortex/sim_{args.task}", fps=FPS, root=root, robot_type="so101_follower",
                               use_videos=True, features={
                                   "observation.images.front": img, "observation.images.wrist": img,
                                   "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
                                   "action": {"dtype": "float32", "shape": (6,), "names": JOINTS}})
    ex_dir = Path(args.examples) if args.examples else None
    if ex_dir:
        ex_dir.mkdir(parents=True, exist_ok=True)
    kept, failed, seed = [], [], args.shard
    want = -(-args.episodes // args.shards)  # this shard's share, rounded up
    presenter = None
    if ex_dir:
        env.reset(0)  # the presentation renderer needs the model
        presenter = Presenter(env)
    while len(kept) < want and seed < args.episodes * 2:
        example = presenter is not None and len(kept) < args.n_examples  # example videos show the clean expert
        ep = episode(env, oracle, seed, record=True, presenter=presenter if example else None,
                     noise_deg=0.0 if example else args.noise_deg)
        if ep["success"]:
            for front, wrist, state, action in ep["rows"]:
                ds.add_frame({"observation.images.front": front, "observation.images.wrist": wrist,
                              "observation.state": state, "action": action, "task": env.instruction})
            ds.save_episode(parallel_encoding=False)
            if ep["video"]:
                write_mp4(ep["video"], ex_dir / f"demo_seed{seed}.mp4")
            kept.append(seed)
        else:
            failed.append(seed)
        print(json.dumps({"seed": seed, "success": ep["success"], "steps": ep["steps"], "kept": len(kept)}), flush=True)
        seed += args.shards
    ds.finalize()
    (root / "meta" / "cortex_sim.json").write_text(json.dumps({
        "task": args.task, "instruction": env.instruction, "size": args.size, "noise_deg": args.noise_deg,
        "kept_seeds": kept,
        "failed_seeds": failed, "oracle_success_rate": len(kept) / max(1, len(kept) + len(failed))}, indent=1))
    if presenter:
        presenter.close()
    env.close()


def merge(args: argparse.Namespace) -> None:
    """Join shard datasets into one, with their oracle statistics combined."""
    from lerobot.datasets.aggregate import aggregate_datasets

    roots = [Path(r) for r in args.shards_roots]
    aggregate_datasets([f"cortex/sim_{args.task}_{i}" for i in range(len(roots))], f"cortex/sim_{args.task}",
                       roots=roots, aggr_root=Path(args.root))
    metas = [json.loads((r / "meta" / "cortex_sim.json").read_text()) for r in roots]
    kept = sorted(s for m in metas for s in m["kept_seeds"])
    failed = sorted(s for m in metas for s in m["failed_seeds"])
    (Path(args.root) / "meta" / "cortex_sim.json").write_text(json.dumps({
        **{k: metas[0].get(k) for k in ("task", "instruction", "size", "noise_deg")}, "kept_seeds": kept,
        "failed_seeds": failed,
        "oracle_success_rate": len(kept) / max(1, len(kept) + len(failed)), "shards": len(roots)}, indent=1))
    print(json.dumps({"merged_episodes": len(kept), "failed": len(failed)}), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    from ...workers.sim_worker import write_mp4
    from ..policies import LeRobotPolicy

    env, _ = make(args.task, *_size(args.size))
    cams = dict(kv.split("=") for kv in args.cameras.split(",")) if args.cameras else None
    policy = LeRobotPolicy(args.checkpoint, device=args.device, task=env.instruction, cameras=cams)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env.reset(EVAL_SEED0)
    presenter = Presenter(env)
    results = []
    for i in range(args.episodes):
        seed = EVAL_SEED0 + i
        ep = episode(env, policy, seed, record=False, presenter=presenter if i < args.videos else None)
        ep.pop("rows")
        video = ep.pop("video")
        if video:
            ep["video"] = f"seed{seed}_{'ok' if ep['success'] else 'fail'}.mp4"
            write_mp4(video, out / ep["video"])
        results.append(ep)
        print(json.dumps(ep), flush=True)
    rate = sum(r["success"] for r in results) / len(results)
    (out / "metrics.json").write_text(json.dumps({
        "task": args.task, "checkpoint": str(args.checkpoint), "policy": policy.name, "episodes": len(results),
        "success_rate": rate, "seeds": f"{EVAL_SEED0}..{EVAL_SEED0 + args.episodes - 1}", "results": results}, indent=1))
    print(json.dumps({"success_rate": rate}), flush=True)
    presenter.close()
    env.close()


def main() -> None:
    configure_gl()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demos")
    d.add_argument("task")
    d.add_argument("root")
    d.add_argument("--episodes", type=int, default=200)
    d.add_argument("--size", default="240x320")
    d.add_argument("--examples")
    d.add_argument("--n-examples", type=int, default=3)
    d.add_argument("--noise-deg", type=float, default=0.0, help="DART exploration noise on the executed arm command")
    d.add_argument("--shard", type=int, default=0)
    d.add_argument("--shards", type=int, default=1)
    m = sub.add_parser("merge")
    m.add_argument("task")
    m.add_argument("root")
    m.add_argument("shards_roots", nargs="+")
    e = sub.add_parser("evaluate")
    e.add_argument("task")
    e.add_argument("checkpoint")
    e.add_argument("out")
    e.add_argument("--episodes", type=int, default=50)
    e.add_argument("--size", default="240x320")
    e.add_argument("--videos", type=int, default=4)
    e.add_argument("--device", default="cuda")
    e.add_argument("--cameras", help="policy=env camera names, e.g. camera1=front,camera2=wrist")
    args = p.parse_args()
    {"demos": demos, "merge": merge, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
