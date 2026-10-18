"""Live closed-loop inference for the sim-trained policies (issue #161), served by serve_policy.py.

For each hand-built sim task with a trained policy under <sim_dir>/<task>/runs/<variant>/, the
checkpoint with the best closed-loop evaluation is loaded on the GPU. `stream(...)` then runs it in
MuJoCo on a layout it never saw (any seed), rendering the presentation camera with the policy's own
two cameras inset, and yields JPEG frames with what is happening: step, inference time and whether
the simulator's verifier says the task is done.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    FONT = ImageFont.load_default(size=24)  # readable on a projector at 1280x960
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()

TITLES = {"jenga": "Jenga: extract the marked block", "screwdriver": "Factory: unscrew a screw with a screwdriver",
          "unscrew": "Factory: unscrew the bolt", "palletize": "Factory: unload boxes from the forklift",
          "suture": "Suture: pass the needle through the eyelets"}
CAMERAS = {"smolvla": {"camera1": "front", "camera2": "wrist"}}  # smolvla_base's camera names -> the sim's


def best_checkpoints(sim_dir: Path) -> dict[str, dict[str, Any]]:
    """Per task, the evaluated checkpoint with the highest closed-loop success rate; a task whose runs are
    trained but not evaluated yet serves its newest checkpoint, marked score pending (success_rate None)."""
    best: dict[str, dict[str, Any]] = {}
    for ck in sorted(sim_dir.glob("*/runs/*/checkpoints/*/pretrained_model"), key=lambda c: c.stat().st_mtime):
        if (ck / "config.json").exists():
            task, variant, step = ck.parts[-6], ck.parts[-4], ck.parts[-2]
            best[task] = {"task": task, "variant": variant, "step": int(step), "checkpoint": str(ck),
                          "success_rate": None, "eval_episodes": 0, "eval_seeds": ""}
    unscored = set(best)
    for m in sim_dir.glob("*/runs/*/eval/*/metrics.json"):
        task, variant, step = m.parts[-6], m.parts[-4], m.parts[-2]
        r = json.loads(m.read_text())
        ck = sim_dir / task / "runs" / variant / "checkpoints" / step / "pretrained_model"
        if not (ck / "config.json").exists():
            continue
        cand = {"task": task, "variant": variant, "step": int(step), "checkpoint": str(ck),
                "success_rate": r["success_rate"], "eval_episodes": r["episodes"], "eval_seeds": r["seeds"]}
        if task in unscored or (cand["success_rate"], cand["step"]) > (best[task]["success_rate"], best[task]["step"]):
            best[task] = cand
            unscored.discard(task)
    return best


def load(sim_dir: Path) -> dict[str, dict[str, Any]]:
    from cortex.sim.policies import LeRobotPolicy
    from cortex.sim.tasks import make

    out = {}
    for task, info in best_checkpoints(sim_dir).items():
        env, _ = make(task, 240, 320)
        policy = LeRobotPolicy(info["checkpoint"], device="cuda", task=env.instruction,
                               cameras=CAMERAS.get(info["variant"]))
        out[task] = {**info, "env": env, "policy": policy, "instruction": env.instruction}
        score = (f"{info['success_rate']:.0%} on {info['eval_episodes']} unseen layouts"
                 if info["success_rate"] is not None else "score pending")
        print(f"[serve] sim {task}: {info['variant']} step {info['step']}, {score}", flush=True)
    return out


def visible_seed(task: str, env, seed: int) -> int:
    """For the live demo, the bolt in its brightest trained finish (brass) so people can see it. Brass is one of
    the finishes the policy was trained on, so this stays in distribution: no retraining. Picks the first seed
    from the requested one whose layout has a brass bolt (deterministic)."""
    if task != "unscrew":
        return seed
    for s in range(seed, seed + 60):
        env.reset(s)
        if getattr(env, "finish", "") == "brass":
            return s
    return seed


def _overlay(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> bytes:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = max(d.textlength(t, font=FONT) for t, _ in lines) + 32
    d.rectangle([16, 16, 16 + w, 28 + 32 * len(lines)], fill=(14, 14, 14))
    for i, (t, color) in enumerate(lines):
        d.text((32, 26 + 32 * i), t, fill=color, font=FONT)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def stream(p: dict[str, Any], seed: int, fps: int = 30) -> Iterator[tuple[bytes, int, float, bool]]:
    """(jpeg, step, inference ms, success) per control step, at real time, until done or out of steps."""
    import torch
    from cortex.sim.tasks.run import SETTLE, Presenter

    env, policy = p["env"], p["policy"]
    seed = visible_seed(p["task"], env, seed)
    obs = env.reset(seed)
    policy.reset(env, obs, seed)
    presenter = Presenter(env)
    held = 0
    try:
        for step in range(env.max_steps):
            t0 = time.perf_counter()
            action = policy.act(obs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            frame = presenter.frame(obs)
            obs, _, success, _, _ = env.step(action)
            held = held + 1 if success else 0  # keep showing it a few seconds after the verified success
            status = ("Task complete (verified by the simulator)", (110, 220, 130)) if success else (
                "Running", (200, 200, 200))
            jpg = _overlay(frame, [
                (TITLES.get(p["task"], p["task"]), (240, 240, 240)),
                (f"{p['variant'].upper()} trained only in sim  ·  unseen layout (seed {seed})", (160, 200, 255)),
                (f"step {step}  ·  inference {ms:5.1f} ms on GPU", (160, 200, 255)),
                status])
            yield jpg, step, ms, success
            if held >= 3 * SETTLE:
                break
            time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
    finally:
        presenter.close()
