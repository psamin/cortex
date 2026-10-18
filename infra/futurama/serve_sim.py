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
    FONT = ImageFont.load_default(size=15)  # readable on a projector at 640x480
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()

TITLES = {"jenga": "Jenga: extract the marked block", "screwdriver": "Factory: unscrew a screw with a screwdriver",
          "unscrew": "Factory: unscrew the bolt", "unscrew_black": "Factory: unscrew the bolt (policy 2)", "palletize": "Factory: unload boxes from the forklift",
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
                "success_rate": r["success_rate"], "eval_episodes": r["episodes"], "eval_seeds": r["seeds"],
                "layouts": r.get("layouts"), "layout_seeds": r.get("layout_seeds")}
        if task in unscored or (cand["success_rate"], cand["step"]) > (best[task]["success_rate"], best[task]["step"]):
            best[task] = cand
            unscored.discard(task)
    return best


def load(sim_dir: Path) -> dict[str, dict[str, Any]]:
    """Every task's best checkpoint on the GPU, each with its own sim. Runs in the sim process (see Sims)."""
    from cortex.sim.policies import LeRobotPolicy
    from cortex.sim.tasks import make

    out = {}
    for task, info in best_checkpoints(sim_dir).items():
        meta = sim_dir / task / "dataset" / "meta" / "cortex_sim.json"  # practice layouts it was trained on, if any
        m = json.loads(meta.read_text()) if meta.exists() else {}
        info["layouts"] = info.get("layouts") or m.get("layouts")
        info["layout_seeds"] = info.get("layout_seeds") or m.get("layout_seeds")
        env, _ = make(task, 240, 320)
        policy = LeRobotPolicy(info["checkpoint"], device="cuda", task=env.instruction,
                               cameras=CAMERAS.get(info["variant"]))
        out[task] = {**info, "task": task, "policy": policy, "env": env, "instruction": env.instruction,
                     "max_steps": env.max_steps}
        where = f"{info['layouts']} practice layouts" if info.get("layouts") else "unseen layouts"
        score = (f"{info['success_rate']:.0%} on {info['eval_episodes']} runs over {where}"
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


def _inset(img: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """The policy's wrist camera in the bottom-right corner of the live view."""
    out = img.copy()
    h, w = out.shape[0] // 4, out.shape[1] // 4
    small = np.asarray(Image.fromarray(wrist).resize((w, h)))
    y, x = out.shape[0] - h - 10, out.shape[1] - w - 10
    out[y - 2:y + h + 2, x - 2:x + w + 2] = 235
    out[y:y + h, x:x + w] = small
    return out


def _overlay(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> bytes:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    w = max(d.textlength(t, font=FONT) for t, _ in lines) + 32
    d.rectangle([8, 8, 8 + w, 16 + 20 * len(lines)], fill=(14, 14, 14))
    for i, (t, color) in enumerate(lines):
        d.text((16, 12 + 20 * i), t, fill=color, font=FONT)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)
    return buf.getvalue()


VIEWS = ("both", "front", "wrist")


def _view(obs: dict[str, Any], view: str) -> np.ndarray:
    """The live view, from the policy's own cameras (already rendered every step), scaled up. A second MuJoCo
    renderer in this process drew garbage under headless EGL, so the live view adds no renderer of its own."""
    cam = obs["pixels"]["wrist" if view == "wrist" else "front"]
    img = np.asarray(Image.fromarray(cam).resize((640, 480), Image.BILINEAR))
    return _inset(img, obs["pixels"]["wrist"]) if view == "both" else img


def _episode(p: dict[str, Any], seed: int, stop, view: str = "both",
             fps: int = 30) -> Iterator[tuple[bytes, int, float, bool]]:
    """(jpeg, step, inference ms, success) per shown frame, at real time, until done, out of steps or stopped."""
    import torch
    from cortex.sim.tasks.run import SETTLE

    policy, env = p["policy"], p["env"]
    if p.get("layout_seeds"):
        seed = p["layout_seeds"][seed % len(p["layout_seeds"])]  # a practice layout it was trained on
    elif p.get("layouts"):
        seed = seed % p["layouts"]  # a practice layout it was trained on (demo reliability over generalization)
    else:
        seed = visible_seed(p["task"], env, seed)
    obs = env.reset(seed)
    policy.reset(env, obs, seed)
    held = 0
    t_act = t_step = t_frame = 0.0
    steps = 0
    success = False
    for step in range(env.max_steps):
        if stop():
            break
        t0 = time.perf_counter()
        action = policy.act(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
        shown = step % 2 == 0  # 15 fps of video is plenty and halves the bandwidth; the policy still runs every step
        frame = _view(obs, view) if shown else None
        t2 = time.perf_counter()
        obs, _, success, _, _ = env.step(action)
        t3 = time.perf_counter()
        t_act, t_step, t_frame, steps = t_act + t1 - t0, t_step + t3 - t2, t_frame + t2 - t1, steps + 1
        held = held + 1 if success else 0  # keep showing it a few seconds after the verified success
        if frame is not None:
            status = ("Task complete (verified by the simulator)", (110, 220, 130)) if success else (
                "Running", (200, 200, 200))
            jpg = _overlay(frame, [
                (TITLES.get(p["task"], p["task"]), (240, 240, 240)),
                (f"{p['variant'].upper()} trained only in sim  ·  "
                 + (f"practice layout {seed}" if p.get("layouts") or p.get("layout_seeds")
                    else f"unseen layout (seed {seed})"), (160, 200, 255)),
                (f"step {step}  ·  inference {ms:5.1f} ms on GPU", (160, 200, 255)),
                status])
            yield jpg, step, ms, success
        if held >= 3 * SETTLE:
            break
        time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
    n = max(steps, 1)
    print(f"[serve] sim {p['task']} layout {seed}: {steps} steps, success={success}, per step: act "
          f"{t_act / n * 1000:.1f} ms, sim+render {t_step / n * 1000:.1f} ms, frame {t_frame / n * 1000:.1f} ms",
          flush=True)


# ---------------------------------------------------------------- the sims run in their own process
#
# MuJoCo's renderer aborted the whole server (SIGABRT in Renderer.render) when it ran on a worker thread next to
# the HTTP threads, and every stream died with it until the job was resubmitted (#307). So the policies and sims
# live in a child process that does everything on its main thread, like the recording and evaluation jobs, and if
# it dies anyway the next run restarts it. Runs are numbered: starting one bumps the number, and the child drops
# any run that is no longer current (a newer run, or a viewer that left).

import multiprocessing as mp  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402

_MP = mp.get_context("spawn")  # never fork a process that holds CUDA / EGL state
INFO = ("task", "variant", "step", "success_rate", "eval_episodes", "eval_seeds", "instruction", "max_steps",
        "layouts", "layout_seeds")
LOOP = 50  # layouts per stream: after each episode it resets onto the next practice layout


def _put(q, item, stop) -> bool:
    while not stop():
        try:
            q.put(item, timeout=1)
            return True
        except queue.Full:
            continue
    return False


def _child(sim_dir: str, jobs, frames, info, gen) -> None:
    import torch

    torch.set_num_threads(4)
    policies = load(Path(sim_dir))
    info.put({t: {k: p.get(k) for k in INFO} for t, p in policies.items()})
    while True:
        g, task, seed, view = jobs.get()

        def stop(g=g):
            return gen.value != g

        try:
            for i in range(LOOP):
                for item in _episode(policies[task], seed + i, stop, view):
                    if not _put(frames, (g, item), stop):
                        break
                if stop():
                    break
        except Exception as e:  # noqa: BLE001 - report it, keep the process serving
            print(f"[serve] sim {task} run failed: {type(e).__name__}: {e}", flush=True)
        _put(frames, (g, None), stop)


class Sims:
    """The HTTP server's handle on the sim process."""

    def __init__(self, sim_dir: Path):
        self.sim_dir = str(sim_dir)
        self.gen = _MP.Value("i", 0)
        self.lock = threading.Lock()
        self.info: dict[str, dict[str, Any]] = {}
        self._spawn()

    def _spawn(self) -> None:
        self.jobs, self.frames, info = _MP.Queue(), _MP.Queue(maxsize=30), _MP.Queue()
        self.proc = _MP.Process(target=_child, args=(self.sim_dir, self.jobs, self.frames, info, self.gen),
                                daemon=True, name="sims")
        self.proc.start()
        while True:
            try:
                self.info = info.get(timeout=5)
                return
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"the sim process exited ({self.proc.exitcode}) while loading") from None

    def run(self, task: str, seed: int, view: str = "both") -> Iterator[tuple[bytes, int, float, bool]]:
        """Frames of a new run; any older run stops. Ends when the run does, or when the caller closes it."""
        with self.lock:
            if not self.proc.is_alive():
                print(f"[serve] sim process exited ({self.proc.exitcode}); restarting it", flush=True)
                self._spawn()
            with self.gen.get_lock():
                self.gen.value += 1
                mine = self.gen.value
            frames = self.frames
            self.jobs.put((mine, task, seed, view if view in VIEWS else "both"))
        idle = 0
        try:
            while self.gen.value == mine:
                try:
                    g, item = frames.get(timeout=1)
                except queue.Empty:
                    idle += 1
                    if idle > 120 or not self.proc.is_alive():
                        return
                    continue
                idle = 0
                if g != mine:
                    continue  # left over from an older run
                if item is None:
                    return
                yield item
        finally:
            with self.gen.get_lock():
                if self.gen.value == mine:
                    self.gen.value += 1  # the viewer left: stop the run
