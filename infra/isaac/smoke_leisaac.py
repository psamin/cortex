"""Isaac Lab smoke test: build a LeIsaac SO-101 scene headless, step it, save camera videos."""
import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="LeIsaac-SO101-PickOrange-v0")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
t_launch = time.time()
app = AppLauncher(vars(args)).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import leisaac  # noqa: E402,F401  (registers the tasks)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("so101leader")
cfg.recorders = None
if hasattr(cfg.terminations, "time_out"):
    cfg.terminations.time_out = None
env = gym.make(args.task, cfg=cfg).unwrapped
obs, _ = env.reset()
cams = [k for k, s in env.scene.sensors.items() if isinstance(s, Camera)]
writers = {k: imageio.get_writer(f"{args.out}/{args.task}_{k}.mp4", fps=30, quality=8) for k in cams}
t0 = time.time()
for i in range(args.steps):
    a = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    a[0, 0] = 0.6 * math.sin(i / 25)          # shoulder pan sweep
    a[0, 1] = -0.4 * (1 - math.cos(i / 35))   # shoulder lift
    a[0, 2] = 0.4 * (1 - math.cos(i / 35))    # elbow
    a[0, 5] = 0.5 * (1 + math.sin(i / 15))    # gripper
    obs, *_ = env.step(a)
    for k in cams:
        writers[k].append_data(env.scene.sensors[k].data.output["rgb"][0, ..., :3].cpu().numpy())
dt = time.time() - t0
for w in writers.values():
    w.close()
report = {"task": args.task, "cameras": {k: list(env.scene.sensors[k].image_shape) for k in cams},
          "action_dim": env.action_manager.total_action_dim, "steps": args.steps,
          "step_fps": round(args.steps / dt, 1), "launch_s": round(t0 - t_launch, 1),
          "gpu": torch.cuda.get_device_name(0)}
print("[isaac] REPORT " + json.dumps(report), flush=True)
open(f"{args.out}/{args.task}.json", "w").write(json.dumps(report, indent=1))
env.close()
app.close()
