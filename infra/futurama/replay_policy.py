"""Run a trained SmolVLA policy on held-out real demos and render what it does (run inside a GPU job).

    python replay_policy.py <task> <checkpoint_dir> <data_root> <out_dir>

For each held-out episode (never trained on), the policy gets the recorded camera images and arm
state frame by frame, exactly as on the robot, and its chosen joint targets are compared with what
the human operator did. Writes ep<idx>.mp4 (camera view + predicted vs human joint traces drawn
live) and metrics.json (mean absolute joint error in degrees, per episode and overall). This is an
open-loop replay on recordings: the arm is not moved, so it measures imitation, not task success.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402

CAMS = {"jenga": ["main", "secondary_0", "secondary_1"], "conveyor": ["top", "wrist"],
        "surgical": ["front", "side"], "needle": ["external", "wrist"]}
JOINTS = ["shoulder pan", "shoulder lift", "elbow flex", "wrist flex", "wrist roll", "gripper"]

task, ckpt, data_root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
stage = "needle" if task == "suture" else task
out.mkdir(parents=True, exist_ok=True)
split = json.loads((data_root / "split.json").read_text())[stage]
cfg_type = json.loads((ckpt / "config.json").read_text())["type"]
policy = get_policy_class(cfg_type).from_pretrained(str(ckpt)).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt),
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
rename = {f"observation.images.{c}": f"observation.images.camera{i + 1}" for i, c in enumerate(CAMS[stage])}

metrics = {"task": task, "checkpoint": str(ckpt), "episodes": {}}
for ep in split["held_out"]:
    ds = LeRobotDataset(f"local/{stage}", root=data_root / stage, episodes=[ep])
    policy.reset()
    pred, human, frames = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        batch = {rename.get(k, k): (v[None] if torch.is_tensor(v) else v) for k, v in item.items()
                 if k.startswith("observation.")}
        batch["task"] = [item["task"]]
        with torch.inference_mode():
            a = post(policy.select_action(pre(batch)))
        pred.append(a.squeeze(0).float().cpu().numpy())
        human.append(item["action"].numpy())
        if i % 2 == 0:
            frames.append((item[f"observation.images.{CAMS[stage][0]}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    pred, human = np.array(pred), np.array(human)
    err = np.abs(pred - human)
    metrics["episodes"][ep] = {"frames": len(pred), "joint_mae_deg": err[:, :5].mean().round(2).item(),
                               "gripper_mae": err[:, 5].mean().round(2).item(), "task": item["task"]}
    # video: camera on the left, predicted (solid) vs human (dashed) joint traces drawn live on the right
    writer = imageio.get_writer(out / f"ep{ep}.mp4", fps=15, codec="libx264", quality=7)
    for k, img in enumerate(frames):
        t = 2 * k + 1
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(11, 4), dpi=90, gridspec_kw={"width_ratios": [1.1, 1.4]})
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title(f"{task}: {item['task'][:60]}", fontsize=9)
        for j in range(6):
            line, = ax.plot(pred[:t, j], lw=1.6, label=JOINTS[j])
            ax.plot(human[:t, j], lw=1.0, ls="--", color=line.get_color())
        ax.set_xlim(0, len(pred))
        ax.set_ylim(min(pred.min(), human.min()) - 5, max(pred.max(), human.max()) + 5)
        ax.set_title("policy (solid) vs human demo (dashed), held-out episode", fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    writer.close()
    print(f"[replay] {task} ep{ep}: {metrics['episodes'][ep]}", flush=True)

eps = metrics["episodes"].values()
metrics["overall_joint_mae_deg"] = round(float(np.mean([e["joint_mae_deg"] for e in eps])), 2)
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(f"[replay] done: {metrics['overall_joint_mae_deg']} deg mean joint error over {len(metrics['episodes'])} episodes")
