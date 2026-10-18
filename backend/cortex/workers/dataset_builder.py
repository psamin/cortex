"""Build a LeRobotDataset v3 directory from recorded episodes.

    python -m cortex.workers.dataset_builder <config.json>

config: {root, repo_id, fps, task, items: [{npz, augmentation|null, aug_seed}], lineage: {...}}
Augmentations change pixels only; state and action arrays are copied unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..errors import worker_error

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def emit(**kw) -> None:
    print(json.dumps(kw), flush=True)


def main() -> None:
    from lerobot.datasets import LeRobotDataset

    from ..models import Augmentation
    from ..services.augment import apply

    cfg = json.loads(Path(sys.argv[1]).read_text())
    root = Path(cfg["root"])
    first = np.load(cfg["items"][0]["npz"])
    h, w = first["front"].shape[1:3]
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.front": img,
        "observation.images.wrist": img,
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    ds = LeRobotDataset.create(repo_id=cfg["repo_id"], fps=cfg["fps"], features=features, root=root,
                               robot_type="so101_follower", use_videos=True)
    frames_total = 0
    for i, item in enumerate(cfg["items"]):
        with np.load(item["npz"]) as z:
            front, wrist, state, action = z["front"], z["wrist"], z["state"], z["action"]
        if item.get("augmentation"):
            aug = Augmentation.model_validate(item["augmentation"])
            front = apply(front, aug, seed=item.get("aug_seed", 0))
            wrist = apply(wrist, aug, seed=item.get("aug_seed", 0) + 1)
        for t in range(len(action)):
            ds.add_frame({"observation.images.front": front[t], "observation.images.wrist": wrist[t],
                          "observation.state": state[t], "action": action[t], "task": cfg["task"]})
        ds.save_episode(parallel_encoding=False)  # per-episode process pools break intermittently under spawn
        frames_total += len(action)
        emit(type="progress", done=i + 1, total=len(cfg["items"]))
    ds.finalize()
    (root / "meta" / "cortex_lineage.json").write_text(json.dumps(cfg["lineage"], indent=1))
    emit(type="result", ok=True, frames=frames_total, episodes=len(cfg["items"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit(type="result", ok=False, **worker_error(e))
        sys.exit(1)
