"""Import human/robot demonstrations from an existing LeRobot dataset.

    python -m cortex.workers.import_worker <config.json>

config: {source: {"hub": "<repo_id>", "revision"?: str} | {"path": "<dir with meta/info.json>"},
         root: <download dir for hub>, max_episodes: int, out_dir: <per-episode output base>, token?: str}
Validates the dataset by loading it with LeRobot, then for each episode writes:
  episode.npz  front/wrist (first two cameras, 96x128 for Cortex's checks), state, action, timestamp
  front.mp4    the first camera at 640x480 for the viewer
  thumb.jpg
and a result.json with the dataset's metadata (fps, robot type, cameras, task text).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import worker_error


def emit(**kw: Any) -> None:
    print(json.dumps(kw, default=str), flush=True)


def to_hwc_uint8(img) -> np.ndarray:
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW float [0,1]
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def resize(a: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(a).resize((w, h), Image.BILINEAR))


def main() -> None:
    from lerobot.datasets import LeRobotDataset

    from .sim_worker import write_mp4

    cfg = json.loads(Path(sys.argv[1]).read_text())
    src = cfg["source"]
    n_max = int(cfg.get("max_episodes", 10))
    if "hub" in src:
        meta_ds = LeRobotDataset(src["hub"], root=cfg["root"], revision=src.get("revision"), episodes=[0])
        total = meta_ds.meta.total_episodes
        eps = list(range(min(n_max, total)))
        ds = LeRobotDataset(src["hub"], root=cfg["root"], revision=src.get("revision"), episodes=eps)
        repo, root = src["hub"], cfg["root"]
    else:
        root = src["path"]
        info = json.loads((Path(root) / "meta" / "info.json").read_text())
        total = int(info["total_episodes"])
        eps = list(range(min(n_max, total)))
        ds = LeRobotDataset("local/imported", root=root, episodes=eps)
        repo = f"local:{Path(root).name}"
    feats = ds.meta.features
    cams = [k for k, v in feats.items() if v.get("dtype") in ("video", "image")]
    if "action" not in feats or "observation.state" not in feats or not cams:
        raise ValueError(f"not a manipulation dataset Cortex can use (features: {list(feats)})")
    t = getattr(ds.meta, "tasks", None)  # v3: DataFrame indexed by task text; v2: dict
    tasks = [str(x) for x in (t.index if hasattr(t, "index") else (t or []))]
    emit(type="meta", repo=repo, fps=ds.meta.fps, robot_type=ds.meta.robot_type, cameras=cams, total_episodes=total,
         importing=len(eps), action_dim=feats["action"]["shape"], codebase_version=ds.meta.info.get("codebase_version"))

    # frame index ranges per episode (dataset rows are ordered by episode)
    by_ep: dict[int, list[int]] = {}
    for i, e in enumerate(ds.hf_dataset["episode_index"]):
        by_ep.setdefault(int(e), []).append(i)

    out_base = Path(cfg["out_dir"])
    results = []
    for k, ep in enumerate(sorted(by_ep)):
        idx = by_ep[ep]
        front, wrist, state, action, ts, video = [], [], [], [], [], []
        task = ""
        for i in idx:
            item = ds[i]
            f0 = to_hwc_uint8(item[cams[0]])
            f1 = to_hwc_uint8(item[cams[1]]) if len(cams) > 1 else f0
            front.append(resize(f0, 96, 128))
            wrist.append(resize(f1, 96, 128))
            video.append(resize(f0, 480, 640))
            state.append(np.asarray(item["observation.state"], np.float32))
            action.append(np.asarray(item["action"], np.float32))
            ts.append(float(item["timestamp"]))
            task = item.get("task", task) or task
        out = out_base / f"ep{ep:04d}"
        out.mkdir(parents=True, exist_ok=True)
        a = np.stack(action)
        np.savez_compressed(out / "episode.npz", front=np.stack(front), wrist=np.stack(wrist),
                            state=np.stack(state), action=a, reward=np.zeros(len(a), np.float32),
                            timestamp=np.asarray(ts, np.float32),
                            action_low=a.min(axis=0) - 1e-3, action_high=a.max(axis=0) + 1e-3)
        write_mp4(video, out / "front.mp4", fps=int(ds.meta.fps))
        from PIL import Image

        Image.fromarray(video[len(video) // 2]).save(out / "thumb.jpg", quality=85)
        r = {"source_episode_index": ep, "num_frames": len(idx), "out_dir": str(out), "task": task}
        results.append(r)
        emit(type="episode", index=k, total=len(by_ep), **r)
    (out_base / "result.json").write_text(json.dumps({"repo": repo, "root": str(root), "fps": ds.meta.fps,
                                                       "robot_type": ds.meta.robot_type, "cameras": cams,
                                                       "tasks": tasks, "episodes": results}, default=str))
    emit(type="result", ok=True, episodes=len(results))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:

        emit(type="result", ok=False, **worker_error(e))
        sys.exit(1)
