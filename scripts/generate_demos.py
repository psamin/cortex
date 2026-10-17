"""Deterministically generate a sample LeRobotDataset v3 of SO-101 pick-and-place demos.

    uv --project backend run python scripts/generate_demos.py --out sample_data --per-condition 2

Same seeds -> same demos: the scripted IK expert drives the pinned EnvHub MuJoCo
env, then the Cortex dataset builder writes a LeRobot v3 dataset with lineage.
No backend or database needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
CONDITIONS = [("left", "normal"), ("center", "normal"), ("right", "normal"), ("center", "dim"), ("center", "bright")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("sample_data"))
    ap.add_argument("--per-condition", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=10_000)
    a = ap.parse_args()
    out = a.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    raw = out / "raw"
    episodes = []
    for i, (region, light) in enumerate(c for c in CONDITIONS for _ in range(a.per_condition)):
        episodes.append({"seed": a.base_seed + i, "condition": {"cube_region": region, "lighting": light},
                         "out_dir": str(raw / f"ep{i:03d}")})
    rec = {"mode": "record", "max_steps": 240, "live_dir": str(raw / "live"), "episodes": episodes}
    (out).mkdir(parents=True)
    (out / "record.json").write_text(json.dumps(rec))
    print(f"recording {len(episodes)} demos ...", flush=True)
    subprocess.run([sys.executable, "-m", "cortex.workers.sim_worker", str(out / "record.json")],
                   cwd=BACKEND, check=True, stdout=subprocess.DEVNULL)
    results = [json.loads((Path(e["out_dir"]) / "result.json").read_text()) for e in episodes]
    ok = [r["success"] for r in results]
    print(f"expert success: {sum(ok)}/{len(ok)}")

    build = {
        "root": str(out / "lerobot"), "repo_id": "cortex/sample_so101_pick_place", "fps": 30,
        "task": "pick up the cube and place it in the bowl",
        "items": [{"npz": str(Path(e["out_dir"]) / "episode.npz"), "augmentation": None} for e in episodes],
        "lineage": {"generator": "scripts/generate_demos.py", "env": "lerobot/so101-tabletop-mujoco@12e504e8",
                    "episodes": [{**e, "success": r["success"]} for e, r in zip(episodes, results)]},
    }
    (out / "build.json").write_text(json.dumps(build))
    subprocess.run([sys.executable, "-m", "cortex.workers.dataset_builder", str(out / "build.json")],
                   cwd=BACKEND, check=True, stdout=subprocess.DEVNULL)
    print(f"LeRobotDataset v3 written to {out / 'lerobot'}")


if __name__ == "__main__":
    main()
