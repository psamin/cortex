"""Build a training set from real SO-100/SO-101 LeRobot datasets (run inside the Slurm job).

    python prepare_dataset.py <task> <out_root>      task: jenga | conveyor | suture

Each source is downloaded, converted to LeRobot v3.0 locally when it is v2.1 (never pushed
anywhere), given a real task sentence when its own is missing, and sources of one stage are
merged into <out_root>/<stage>. The last episodes of every source are held out (not trained on)
for the live replay; <out_root>/split.json records them per stage.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

# stage -> [(repo, task sentence to use when the dataset's own is missing/"None", episodes held out)]
TASKS = {
    "jenga": {"jenga": [("jakmilller/jenga-v2", "push a wooden block out of the Jenga tower without tipping it", 3),
                        ("Mahanthesh0r/jenga_pull", "pull a wooden block out of the Jenga tower", 3)]},
    "conveyor": {"conveyor": [("k1000dai/so101_put_on_conveyor_slow_training", "put the yellow block on the cross mark on the conveyor", 3),
                              ("k1000dai/so101_put_on_conveyor_fast_training", "put the yellow block on the cross mark on the conveyor", 3)]},
    # suture: broad surgical-instrument handling first, then the real needle grasp-and-pass demos
    "suture": {"surgical": [("sajibpra/surgical-instruments-pick-place", "pick up the surgical instrument and place it on the tray", 3)],
               "needle": [("maxboels/lesurgeon-suturing", "grasp the suture needle and pass it through", 1)]},
}


def fetch(repo: str, root: Path) -> None:
    snapshot_download(repo, repo_type="dataset", local_dir=root)
    if json.loads((root / "meta" / "info.json").read_text()).get("codebase_version") != "v3.0":
        convert_dataset(repo_id=repo, root=root, push_to_hub=False)


def name_tasks(root: Path, task: str) -> None:
    """Replace a missing task text ("None"/empty) with a real sentence, everywhere the dataset names tasks."""
    bad = lambda t: str(t).strip() in ("", "None", "nan")  # noqa: E731
    tp = root / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tp)
    if not any(bad(i) for i in tasks.index):
        return
    tasks.index = [task if bad(i) else i for i in tasks.index]
    tasks.to_parquet(tp)
    for ep in (root / "meta" / "episodes").rglob("*.parquet"):
        df = pd.read_parquet(ep)
        df["tasks"] = df["tasks"].apply(lambda ts: [task if bad(t) else t for t in ts])
        df.to_parquet(ep)


task_name, out = sys.argv[1], Path(sys.argv[2])
split = {}
for stage, sources in TASKS[task_name].items():
    roots, counts, holds = [], [], []
    for repo, task, hold in sources:
        root = out / "src" / repo.replace("/", "__")
        fetch(repo, root)
        name_tasks(root, task)
        roots.append(root)
        counts.append(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])
        holds.append(hold)
        print(f"[prep] {stage}: {repo} {counts[-1]} episodes", flush=True)
    aggregate_datasets([f"local/{r.name}" for r in roots], f"local/{stage}", roots=roots, aggr_root=out / stage)
    held, start = [], 0
    for n, h in zip(counts, holds, strict=True):
        held += list(range(start + n - h, start + n))
        start += n
    split[stage] = {"train": [i for i in range(start) if i not in held], "held_out": held}
    print(f"[prep] {stage}: {start} episodes merged, held out {held}", flush=True)
(out / "split.json").write_text(json.dumps(split))
