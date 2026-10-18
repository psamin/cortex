"""Dataset versions: immutable snapshots of accepted episodes with lineage,
deterministic train/eval splits and optional image-only augmentation.

Splits are made over *source* episodes (stratified by condition), and augmented
copies inherit their source's split and are only added to train — so no
augmented view of an eval episode can leak into training, and augmented frames
are never counted as additional demonstrations.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict

from ..context import Ctx
from ..models import (
    EpisodeSource,
    CreateDatasetVersionRequest,
    DatasetEpisodeLink,
    DatasetVersion,
    Episode,
    EpisodeStatus,
    Project,
)
from ..sim.so101 import ENVHUB_URI, FPS
from .jobs import Job
from .recording_service import list_episodes, npz_key


def split_episodes(eps: list[Episode], eval_fraction: float, seed: int) -> dict[str, str]:
    by_cond: dict[str, list[Episode]] = defaultdict(list)
    for e in sorted(eps, key=lambda e: e.id):
        by_cond[e.condition.key].append(e)
    rng = random.Random(seed)
    split = {}
    for _, group in sorted(by_cond.items()):
        rng.shuffle(group)
        n_eval = int(round(len(group) * eval_fraction)) if len(group) >= 3 else 0
        for i, e in enumerate(group):
            split[e.id] = "eval" if i < n_eval else "train"
    return split


def allocate_views(fractions: list[float], n_train: int) -> list[int]:
    """Train episodes per augmented view. A fraction >= 1 means every train episode. Views with
    fractions < 1 share a budget of floor(sum(fractions) * n_train) copies, split by largest
    remainder, so rounding can never push the augmented share past the planned cap (on tiny
    datasets the last views may get none)."""
    out = [n_train if f >= 1 else int(f * n_train) for f in fractions]
    part = [i for i, f in enumerate(fractions) if f < 1]
    spare = int(sum(fractions[i] for i in part) * n_train + 1e-9) - sum(out[i] for i in part)
    for i in sorted(part, key=lambda i: (-(fractions[i] * n_train % 1), i))[:max(0, spare)]:
        out[i] += 1
    return out


def dataset_root(project_id: str, ds_id: str) -> str:
    return f"projects/{project_id}/datasets/{ds_id}/lerobot"


async def create_version(ctx: Ctx, project: Project, req: CreateDatasetVersionRequest, actor: str = "gui") -> Job:
    eps = [e for e in await list_episodes(ctx, project.id) if e.status == EpisodeStatus.accepted]
    if req.episode_ids is not None:
        wanted = set(req.episode_ids)
        eps = [e for e in eps if e.id in wanted]
    if req.sources:
        eps = [e for e in eps if e.source in req.sources]
    else:  # imported real-world demos never mix into simulation datasets unless asked for explicitly
        eps = [e for e in eps if e.source != EpisodeSource.imported]
    if not eps:
        raise ValueError("no accepted episodes to snapshot")
    versions = await ctx.repo.list("dataset_versions", DatasetVersion, project_id=project.id)
    ds = DatasetVersion(project_id=project.id, name=req.name or f"v{len(versions) + 1}",
                        parent_id=req.parent_id or (versions[-1].id if versions else None),
                        links=[], augmentations=req.augmentations, eval_fraction=req.eval_fraction,
                        split_seed=req.split_seed)
    split = split_episodes(eps, req.eval_fraction, req.split_seed)
    items = []
    for e in sorted(eps, key=lambda e: e.index):
        ds.links.append(DatasetEpisodeLink(episode_id=e.id, split=split[e.id],  # type: ignore[arg-type]
                                           lerobot_episode_index=len(items)))
        items.append({"npz": str(await ctx.ensure_local(npz_key(e))),
                      "augmentation": None})
    train = [e for e in sorted(eps, key=lambda e: e.index) if split[e.id] == "train"]
    counts = allocate_views([a.params.get("fraction", 1.0) for a in req.augmentations], len(train))
    for ai, (aug, k) in enumerate(zip(req.augmentations, counts)):
        # each view goes to a deterministic slice of train episodes, rotated per view so different views
        # cover different demos
        for j in range(k):
            e = train[(ai * k + j) % len(train)]
            ds.links.append(DatasetEpisodeLink(episode_id=e.id, split="train", augmentation=aug,
                                               lerobot_episode_index=len(items)))
            items.append({"npz": items[[ln.episode_id for ln in ds.links].index(e.id)]["npz"],
                          "augmentation": aug.model_dump(), "aug_seed": 1000 * ai + e.index})
    ds.lerobot_root = dataset_root(project.id, ds.id)
    ds.storage_prefix = ds.lerobot_root
    key = f"{project.id}:dataset:{json.dumps(req.model_dump(), sort_keys=True)}:{sorted(e.id for e in eps)}"

    async def run(job: Job) -> str:
        root = ctx.local(ds.lerobot_root)  # type: ignore[arg-type]
        shutil.rmtree(root, ignore_errors=True)
        lineage = {
            "dataset_version_id": ds.id, "parent_id": ds.parent_id, "env": ENVHUB_URI,
            "source_episodes": [{"episode_id": e.id, "seed": e.seed, "condition": e.condition.model_dump(),
                                 "split": split[e.id]} for e in eps],
            "links": [ln.model_dump() for ln in ds.links],
            "augmentations": [a.model_dump() for a in req.augmentations],
            "note": "augmented episodes are image-only transforms of their source episode, train split only",
        }

        def on_event(ev):
            if ev.get("type") == "progress":
                ctx.jobs.update(job, progress=0.95 * ev["done"] / ev["total"],
                                message=f"writing LeRobot episode {ev['done']}/{ev['total']}")
            if ev.get("type") == "result" and ev.get("ok"):
                ds.num_frames = ev["frames"]

        await ctx.jobs.run_worker(job, "cortex.workers.dataset_builder", {
            "root": str(root), "repo_id": f"cortex/{project.id}_{ds.id}", "fps": FPS,
            "task": project.spec.goal, "items": items, "lineage": lineage,
        }, ctx.local(f"projects/{project.id}/jobs/{job.id}.json"), on_event)
        for p in root.rglob("*"):
            if p.is_file():
                await ctx.publish_file(str(p.relative_to(ctx.data_dir)))
        await ctx.repo.save("dataset_versions", ds, status="ready")
        for ln in ds.links:
            lid = f"{ds.id}_{ln.lerobot_episode_index}"
            await ctx.store.put("dataset_episode_links", {
                "id": lid, "project_id": project.id, "status": ln.split,
                "data": {"id": lid, "dataset_version_id": ds.id, **ln.model_dump(mode="json")}})
        ctx.changed(project.id, "dataset_versions", ds.id)
        await ctx.audit(actor, "create_dataset_version", project.id, req.model_dump(mode="json"), "ok",
                        f"{ds.name}: {ds.num_source_episodes} source episodes, {len(ds.links)} LeRobot episodes")
        return ds.id

    return ctx.jobs.submit("dataset", project.id, run, idempotency_key=key)


async def list_versions(ctx: Ctx, project_id: str) -> list[DatasetVersion]:
    return await ctx.repo.list("dataset_versions", DatasetVersion, project_id=project_id, status="ready")
