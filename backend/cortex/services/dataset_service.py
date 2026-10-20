"""Dataset versions: immutable snapshots of accepted episodes with lineage,
deterministic train/eval splits and optional image-only augmentation.

Splits are made over *source* episodes (stratified by condition), and augmented
copies inherit their source's split and are only added to train — so no
augmented view of an eval episode can leak into training, and augmented frames
are never counted as additional demonstrations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict

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


def _unit(episode_id: str, seed: int) -> float:
    """A stable number in [0, 1) per (episode, seed)."""
    h = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_episodes(eps: list[Episode], eval_fraction: float, seed: int) -> dict[str, str]:
    """Each episode's split depends only on its id and the seed, never on which other episodes are in the version
    (#368): v1's held-out demos stay held out in v2, so checkpoints from both can be compared on them. A condition
    with fewer than 3 episodes keeps them all for training (an eval-only condition would never be learned)."""
    by_cond: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        by_cond[e.condition.key].append(e)
    split = {}
    for group in by_cond.values():
        for e in group:
            split[e.id] = "eval" if len(group) >= 3 and _unit(e.id, seed) < eval_fraction else "train"
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


def snapshot_episodes(eps: list[Episode], req: CreateDatasetVersionRequest) -> list[Episode]:
    """The accepted episodes a dataset version takes. Raises ValueError naming the filter that left none."""
    accepted = [e for e in eps if e.status == EpisodeStatus.accepted]
    eps = accepted
    if req.episode_ids is not None:
        wanted = set(req.episode_ids)
        eps = [e for e in eps if e.id in wanted]
    pool = eps
    if req.sources:
        eps = [e for e in eps if e.source in req.sources]
    else:  # imported real-world demos never mix into simulation datasets unless asked for explicitly
        eps = [e for e in eps if e.source != EpisodeSource.imported]
    by_source = eps
    # cycle protocol: demo + reset segments by default; recover segments only when asked for
    eps = [e for e in eps if e.segment in req.segments]
    if eps:
        return eps
    if not accepted:
        raise ValueError("no accepted episodes to snapshot: record demos (or accept recorded ones) first")
    if not pool:
        raise ValueError(f"none of the {len(req.episode_ids or [])} requested episodes is accepted")
    if not by_source:
        have = ", ".join(f"{n} {s}" for s, n in sorted(Counter(str(e.source) for e in pool).items()))
        want = ", ".join(str(s) for s in req.sources) if req.sources else "simulation"
        raise ValueError(f"no accepted {want} episodes to snapshot: the {len(pool)} accepted are {have}. "
                         "Pass those sources to use them, or collect demos of the source you asked for.")
    raise ValueError(f"no accepted episodes in the segments {', '.join(req.segments)} "
                     f"({len(by_source)} accepted are other segments)")


async def create_version(ctx: Ctx, project: Project, req: CreateDatasetVersionRequest, actor: str = "gui") -> Job:
    eps = snapshot_episodes(await list_episodes(ctx, project.id), req)
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
                                 "split": split[e.id], "segment": e.segment,
                                 "cycle_id": e.provenance.get("cycle_id"), "planner": e.provenance.get("planner"),
                                 "rl_sandbox_id": e.provenance.get("sandbox_id")} for e in eps],
            "segments": {"included": list(req.segments),
                         "counts": {k: sum(e.segment == k for e in eps) for k in ("demo", "reset", "recover")}},
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
                await ctx.publish_file(p.relative_to(ctx.data_dir).as_posix())  # '/' keys on Windows too (#367)
        for ln in ds.links:
            lid = f"{ds.id}_{ln.lerobot_episode_index}"
            await ctx.store.put("dataset_episode_links", {
                "id": lid, "project_id": project.id, "status": ln.split,
                "data": {"id": lid, "dataset_version_id": ds.id, **ln.model_dump(mode="json")}})
        # last: a 'ready' row means its files and links are all written. A restart part-way leaves no row, so the
        # build is simply run again (#365).
        await ctx.repo.save("dataset_versions", ds, status="ready")
        ctx.changed(project.id, "dataset_versions", ds.id)
        await ctx.audit(actor, "create_dataset_version", project.id, req.model_dump(mode="json"), "ok",
                        f"{ds.name}: {ds.num_source_episodes} source episodes, {len(ds.links)} LeRobot episodes")
        return ds.id

    return ctx.jobs.submit("dataset", project.id, run, idempotency_key=key)


async def list_versions(ctx: Ctx, project_id: str) -> list[DatasetVersion]:
    return await ctx.repo.list("dataset_versions", DatasetVersion, project_id=project_id, status="ready")
