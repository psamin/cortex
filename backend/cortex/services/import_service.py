"""Import existing LeRobot demonstrations (the user's own SO-101 recordings, or public
Hugging Face datasets) into a project.

Imported episodes get provenance, viewer videos and Cortex's quality checks, and the
whole dataset becomes a DatasetVersion that can be trained on as-is (ACT on Futurama).
They are kept out of simulation coverage and simulation datasets: their cameras and
scene differ from the MuJoCo environment, so simulator evaluation cannot score a
policy trained on them fairly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from ..context import Ctx
from ..models import (
    Condition, DatasetEpisodeLink, DatasetVersion, Episode, EpisodeSource, EpisodeStatus, Project, new_id,
)
from .jobs import Job
from .recording_service import list_episodes, quality_for


def import_key(project_id: str, imp_id: str, name: str = "") -> str:
    return f"projects/{project_id}/imports/{imp_id}/{name}".rstrip("/")


def find_dataset_root(base: Path) -> Path:
    for info in sorted(base.rglob("meta/info.json"), key=lambda p: len(p.parts)):
        return info.parent.parent
    raise ValueError("no LeRobot dataset found (expected a folder containing meta/info.json)")


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():  # refuse path traversal
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise ValueError(f"unsafe path in archive: {m}")
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    else:
        raise ValueError("upload a .zip or .tar.gz of a LeRobot dataset folder")
    return find_dataset_root(dest)


async def start_import(ctx: Ctx, project: Project, source: dict[str, Any], max_episodes: int = 10,
                       actor: str = "gui") -> Job:
    imp_id = new_id("imp")
    base = ctx.local(import_key(project.id, imp_id))
    label = f"hf:{source['hub']}" if "hub" in source else f"upload:{Path(source['path']).name}"

    async def run(job: Job) -> str:
        cfg: dict[str, Any] = {"source": source, "root": str(base / "lerobot"), "max_episodes": max_episodes,
                               "out_dir": str(base / "episodes")}
        meta: dict[str, Any] = {}

        def on_event(ev: dict[str, Any]) -> None:
            if ev.get("type") == "meta":
                meta.update(ev)
                ctx.jobs.update(job, message=f"importing {ev['importing']} of {ev['total_episodes']} episodes "
                                             f"from {label}")
            elif ev.get("type") == "episode":
                ctx.jobs.update(job, progress=(ev["index"] + 1) / max(ev["total"], 1),
                                message=f"imported episode {ev['index'] + 1}/{ev['total']}")

        token = hf_token(ctx)  # passed in the environment only, never written to the config file
        env = {"HF_TOKEN": token} if token else None
        await ctx.jobs.run_worker(job, "cortex.workers.import_worker", cfg, base / "config.json", on_event, env=env)
        res = json.loads((base / "episodes" / "result.json").read_text())
        start = len(await list_episodes(ctx, project.id))
        ds = DatasetVersion(project_id=project.id, name=f"imported {label}", links=[], lerobot_root=res["root"],
                            imported_from=label, eval_fraction=0.0)
        for k, r in enumerate(res["episodes"]):
            out = Path(r["out_dir"])
            rel = lambda n, out=out: str((out / n).relative_to(ctx.data_dir))  # noqa: E731
            ep = Episode(project_id=project.id, index=start + k, source=EpisodeSource.imported,
                         condition=Condition(), seed=-1, num_frames=r["num_frames"], fps=int(res["fps"]),
                         success=None, status=EpisodeStatus.accepted,
                         artifacts={"episode": rel("episode.npz"), "front": rel("front.mp4"), "thumb": rel("thumb.jpg")},
                         provenance={"planner": "human demonstration (imported)", "source": label,
                                     "source_episode_index": r["source_episode_index"], "robot_type": res["robot_type"],
                                     "cameras": res["cameras"], "task": r.get("task") or None,
                                     "note": "success not judged by the simulator; condition unknown"})
            ep.quality = quality_for(ctx, ep, None)
            for kind in ("episode", "front", "thumb"):
                await ctx.publish_file(ep.artifacts[kind])
            await ctx.repo.save("episodes", ep)
            ds.links.append(DatasetEpisodeLink(episode_id=ep.id, split="train",
                                               lerobot_episode_index=r["source_episode_index"]))
            ds.num_frames += r["num_frames"]
        await ctx.repo.save("dataset_versions", ds, status="ready")
        ctx.changed(project.id, "dataset_versions", ds.id)
        await ctx.audit(actor, "import_demonstrations", project.id, {"source": label, "max_episodes": max_episodes},
                        "ok", f"{len(res['episodes'])} episodes imported as {ds.id}")
        return ds.id

    return ctx.jobs.submit("import", project.id, run, idempotency_key=dedup_key(project.id, source, max_episodes))


def hf_token(ctx: Ctx) -> str | None:
    """The user's own Hugging Face login (`hf auth login`, the account LeRobot uses), else HF_TOKEN from .env."""
    from huggingface_hub import constants

    p = Path(constants.HF_TOKEN_PATH)
    login = p.read_text().strip() if p.is_file() else ""
    return login or ctx.settings.hf_token


def dedup_key(project_id: str, source: dict[str, Any], max_episodes: int) -> str:
    """The same Hub repo or the same archive bytes (not merely the same folder name) imports once."""
    what = f"hf:{source['hub']}@{source.get('revision') or 'main'}" if "hub" in source else \
        f"upload:{source.get('sha256') or source['path']}"
    return f"import:{project_id}:{what}:{max_episodes}"


def save_upload(ctx: Ctx, project_id: str, filename: str, data: bytes) -> dict[str, str]:
    """Store and unpack an uploaded archive; returns the import source (dataset path + archive hash)."""
    imp = new_id("upl")
    base = ctx.local(import_key(project_id, imp))
    base.mkdir(parents=True, exist_ok=True)
    archive = base / Path(filename).name
    archive.write_bytes(data)
    try:
        return {"path": str(extract_archive(archive, base / "raw")), "sha256": hashlib.sha256(data).hexdigest()}
    except Exception:
        shutil.rmtree(base, ignore_errors=True)
        raise
