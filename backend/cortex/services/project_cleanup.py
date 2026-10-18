"""Empty projects: hidden from Home, and deleted once they have stayed empty for a while.

A project has history once it holds something a person would come back to: a chat message, demos, an
agent collection, an analysis, a dataset, a training job, a checkpoint, an evaluation, a policy replay, an
autopilot run or an RL sandbox, or a job has run for it. Home lists only those. A project that is still empty
EMPTY_TTL after it was created (e.g. a draft from an abandoned intake) is deleted: its row is marked
`deleted`, which drops it from every project list. Its id still resolves, so an agent that holds it keeps
working, and the project comes back as soon as it is saved again or gets history (see `restore`).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from ..context import Ctx
from ..models import Project, now
from . import chat_store

log = logging.getLogger(__name__)
EMPTY_TTL = timedelta(minutes=10)
HISTORY_TABLES = ("episodes", "agent_collections", "dataset_versions", "training_jobs", "checkpoints",
                  "evaluation_runs", "analyses", "autopilot_runs", "rl_sandboxes", "policy_replays")
DELETED = "deleted"
_KNOWN: set[str] = set()  # projects seen with history; history is never removed, so this only grows


async def has_history(ctx: Ctx, pid: str) -> bool:
    if pid in _KNOWN:
        return True
    found = bool(ctx.jobs.list(pid)) or await _chatted(ctx, pid)
    if not found:
        rows = await asyncio.gather(*(ctx.store.list(t, project_id=pid) for t in HISTORY_TABLES))
        found = any(rows)
    if found:
        _KNOWN.add(pid)
    return found


async def _chatted(ctx: Ctx, pid: str) -> bool:
    for t in await chat_store.threads(ctx, pid):
        if await chat_store.load(ctx, pid, t["id"]):
            return True
    return False


async def listed(ctx: Ctx) -> list[Project]:
    """Every project that is not deleted, newest first."""
    rows = await ctx.store.list("projects")
    return [Project.model_validate(r["data"]) for r in reversed(rows) if r.get("status") != DELETED]


async def with_history(ctx: Ctx, projects: list[Project]) -> list[Project]:
    """The projects that have history. Empty ones older than EMPTY_TTL are deleted on the way."""
    sem = asyncio.Semaphore(4)
    cutoff = now() - EMPTY_TTL

    async def check(p: Project) -> bool:
        async with sem:
            try:
                keep = await has_history(ctx, p.id)
            except Exception as e:  # noqa: BLE001 - can't tell: show it, and never delete on a failed check
                log.warning("history check failed for %s, keeping it: %s", p.id, e)
                return True
            if not keep and p.created_at < cutoff:
                await _delete(ctx, p.id)
            return keep

    keep = await asyncio.gather(*(check(p) for p in projects))
    return [p for p, k in zip(projects, keep, strict=True) if k]


async def _delete(ctx: Ctx, pid: str) -> None:
    row = await ctx.store.get("projects", pid)  # the latest copy, so the delete never reverts another save
    if not row or row.get("status") == DELETED:
        return
    p = Project.model_validate(row["data"])
    await ctx.repo.save("projects", p, status=DELETED)
    await ctx.audit("cortex", "delete_empty_project", pid, {}, "ok",
                    f"'{p.name}' had no history {int(EMPTY_TTL.total_seconds() // 60)} minutes after it was created")
    log.info("deleted empty project %s (%s)", pid, p.name)


async def restore(ctx: Ctx, pid: str) -> None:
    """Bring a deleted project back (it is getting history now)."""
    row = await ctx.store.get("projects", pid)
    if row and row.get("status") == DELETED:
        await ctx.repo.save("projects", Project.model_validate(row["data"]), status="active")
