"""Chat threads per project, kept in the blob store (InsForge storage when configured).

    projects/<pid>/chats/index.json    [{id, title, created_at, updated_at, session_id}], newest first
    projects/<pid>/chats/<thread>.jsonl  one entry per line, as the studio renders them

"New chat" starts a thread; old threads stay listed and can be reopened and continued, each with its
own Claude Code session. Files are written locally first and uploaded in the background, so a
chat keeps working when storage is slow; another machine reads them back from storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..context import Ctx
from ..models import new_id, now

log = logging.getLogger(__name__)
_PENDING: dict[str, asyncio.Task] = {}


def _key(pid: str, name: str) -> str:
    return f"projects/{pid}/chats/{name}"


def _read(ctx: Ctx, key: str) -> str | None:
    p = ctx.local(key)
    return p.read_text() if p.exists() else None


async def _fetch(ctx: Ctx, key: str) -> str | None:
    """Local copy, else the stored one (e.g. written on another machine)."""
    text = _read(ctx, key)
    if text is None:
        try:
            text = (await ctx.ensure_local(key)).read_text()
        except Exception:  # not stored yet
            return None
    return text


def _upload_soon(ctx: Ctx, key: str) -> None:
    """Upload after a short pause, coalescing bursts of appends into one upload."""
    async def go() -> None:
        await asyncio.sleep(1.0)
        try:
            await ctx.publish_file(key)
        except Exception as e:  # the local copy is kept; the next write retries
            log.warning("chat upload failed for %s: %s", key, str(e)[:120])
        finally:
            _PENDING.pop(key, None)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if key not in _PENDING:
        _PENDING[key] = loop.create_task(go())


def _write(ctx: Ctx, key: str, text: str) -> None:
    p = ctx.local(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _upload_soon(ctx, key)


async def threads(ctx: Ctx, pid: str) -> list[dict[str, Any]]:
    raw = await _fetch(ctx, _key(pid, "index.json"))
    if raw is None:
        return await _adopt_legacy(ctx, pid)
    if not raw.strip():  # an interrupted write leaves an empty index: no threads yet
        return []
    return json.loads(raw)


def _save_index(ctx: Ctx, pid: str, items: list[dict[str, Any]]) -> None:
    items.sort(key=lambda t: t["updated_at"], reverse=True)
    _write(ctx, _key(pid, "index.json"), json.dumps(items))


async def _adopt_legacy(ctx: Ctx, pid: str) -> list[dict[str, Any]]:
    """A project from before threads keeps its one chat as the first thread."""
    legacy = ctx.data_dir / "agent_sessions" / pid / "chat.jsonl"
    if not legacy.exists():
        return []
    lines = [ln for ln in legacy.read_text().splitlines() if ln.strip()]
    first = next((json.loads(ln) for ln in lines if json.loads(ln).get("kind") == "user"), {})
    sess = ctx.data_dir / "agent_sessions" / f"{pid}.json"
    t = {"id": new_id("th"), "title": _title(first.get("text", "Earlier chat")),
         "created_at": json.loads(lines[0]).get("at", now().isoformat()) if lines else now().isoformat(),
         "updated_at": json.loads(lines[-1]).get("at", now().isoformat()) if lines else now().isoformat(),
         "session_id": json.loads(sess.read_text()).get("session_id") if sess.exists() else None}
    _write(ctx, _key(pid, f"{t['id']}.jsonl"), "\n".join(lines) + "\n")
    _save_index(ctx, pid, [t])
    return [t]


def _title(text: str) -> str:
    text = " ".join(str(text).split())
    return (text[:57] + "…") if len(text) > 58 else (text or "New chat")


async def new_thread(ctx: Ctx, pid: str) -> dict[str, Any]:
    items = await threads(ctx, pid)
    t = {"id": new_id("th"), "title": "New chat", "created_at": now().isoformat(), "updated_at": now().isoformat(),
         "session_id": None}
    _save_index(ctx, pid, [t, *items])
    return t


async def current(ctx: Ctx, pid: str, tid: str | None = None) -> dict[str, Any]:
    """The requested thread, else the most recent one, else a new one."""
    items = await threads(ctx, pid)
    for t in items:
        if t["id"] == tid or tid is None:
            return t
    return await new_thread(ctx, pid)


async def load(ctx: Ctx, pid: str, tid: str) -> list[dict[str, Any]]:
    raw = await _fetch(ctx, _key(pid, f"{tid}.jsonl"))
    return [json.loads(ln) for ln in (raw or "").splitlines() if ln.strip()]


async def append(ctx: Ctx, pid: str, tid: str, entry: dict[str, Any]) -> None:
    key = _key(pid, f"{tid}.jsonl")
    if _read(ctx, key) is None:
        await _fetch(ctx, key)  # continue a thread written elsewhere
    p = ctx.local(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({**entry, "at": now().isoformat()}, default=str) + "\n")
    _upload_soon(ctx, key)
    items = await threads(ctx, pid)
    for t in items:
        if t["id"] == tid:
            t["updated_at"] = now().isoformat()
            if t["title"] == "New chat" and entry.get("kind") == "user":
                t["title"] = _title(entry.get("text", ""))
    _save_index(ctx, pid, items)


async def set_session(ctx: Ctx, pid: str, tid: str, session_id: str) -> None:
    items = await threads(ctx, pid)
    for t in items:
        if t["id"] == tid:
            t["session_id"] = session_id
    _save_index(ctx, pid, items)
