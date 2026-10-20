"""Agent observability (#105): which person an MCP agent acts for, its sessions, and every tool call.

The MCP server forwards three headers on each backend call:
  X-Cortex-Agent-Key      an agent key (ck_...) created in Settings -> Agents; it names the owner
  X-Cortex-Agent-Session  the per-connection MCP session id
  X-Cortex-Client         the client's clientInfo from MCP initialize, "name/version"
`Authorization: Bearer CORTEX_MCP_KEY` stays the tunnel guard (api/app.py) and is unrelated.

Without a key, calls belong to the implicit local owner ("local mode, single user"). Each call writes
a `tool_audit_logs` row at start (status running) and updates it at the end (ok | error |
approval_required) with duration, a truncated result and any returned image stored as a blob.
Events go to the owner's channel on the bus (`user:<owner>`), streamed by GET /api/agents/events.
Agent keys only attribute calls: they can never approve training or hardware (those stay human clicks).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..context import CURRENT_CALL, Ctx, redact_secrets
from ..errors import NotFound
from ..models import new_id, now

log = logging.getLogger(__name__)

KEY_HEADER, SESSION_HEADER, CLIENT_HEADER = "x-cortex-agent-key", "x-cortex-agent-session", "x-cortex-client"
LOCAL_OWNER = "local"
IDLE_AFTER_S = 300
SUMMARY_MAX, PREVIEW_MAX = 500, 4000
_SAFE = re.compile(r"[^A-Za-z0-9_.:-]")


class AgentKeyRefused(PermissionError):
    code = "agent_key_invalid"
    approval_required = False


@dataclass
class Caller:
    owner: str
    actor: str = "mcp"
    session_id: str | None = None  # ags_... (scoped to the owner)
    mcp_session_id: str | None = None
    client: str | None = None
    client_version: str | None = None
    key_id: str | None = None


def channel(owner: str) -> str:
    return f"user:{owner}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def has_agent_headers(headers: Mapping[str, str]) -> bool:
    return any(headers.get(h) for h in (KEY_HEADER, SESSION_HEADER))


# ------------------------------------------------------------------ keys
async def create_key(ctx: Ctx, owner: str, label: str) -> dict[str, Any]:
    """The plaintext key is returned here once and never stored."""
    key = "ck_" + secrets.token_urlsafe(24)
    kid = new_id("agk")
    data = {"id": kid, "label": (label or "agent").strip()[:80], "prefix": key[:7], "created_at": now().isoformat(),
            "revoked_at": None}
    await ctx.store.put("agent_keys", {"id": kid, "project_id": None, "status": "active", "owner_id": owner,
                                       "key_hash": hash_key(key), "data": data, "created_at": data["created_at"]})
    return {**data, "status": "active", "key": key}


async def list_keys(ctx: Ctx, owner: str) -> list[dict[str, Any]]:
    rows = await ctx.store.list("agent_keys", where={"owner_id": owner})
    return [{**r["data"], "status": r["status"]} for r in reversed(rows)]


async def revoke_key(ctx: Ctx, owner: str, key_id: str) -> dict[str, Any]:
    row = await ctx.store.get("agent_keys", key_id)
    if not row or row["owner_id"] != owner:
        raise NotFound(f"agent key {key_id}")
    row["data"]["revoked_at"] = row["data"].get("revoked_at") or now().isoformat()
    await ctx.store.put("agent_keys", {**row, "status": "revoked"})
    return {**row["data"], "status": "revoked"}


# ------------------------------------------------------------------ callers
async def resolve(ctx: Ctx, headers: Mapping[str, str], actor: str = "mcp") -> Caller:
    """Who a tool call is for. An unknown or revoked key is refused, never downgraded to the local owner."""
    owner, key_id = LOCAL_OWNER, None
    key = (headers.get(KEY_HEADER) or "").strip()
    if key:
        rows = await ctx.store.list("agent_keys", where={"key_hash": hash_key(key)})
        if not rows or rows[0]["status"] != "active":
            raise AgentKeyRefused("This agent key is unknown or revoked. Create a new one in the Cortex studio "
                                  "(Settings -> Agents).")
        owner, key_id = rows[0]["owner_id"], rows[0]["id"]
    raw_client = (headers.get(CLIENT_HEADER) or "").strip()[:120]
    name, _, version = raw_client.partition("/")
    mcp_sid = _SAFE.sub("", (headers.get(SESSION_HEADER) or "").strip())[:128] or None
    sid = None
    if mcp_sid:  # scoped by owner, so one key cannot write into another person's session
        sid = "ags_" + hashlib.sha256(f"{owner}\n{mcp_sid}".encode()).hexdigest()[:20]
    return Caller(owner=owner, actor=actor, session_id=sid, mcp_session_id=mcp_sid, client=name or None,
                  client_version=version or None, key_id=key_id)


_locks: dict[str, asyncio.Lock] = {}
_watchers: dict[int, asyncio.Task] = {}


async def touch_session(ctx: Ctx, caller: Caller, tool: str, project_id: str | None) -> dict[str, Any] | None:
    if not caller.session_id:
        return None
    async with _locks.setdefault(caller.session_id, asyncio.Lock()):
        row = await ctx.store.get("agent_sessions", caller.session_id)
        at = now().isoformat()
        data = row["data"] if row else {
            "id": caller.session_id, "mcp_session_id": caller.mcp_session_id, "key_id": caller.key_id,
            "client": caller.client, "client_version": caller.client_version, "first_seen": at, "call_count": 0,
            "projects": [], "created_at": at}
        data.update(last_seen=at, last_tool=tool, call_count=int(data.get("call_count", 0)) + 1)
        if caller.client:
            data.update(client=caller.client, client_version=caller.client_version)
        if project_id and project_id not in data["projects"]:
            data["projects"] = [*data["projects"], project_id]
        await ctx.store.put("agent_sessions", {"id": caller.session_id, "project_id": None, "status": "live",
                                               "owner_id": caller.owner, "data": data,
                                               "created_at": row["created_at"] if row else at})
    if row is None or row["status"] != "live":
        ctx.bus.publish(channel(caller.owner), "session_start", _session_view(data, "live"))
    _ensure_watcher(ctx)
    return data


async def name_session(ctx: Ctx, session_id: str, name: str, purpose: str | None) -> dict[str, Any] | None:
    """The agent's chosen name (cortex.register_agent): shown on its session, its calls and the Agent log."""
    async with _locks.setdefault(session_id, asyncio.Lock()):
        row = await ctx.store.get("agent_sessions", session_id)
        if not row:
            return None
        row["data"].update(agent_name=name, purpose=purpose)
        await ctx.store.put("agent_sessions", row)
    ctx.bus.publish(channel(row["owner_id"]), "session_start", _session_view(row["data"], row["status"]))
    return row["data"]


async def _add_project(ctx: Ctx, caller: Caller, project_id: str) -> None:
    if not caller.session_id:
        return
    async with _locks.setdefault(caller.session_id, asyncio.Lock()):
        row = await ctx.store.get("agent_sessions", caller.session_id)
        if row and project_id not in row["data"]["projects"]:
            row["data"]["projects"] = [*row["data"]["projects"], project_id]
            await ctx.store.put("agent_sessions", row)


def _ensure_watcher(ctx: Ctx) -> None:
    t = _watchers.get(id(ctx))
    if t is None or t.done():
        _watchers[id(ctx)] = asyncio.get_running_loop().create_task(_idle_loop(ctx))


async def _idle_loop(ctx: Ctx) -> None:
    while True:
        await asyncio.sleep(30)
        try:
            await mark_idle(ctx)
        except Exception:  # never let the watcher die on a transient store error
            log.exception("agent session idle check failed")


async def mark_idle(ctx: Ctx, idle_after_s: float = IDLE_AFTER_S) -> int:
    n = 0
    for row in await ctx.store.list("agent_sessions", status="live"):
        if _age_s(row["data"].get("last_seen")) >= idle_after_s:
            await ctx.store.put("agent_sessions", {**row, "status": "idle"})
            ctx.bus.publish(channel(row["owner_id"]), "session_idle", _session_view(row["data"], "idle"))
            n += 1
    return n


def _age_s(iso: str | None) -> float:
    try:
        return (now() - datetime.fromisoformat(iso)).total_seconds() if iso else 1e9
    except ValueError:
        return 1e9


def _session_view(data: dict[str, Any], status: str) -> dict[str, Any]:
    live = status == "live" and _age_s(data.get("last_seen")) < IDLE_AFTER_S
    return {**data, "status": "live" if live else "idle", "live": live}


async def list_sessions(ctx: Ctx, owner: str) -> list[dict[str, Any]]:
    rows = await ctx.store.list("agent_sessions", where={"owner_id": owner})
    out = [_session_view(r["data"], r["status"]) for r in rows]
    return sorted(out, key=lambda s: s.get("last_seen") or "", reverse=True)


async def session_calls(ctx: Ctx, owner: str, session_id: str, limit: int = 500) -> list[dict[str, Any]]:
    row = await ctx.store.get("agent_sessions", session_id)
    if not row or row["owner_id"] != owner:
        raise NotFound(f"agent session {session_id}")
    rows = (await ctx.store.list("tool_audit_logs", where={"session_id": session_id}))[-limit:]
    return [await call_view(ctx, r) for r in rows]


async def project_calls(ctx: Ctx, project_id: str, limit: int = 300) -> list[dict[str, Any]]:
    """Every tool call on one project, from any agent (MCP sessions and the studio agent), oldest first.
    The project is the scope here, like /api/projects/{pid}/audit, so no owner filter."""
    rows = await ctx.store.list("tool_audit_logs", project_id=project_id)
    rows = sorted(rows, key=lambda r: r["data"].get("started_at") or r["data"].get("created_at") or "")[-limit:]
    return [await call_view(ctx, r) for r in rows if r["data"].get("tool")]


async def call_view(ctx: Ctx, row: dict[str, Any]) -> dict[str, Any]:
    d = {**row["data"], "status": row["status"]}
    d["image_urls"] = [await ctx.media_url(k) for k in d.get("images") or []]
    return d


# ------------------------------------------------------------------ calls
async def observe(ctx: Ctx, caller: Caller | None, tool: str, raw_args: dict[str, Any], actor: str,
                  run: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    """Run one registry tool call with a start row, an end row and live events."""
    pid = raw_args.get("project_id") if isinstance(raw_args.get("project_id"), str) else None
    call: dict[str, Any] = {"id": new_id("aud"), "tool": tool, "open": True,
                            "session_id": caller.session_id if caller else None,
                            "key_id": caller.key_id if caller else None}
    base = {"id": call["id"], "actor": actor, "tool": tool, "args": redact_secrets(raw_args), "project_id": pid,
            "session_id": call["session_id"], "client": caller.client if caller else None,
            "owner": caller.owner if caller else None, "started_at": now().isoformat(),
            "created_at": now().isoformat()}
    if caller:
        sess = await _best_effort("touch the agent session", touch_session(ctx, caller, tool, pid))
        if sess and sess.get("agent_name"):
            base["agent_name"] = sess["agent_name"]
    await _best_effort("write the tool_start row", _write(ctx, caller, base, "running", "tool_start"))
    token = CURRENT_CALL.set(call)
    t0 = time.perf_counter()
    try:
        result = await run()
    except Exception as e:
        approval = isinstance(e, PermissionError) and getattr(e, "approval_required", True)
        status = "approval_required" if approval else "error"
        # the tool's own error (e.g. APPROVAL REQUIRED) is what the caller must see, not a store failure
        await _best_effort("write the tool_end row", _finish(ctx, caller, base, call, t0, status, None,
                                                             f"{type(e).__name__}: {e}"))
        raise
    finally:
        call["open"] = False
        CURRENT_CALL.reset(token)
    await _best_effort("write the tool_end row", _finish(ctx, caller, base, call, t0, "ok", result, None))
    return result


async def _best_effort(what: str, aw: Awaitable[Any]) -> Any:
    """Observability never decides a tool call (#355): a store that is down (InsForge unreachable, say) must not
    stop the tool from running, fail it after its side effects happened, or mask its own error."""
    try:
        return await aw
    except Exception:
        log.warning("agent_obs: could not %s; the tool call goes on", what, exc_info=True)
        return None


async def _finish(ctx: Ctx, caller: Caller | None, base: dict[str, Any], call: dict[str, Any], t0: float,
                  status: str, result: dict[str, Any] | None, error: str | None) -> None:
    d = {**base, "ended_at": now().isoformat(), "duration_ms": round((time.perf_counter() - t0) * 1000),
         "domain": call.get("domain"), "images": []}
    if error:
        d["summary"] = d["error"] = error[:SUMMARY_MAX]
    if result is not None:
        d["summary"] = str(result.get("summary", ""))[:SUMMARY_MAX]
        if not d["project_id"] and isinstance(result.get("project_id"), str):  # e.g. start_intake made one
            d["project_id"] = result["project_id"]
            if caller:
                await _add_project(ctx, caller, d["project_id"])
        img = result.get("image_b64")
        if isinstance(img, str) and img:
            try:
                key = f"agents/{_SAFE.sub('_', base['owner'] or 'studio')}/{base['session_id'] or 'none'}/{base['id']}.jpg"
                await ctx.blobs.put_bytes(key, base64.b64decode(img), "image/jpeg")
                d["images"] = [key]
            except (ValueError, OSError) as e:  # a bad image must not fail the call it rode on
                log.warning("could not store image from %s: %s", base["tool"], e)
        preview = redact_secrets({k: v for k, v in result.items() if k != "image_b64"})
        d["result_preview"] = json.dumps(preview, default=str)[:PREVIEW_MAX]
        d["needs_approval"] = bool(result.get("requires_approval") or result.get("requires_approval_for_training"))
    await _write(ctx, caller, d, status, "tool_end")
    if status == "approval_required" or d.get("needs_approval"):
        await _emit(ctx, caller, "approval_required", d)


async def _write(ctx: Ctx, caller: Caller | None, d: dict[str, Any], status: str, event: str) -> None:
    await ctx.store.put("tool_audit_logs", {"id": d["id"], "project_id": d["project_id"], "status": status,
                                            "session_id": d["session_id"], "data": d, "created_at": d["created_at"]})
    if d["project_id"]:
        ctx.changed(d["project_id"], "tool_audit_logs", d["id"])
    await _emit(ctx, caller, event, {**d, "status": status})


async def _emit(ctx: Ctx, caller: Caller | None, event: str, d: dict[str, Any]) -> None:
    if caller and caller.session_id:
        row = {"data": d, "status": d.get("status", "ok")}
        ctx.bus.publish(channel(caller.owner), event, await call_view(ctx, row))


def session_link(ctx: Ctx) -> tuple[str, str | None]:
    call = CURRENT_CALL.get()
    sid = call.get("session_id") if call else None
    base = ctx.settings.cortex_studio_url.rstrip("/") + "/studio/agents"
    return (f"{base}?session={sid}" if sid else base), sid
