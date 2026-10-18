"""REST + SSE for agent observability (#105): agent keys, sessions, call timelines, live events.

Everything is scoped to the current user: in local mode the machine's single implicit owner,
with InsForge configured the user signed in through InsForge Auth. Agents themselves (requests
carrying agent headers or X-Cortex-Actor mcp/agent) can never manage keys or read these views.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..context import Ctx
from ..services import agent_obs as obs
from ..services import insforge_auth

router = APIRouter(prefix="/api/agents")


def _ctx() -> Ctx:
    from .app import ctx

    return ctx()


class AgentsForbidden(PermissionError):
    code = "agents_forbidden"
    approval_required = False


async def current_owner(request: Request) -> dict[str, Any]:
    if obs.has_agent_headers(request.headers) or request.headers.get("x-cortex-actor") in ("mcp", "agent"):
        raise AgentsForbidden("Agent keys and agent sessions are managed by a person in the studio, not by agents.")
    s = _ctx().settings
    if not s.insforge_enabled:
        return {"id": obs.LOCAL_OWNER, "email": None, "name": "local user"}
    token = request.headers.get("x-cortex-user-token") or request.query_params.get("token")
    return await insforge_auth.current_user(s, token)


@router.get("/me")
async def me(request: Request):
    s = _ctx().settings
    if not s.insforge_enabled:
        return {"mode": "local", "signed_in": True, "user": await current_owner(request),
                "note": "local mode, single user: every agent on this machine is yours"}
    try:
        return {"mode": "insforge", "signed_in": True, "user": await current_owner(request)}
    except insforge_auth.SignInRequired as e:
        return {"mode": "insforge", "signed_in": False, "user": None, "note": str(e)}


class LoginReq(BaseModel):
    email: str
    password: str


@router.post("/login")
async def login(req: LoginReq):
    s = _ctx().settings
    if not s.insforge_enabled:
        return {"mode": "local", "access_token": None, "user": {"id": obs.LOCAL_OWNER, "name": "local user"}}
    return {"mode": "insforge", **await insforge_auth.sign_in(s, req.email, req.password)}


class KeyReq(BaseModel):
    label: str = "agent"


@router.post("/keys")
async def create_key(req: KeyReq, request: Request):
    """The key is in this response only; Cortex keeps its SHA-256."""
    return await obs.create_key(_ctx(), (await current_owner(request))["id"], req.label)


@router.get("/keys")
async def list_keys(request: Request):
    return await obs.list_keys(_ctx(), (await current_owner(request))["id"])


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, request: Request):
    return await obs.revoke_key(_ctx(), (await current_owner(request))["id"], key_id)


@router.get("/sessions")
async def sessions(request: Request):
    return await obs.list_sessions(_ctx(), (await current_owner(request))["id"])


@router.get("/sessions/{sid}/calls")
async def session_calls(sid: str, request: Request):
    return await obs.session_calls(_ctx(), (await current_owner(request))["id"], sid)


@router.get("/projects/{pid}/calls")
async def project_calls(pid: str):
    """The project's agent log: every agent's tool calls on it, with image URLs (the chat's log mode)."""
    return await obs.project_calls(_ctx(), pid)


@router.get("/events")
async def events(request: Request):
    """session_start, tool_start, tool_end, approval_required, session_idle - for this user's agents only."""
    owner = (await current_owner(request))["id"]
    bus = _ctx().bus

    async def gen():
        yield "event: hello\ndata: {}\n\n"
        q = bus.subscribe(obs.channel(owner))
        try:
            while True:
                try:
                    ev, data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {ev}\ndata: {json.dumps(data, default=str)}\n\n"
                except TimeoutError:
                    if await request.is_disconnected():
                        return
                    yield ": ping\n\n"
        finally:
            bus.unsubscribe(obs.channel(owner), q)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
