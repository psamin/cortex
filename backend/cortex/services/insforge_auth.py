"""Studio sign-in through InsForge Auth (only when InsForge is configured; local mode is single-user).

REST endpoints, from InsForge docs/sdks/rest/auth (read 2026-10-18):
  POST /api/auth/sessions          {email, password} -> {user, accessToken, ...}
  GET  /api/auth/sessions/current  Authorization: Bearer <accessToken> -> {user: {id, email, role}}
OAuth (GET /api/auth/oauth/{provider} + PKCE exchange) is not wired yet: email and password only.
The studio keeps the access token and sends it as X-Cortex-User-Token (or ?token= on SSE).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import Settings


class SignInRequired(PermissionError):
    code = "sign_in_required"
    approval_required = False


_cache: dict[str, tuple[dict[str, Any], float]] = {}
CACHE_S = 60


def _http(s: Settings, transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=(s.insforge_base_url or "").rstrip("/"), timeout=20, transport=transport)


async def sign_in(s: Settings, email: str, password: str,
                  transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    async with _http(s, transport) as c:
        r = await c.post("/api/auth/sessions", json={"email": email, "password": password})
    if r.status_code >= 400:
        try:
            msg = r.json().get("message") or r.text
        except ValueError:
            msg = r.text
        raise SignInRequired(f"InsForge sign-in failed: {str(msg)[:200]}")
    body = r.json()
    user, token = body.get("user") or {}, body.get("accessToken")
    if not token or not user.get("id"):
        raise SignInRequired("InsForge sign-in returned no access token")
    _cache[token] = (_user(user), time.time() + CACHE_S)
    return {"access_token": token, "user": _user(user)}


async def current_user(s: Settings, token: str | None,
                       transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    if not token:
        raise SignInRequired("Sign in to see your agents.")
    hit = _cache.get(token)
    if hit and hit[1] > time.time():
        return hit[0]
    async with _http(s, transport) as c:
        r = await c.get("/api/auth/sessions/current", headers={"Authorization": f"Bearer {token}"})
    if r.status_code >= 400:
        _cache.pop(token, None)
        raise SignInRequired("Your sign-in expired. Sign in again.")
    user = _user((r.json() or {}).get("user") or {})
    if not user["id"]:
        raise SignInRequired("InsForge returned no user for this token.")
    _cache[token] = (user, time.time() + CACHE_S)
    return user


def _user(u: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(u.get("id") or ""), "email": u.get("email"), "name": u.get("name") or u.get("email")}
