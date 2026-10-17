"""InsForge adapters over its REST API (httpx, server-side admin key).

Endpoints, from InsForge docs/sdks/rest/{database,storage,ai}.mdx:
  records    GET/POST /api/database/records/{table}   (PostgREST filters, array bodies)
  migrations POST /api/database/migrations {version, name, sql}
  storage    POST /api/storage/buckets, PUT/GET /api/storage/buckets/{b}/objects/{key},
             GET  /api/storage/buckets/{b}/download-strategy/objects/{key}?expiresIn=
  AI key     GET  /api/ai/openrouter/api-key  (InsForge's Model Gateway provisions OpenRouter)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .base import Store
from .local import MIGRATIONS

log = logging.getLogger(__name__)


class InsForgeError(RuntimeError):
    pass


class _Client:
    def __init__(self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.base = base_url.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=self.base, timeout=60, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"})

    async def req(self, method: str, path: str, **kw) -> httpx.Response:
        r = await self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise InsForgeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r


class InsForgeStore(Store):
    name = "insforge postgres"

    def __init__(self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.c = _Client(base_url, api_key, transport)

    async def init(self) -> None:
        applied = set()
        try:
            hist = (await self.c.req("GET", "/api/database/migrations")).json()
            rows = hist if isinstance(hist, list) else hist.get("migrations", hist.get("data", []))
            applied = {str(m.get("name")) for m in rows if isinstance(m, dict)}
        except InsForgeError as e:
            log.warning("could not read migration history: %s", e)
        for f in sorted(MIGRATIONS.glob("*.sql")):
            name = migration_name(f)
            if name in applied:
                continue
            body = {"version": migration_version(f), "name": name, "sql": f.read_text()}
            try:
                await self.c.req("POST", "/api/database/migrations", json=body)
                log.info("applied migration %s", name)
            except InsForgeError as e:
                # IF NOT EXISTS makes re-application safe; a duplicate version is fine.
                if "already" not in str(e).lower() and "duplicate" not in str(e).lower():
                    raise

    async def put(self, table: str, row: dict[str, Any]) -> None:
        payload = {k: v for k, v in row.items() if v is not None or k == "status"}
        await self.c.req(
            "POST", f"/api/database/records/{table}", json=[payload],
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"})

    async def get(self, table: str, id: str) -> dict[str, Any] | None:
        r = await self.c.req("GET", f"/api/database/records/{table}", params={"id": f"eq.{id}", "limit": 1})
        rows = r.json()
        return rows[0] if rows else None

    async def list(self, table: str, project_id: str | None = None,
                   status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"order": "created_at.asc", "limit": 1000}
        if project_id is not None:
            params["project_id"] = f"eq.{project_id}"
        if status is not None:
            params["status"] = f"eq.{status}"
        return (await self.c.req("GET", f"/api/database/records/{table}", params=params)).json()


class InsForgeBlobStore:
    name = "insforge storage (private bucket)"

    def __init__(self, base_url: str, api_key: str, bucket: str, transport: httpx.AsyncBaseTransport | None = None):
        self.c = _Client(base_url, api_key, transport)
        self.bucket = bucket

    async def init(self) -> None:
        try:
            await self.c.req("POST", "/api/storage/buckets", json={"bucketName": self.bucket, "isPublic": False})
        except InsForgeError as e:
            if "exist" not in str(e).lower() and "409" not in str(e):
                raise

    def _obj(self, key: str) -> str:
        return f"/api/storage/buckets/{self.bucket}/objects/{quote(key)}"

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        await self.c.req("PUT", self._obj(key),
                         files={"file": (Path(key).name, data, content_type)})
        return key

    async def get_bytes(self, key: str) -> bytes:
        return (await self.c.req("GET", self._obj(key))).content

    async def url(self, key: str, expires_s: int = 3600) -> str:
        """Presigned URL on S3-backed InsForge (Cloud). Self-hosted local-disk
        storage returns a plain URL that needs auth, so the API proxies those."""
        r = await self.c.req("GET", f"/api/storage/buckets/{self.bucket}/download-strategy/objects/{quote(key)}",
                             params={"expiresIn": expires_s})
        info = r.json()
        if info.get("method") == "presigned":
            return info["url"]
        return ""  # caller falls back to the backend proxy route


async def fetch_openrouter_key(base_url: str, api_key: str,
                               transport: httpx.AsyncBaseTransport | None = None) -> str | None:
    try:
        r = await _Client(base_url, api_key, transport).req("GET", "/api/ai/openrouter/api-key")
        return r.json().get("apiKey")
    except (InsForgeError, httpx.HTTPError) as e:
        log.warning("InsForge Model Gateway key unavailable: %s", e)
        return None


def migration_name(f: Path) -> str:
    return "cortex-" + f.stem.replace("_", "-").lower()


def migration_version(f: Path) -> str:
    return f"2026101700{int(f.stem.split('_')[0]):04d}"
