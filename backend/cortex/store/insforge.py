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
from .s3 import S3Gateway

log = logging.getLogger(__name__)
PAGE = 1000  # rows per records request


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
            except InsForgeError:
                # Only tolerate a race where the same migration name got applied meanwhile.
                # InsForge also rejects versions that are not newer than the latest applied one
                # (DATABASE_MIGRATION_ALREADY_EXISTS) - that must fail loudly, not be skipped.
                hist = (await self.c.req("GET", "/api/database/migrations")).json()
                rows = hist if isinstance(hist, list) else hist.get("migrations", [])
                if name not in {str(m.get("name")) for m in rows if isinstance(m, dict)}:
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
        params: dict[str, Any] = {"order": "created_at.asc,id.asc", "limit": PAGE}
        if project_id is not None:
            params["project_id"] = f"eq.{project_id}"
        if status is not None:
            params["status"] = f"eq.{status}"
        rows: list[dict[str, Any]] = []
        while True:  # page until a short page: a single capped request would drop the newest rows
            page = (await self.c.req("GET", f"/api/database/records/{table}",
                                     params={**params, "offset": len(rows)})).json()
            rows += page
            if len(page) < PAGE:
                return rows


class InsForgeBlobStore:
    name = "insforge storage (private bucket)"

    def __init__(self, base_url: str, api_key: str, bucket: str, transport: httpx.AsyncBaseTransport | None = None,
                 rest_max_bytes: int = 50 * 1024 * 1024, s3_keys: tuple[str, str] | None = None,
                 s3_endpoint: str | None = None, s3_region: str | None = None,
                 s3_transport: httpx.AsyncBaseTransport | None = None):
        self.c = _Client(base_url, api_key, transport)
        self.bucket = bucket
        self.rest_max_bytes = rest_max_bytes
        self._s3_keys, self._s3_endpoint, self._s3_region, self._s3_transport = s3_keys, s3_endpoint, s3_region, \
            s3_transport
        self.s3: S3Gateway | None = None
        self.s3_problem = "no S3 access keys (INSFORGE_S3_ACCESS_KEY_ID / INSFORGE_S3_SECRET_ACCESS_KEY)"

    async def init(self) -> None:
        try:
            await self.c.req("POST", "/api/storage/buckets", json={"bucketName": self.bucket, "isPublic": False})
        except InsForgeError as e:
            if "exist" not in str(e).lower() and "409" not in str(e):
                raise
        if self._s3_keys:
            await self._init_s3()

    async def _init_s3(self) -> None:
        """The S3 gateway's endpoint and region: .env overrides, else GET /api/storage/s3/config."""
        endpoint, region = self._s3_endpoint, self._s3_region
        if not (endpoint and region):
            try:
                cfg = (await self.c.req("GET", "/api/storage/s3/config")).json()
                cfg = cfg.get("data", cfg) if isinstance(cfg, dict) else {}
                endpoint = endpoint or cfg.get("endpoint") or cfg.get("endpointUrl")
                region = region or cfg.get("region")
            except (InsForgeError, ValueError) as e:
                log.warning("InsForge S3 config unavailable: %s", e)
        endpoint = endpoint or f"{self.c.base}/storage/v1/s3"
        if not region:
            self.s3_problem = "S3 region unknown: set INSFORGE_S3_REGION (Dashboard -> Storage -> Settings)"
            log.warning("InsForge S3 gateway disabled: %s", self.s3_problem)
            return
        self.s3 = S3Gateway(endpoint, region, *self._s3_keys, transport=self._s3_transport)  # type: ignore[misc]

    async def put_file(self, key: str, path: Path, content_type: str) -> str:
        """REST up to the REST cap; larger files (ACT checkpoints) through the S3 gateway, streamed in parts."""
        size = path.stat().st_size
        if size <= self.rest_max_bytes:
            return await self.put_bytes(key, path.read_bytes(), content_type)
        if self.s3 is None:
            raise InsForgeError(f"{key} is {size / 2**20:.0f} MB, above InsForge's REST upload cap of "
                                f"{self.rest_max_bytes / 2**20:.0f} MB, and the S3 gateway is off: {self.s3_problem}")
        return await self.s3.put_file(self.bucket, key, path, content_type)

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


async def gateway_key(base_url: str, api_key: str,
                      transport: httpx.AsyncBaseTransport | None = None) -> tuple[str | None, str | None]:
    """(OpenRouter key, None) or (None, InsForge's reason). Free InsForge plans answer
    403 AI_UPSTREAM_UNAVAILABLE ("AI is only available on paid plans")."""
    try:
        r = await _Client(base_url, api_key, transport).req("GET", "/api/ai/openrouter/api-key")
        return r.json().get("apiKey"), None
    except InsForgeError as e:
        log.warning("InsForge Model Gateway key unavailable: %s", e)
        body = str(e).split(": ", 1)[-1]
        try:
            import json

            return None, json.loads(body).get("message") or body
        except ValueError:
            return None, body[:200]
    except httpx.HTTPError as e:
        log.warning("InsForge Model Gateway unreachable: %s", e)
        return None, f"InsForge unreachable: {type(e).__name__}"


async def fetch_openrouter_key(base_url: str, api_key: str,
                               transport: httpx.AsyncBaseTransport | None = None) -> str | None:
    return (await gateway_key(base_url, api_key, transport))[0]


def migration_name(f: Path) -> str:
    return "cortex-" + f.stem.replace("_", "-").lower()


def migration_version(f: Path) -> str:
    return f"2026101700{int(f.stem.split('_')[0]):04d}"
