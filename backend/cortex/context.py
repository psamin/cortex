"""Application context: providers selected from settings, shared by REST, MCP
(via REST) and the agent. Services receive a Ctx and never pick providers."""

from __future__ import annotations

import logging
import mimetypes
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .models import new_id, now
from .services.jobs import EventBus, JobManager
from .store.base import BlobStore, Repo, Store
from .store.local import LocalBlobStore, SqliteStore

log = logging.getLogger(__name__)

# The agent tool call in flight on this task (set by agent_obs around call_tool): {id, tool, session_id, open, ...}.
CURRENT_CALL: ContextVar[dict[str, Any] | None] = ContextVar("cortex_current_call", default=None)


def _secret_field(key: str) -> bool:
    k = key.lower()
    return k == "approval_token" or "nonce" in k


def redact_secrets(value: Any) -> Any:
    """Replace operator secrets before a value is written to the audit log or the bus.

    `approval_token` and any field whose name contains "nonce" become "[redacted]".
    The tool's return value is left as-is, so an MCP view still receives its nonce.
    """
    if isinstance(value, dict):
        return {k: ("[redacted]" if isinstance(k, str) and _secret_field(k) else redact_secrets(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


@dataclass
class ModelRoute:
    """Where every model call goes. One route serves planning, diagnostics and chat."""

    provider: str  # selfhosted | insforge | openrouter
    label: str  # shown in the studio and recorded on every attempt
    base_url: str  # OpenAI-compatible base, e.g. https://openrouter.ai/api/v1
    api_key: str
    planner_model: str
    vlm_model: str
    agent_model: str
    point_format: str = "pixels"  # how the model reports image points: pixels | norm1000
    openrouter_extras: bool = True  # send OpenRouter-only fields (reasoning, usage.include)
    price_per_m: dict[str, float] | None = None  # None = no per-token charge (self-hosted)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class Ctx:
    settings: Settings
    store: Store
    blobs: BlobStore
    repo: Repo
    bus: EventBus
    jobs: JobManager
    data_dir: Path
    model: ModelRoute | None = None
    vlm_source: str = "disabled"
    gateway_error: str | None = None  # why the InsForge Model Gateway gave no key (e.g. plan restriction)
    # signed media URLs, reused until close to expiry so polling doesn't reload every <video>
    signed_urls: dict[str, tuple[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ files
    def local(self, key: str) -> Path:
        """Working copy of a blob. Workers read/write here; the BlobStore is durable."""
        return self.data_dir / key

    async def publish_file(self, key: str) -> None:
        """Upload a local working file to the durable blob store (no-op for local disk)."""
        if isinstance(self.blobs, LocalBlobStore):
            return
        key = key.replace("\\", "/")  # a Windows relative path would be stored under %5C-encoded names (#367)
        p = self.local(key)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if hasattr(self.blobs, "put_file"):  # InsForge: REST, or the S3 gateway above the REST cap
            await self.blobs.put_file(key, p, ctype)
        else:
            await self.blobs.put_bytes(key, p.read_bytes(), ctype)

    async def ensure_local(self, key: str) -> Path:
        p = self.local(key)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(await self.blobs.get_bytes(key))
        return p

    async def media_url(self, key: str, ttl_s: int = 3600) -> str:
        if not isinstance(self.blobs, LocalBlobStore):
            hit = self.signed_urls.get(key)
            if hit and hit[1] - time.time() > 600:  # same URL until 10 minutes before it expires
                return hit[0]
            url = await self.blobs.url(key, ttl_s)
            if url:
                self.signed_urls[key] = (url, time.time() + ttl_s)
                return url
        return f"{self.settings.public_url}/files/{key}"

    # ------------------------------------------------------------ audit
    async def audit(self, actor: str, tool: str, project_id: str | None, args: dict[str, Any],
                    status: str, summary: str, approved: bool | None = None) -> None:
        call = CURRENT_CALL.get()
        if call and call.get("open"):
            # A service's own audit during a tool call in flight: fold it into that call's row
            # (written at the call's end) instead of extra rows for the same action.
            call.setdefault("domain", []).append(
                {"tool": tool, "status": status, "summary": summary, "approved": approved})
            return
        row_id = new_id("aud")
        session_id = call.get("session_id") if call else None
        await self.store.put("tool_audit_logs", {
            "id": row_id, "project_id": project_id, "status": status, "session_id": session_id,
            "data": {"id": row_id, "actor": actor, "tool": tool, "args": redact_secrets(args), "summary": summary,
                     "approved": approved, "session_id": session_id, "created_at": now().isoformat()},
            "created_at": now().isoformat()})
        if project_id:
            self.bus.publish(project_id, "entity", {"table": "tool_audit_logs", "id": row_id})

    def changed(self, project_id: str, table: str, id: str) -> None:
        self.bus.publish(project_id, "entity", {"table": table, "id": id})


async def build_ctx(s: Settings) -> Ctx:
    s.data_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    store: Store
    blobs: BlobStore
    if s.insforge_enabled:
        from .store.insforge import InsForgeBlobStore, InsForgeStore

        store = InsForgeStore(s.insforge_base_url, s.insforge_api_key)  # type: ignore[arg-type]
        s3_keys = ((s.insforge_s3_access_key_id, s.insforge_s3_secret_access_key)
                   if s.insforge_s3_access_key_id and s.insforge_s3_secret_access_key else None)
        blobs = InsForgeBlobStore(s.insforge_base_url, s.insforge_api_key, s.insforge_bucket,  # type: ignore[arg-type]
                                  rest_max_bytes=s.insforge_rest_max_mb * 1024 * 1024, s3_keys=s3_keys,
                                  s3_endpoint=s.insforge_s3_endpoint, s3_region=s.insforge_s3_region)
    else:
        store = SqliteStore(s.data_dir / "cortex.db")
        blobs = LocalBlobStore(s.data_dir, f"{s.public_url}/files")
        notes.append("InsForge not configured: using local SQLite + disk (fallback).")
    await store.init()
    await blobs.init()

    route, gw_err = await resolve_model_route(s)
    src = route.label if route else "disabled"
    if not route:
        why = f"InsForge Model Gateway: {gw_err}" if gw_err else "no model endpoint configured"
        notes.append(f"Models disabled ({why}). Agent planning and VLM diagnostics are off; the scripted "
                     "oracle, telemetry analysis and everything else still run.")

    bus = EventBus()
    return Ctx(settings=s, store=store, blobs=blobs, repo=Repo(store), bus=bus,
               jobs=JobManager(bus, persist=s.data_dir / "jobs.json"),
               data_dir=s.data_dir, model=route, vlm_source=src, gateway_error=gw_err, notes=notes)


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
ASTRA = "openai/gpt-6-astra"


async def resolve_model_route(s: Settings) -> tuple[ModelRoute | None, str | None]:
    """Self-hosted endpoint first (our own model), then OpenRouter via the InsForge gateway
    (or a self-hosted InsForge's OPENROUTER_API_KEY). Never mixes providers within a run."""
    if s.planner_base_url:
        m = s.selfhosted_model
        return ModelRoute(provider="selfhosted", label=f"self-hosted {m} ({s.planner_base_url})",
                          base_url=s.planner_base_url, api_key=s.planner_api_key or "none", planner_model=m,
                          vlm_model=m, agent_model=m, point_format=s.selfhosted_point_format,
                          openrouter_extras=False, price_per_m=None), None
    key, gw_err, label, provider = None, None, "", ""
    if s.openrouter_api_key:
        key, label, provider = s.openrouter_api_key, "OpenRouter (OPENROUTER_API_KEY)", "openrouter"
    elif s.insforge_enabled:
        from .store.insforge import gateway_key

        key, gw_err = await gateway_key(s.insforge_base_url, s.insforge_api_key)  # type: ignore[arg-type]
        label, provider = "InsForge Model Gateway → OpenRouter", "insforge"
    if not key:
        return None, gw_err
    return ModelRoute(provider=provider, label=label, base_url=OPENROUTER_BASE, api_key=key, planner_model=ASTRA,
                      vlm_model=s.vlm_model, agent_model=s.agent_model, point_format="pixels",
                      openrouter_extras=True, price_per_m={"input": 10.0, "output": 50.0}), None
