"""Application context: providers selected from settings, shared by REST, MCP
(via REST) and the agent. Services receive a Ctx and never pick providers."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .models import new_id, now
from .services.jobs import EventBus, JobManager
from .store.base import BlobStore, Repo, Store
from .store.local import LocalBlobStore, SqliteStore

log = logging.getLogger(__name__)


@dataclass
class Ctx:
    settings: Settings
    store: Store
    blobs: BlobStore
    repo: Repo
    bus: EventBus
    jobs: JobManager
    data_dir: Path
    openrouter_key: str | None = None
    vlm_source: str = "disabled"
    gateway_error: str | None = None  # why the InsForge Model Gateway gave no key (e.g. plan restriction)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ files
    def local(self, key: str) -> Path:
        """Working copy of a blob. Workers read/write here; the BlobStore is durable."""
        return self.data_dir / key

    async def publish_file(self, key: str) -> None:
        """Upload a local working file to the durable blob store (no-op for local disk)."""
        if isinstance(self.blobs, LocalBlobStore):
            return
        p = self.local(key)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        await self.blobs.put_bytes(key, p.read_bytes(), ctype)

    async def ensure_local(self, key: str) -> Path:
        p = self.local(key)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(await self.blobs.get_bytes(key))
        return p

    async def media_url(self, key: str) -> str:
        if not isinstance(self.blobs, LocalBlobStore):
            url = await self.blobs.url(key)
            if url:
                return url
        return f"{self.settings.public_url}/files/{key}"

    # ------------------------------------------------------------ audit
    async def audit(self, actor: str, tool: str, project_id: str | None, args: dict[str, Any],
                    status: str, summary: str, approved: bool | None = None) -> None:
        row_id = new_id("aud")
        await self.store.put("tool_audit_logs", {
            "id": row_id, "project_id": project_id, "status": status,
            "data": {"id": row_id, "actor": actor, "tool": tool, "args": args, "summary": summary,
                     "approved": approved, "created_at": now().isoformat()},
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
        blobs = InsForgeBlobStore(s.insforge_base_url, s.insforge_api_key, s.insforge_bucket)  # type: ignore[arg-type]
    else:
        store = SqliteStore(s.data_dir / "cortex.db")
        blobs = LocalBlobStore(s.data_dir, f"{s.public_url}/files")
        notes.append("InsForge not configured: using local SQLite + disk (fallback).")
    await store.init()
    await blobs.init()

    key, src, gw_err = None, "disabled", None
    if s.openrouter_api_key:
        key, src = s.openrouter_api_key, "openrouter (OPENROUTER_API_KEY)"
    elif s.insforge_enabled:
        from .store.insforge import gateway_key

        key, gw_err = await gateway_key(s.insforge_base_url, s.insforge_api_key)  # type: ignore[arg-type]
        src = "insforge model gateway (openrouter)" if key else "disabled"
    if not key:
        why = f"InsForge Model Gateway: {gw_err}" if gw_err else "no InsForge Model Gateway key"
        notes.append(f"Models disabled ({why}). GPT-6 Astra planning and VLM diagnostics are off; the scripted "
                     "oracle, telemetry analysis and everything else still run.")

    bus = EventBus()
    return Ctx(settings=s, store=store, blobs=blobs, repo=Repo(store), bus=bus, jobs=JobManager(bus),
               data_dir=s.data_dir, openrouter_key=key, vlm_source=src, gateway_error=gw_err, notes=notes)
