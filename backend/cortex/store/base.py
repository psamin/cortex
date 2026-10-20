"""Storage interfaces. Services depend only on these, never on a provider."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

TABLES = (
    "projects", "robot_configs", "episodes", "episode_artifacts", "dataset_versions",
    "dataset_episode_links", "analyses", "failure_clusters", "recommendations",
    "training_jobs", "checkpoints", "evaluation_runs", "simulation_branches",
    "tool_audit_logs", "autopilot_runs", "agent_collections", "agent_attempts", "rl_sandboxes", "rl_rounds",
    "agent_keys", "agent_sessions", "policy_replays", "real_trials",
)

M = TypeVar("M", bound=BaseModel)


class Store(Protocol):
    name: str

    async def init(self) -> None: ...
    async def put(self, table: str, row: dict[str, Any]) -> None:
        """Upsert by id. Row keys: id, project_id, status, data (dict), created_at."""
    async def get(self, table: str, id: str) -> dict[str, Any] | None: ...
    async def list(self, table: str, project_id: str | None = None,
                   status: str | None = None, where: dict[str, str] | None = None) -> list[dict[str, Any]]:
        """Rows ordered by created_at ascending. `where`: equality filters on extra columns."""


class BlobStore(Protocol):
    name: str

    async def init(self) -> None: ...
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> str: ...
    async def get_bytes(self, key: str) -> bytes: ...
    async def url(self, key: str, expires_s: int = 3600) -> str:
        """A URL the browser can fetch (signed where the provider supports it)."""


class Repo:
    """Typed persistence on top of a Store: model <-> row conversion."""

    def __init__(self, store: Store):
        self.store = store

    async def save(self, table: str, obj: BaseModel, project_id: str | None = None,
                   status: str | None = None) -> None:
        data = obj.model_dump(mode="json")
        row = {
            "id": data["id"],
            "project_id": project_id or data.get("project_id"),
            "status": status if status is not None else _status(data),
            "data": data,
            "created_at": data.get("created_at"),
        }
        if table == "projects":
            row.update(project_id=data["id"], name=data["name"], mode=data["mode"])
        await self.store.put(table, row)

    async def get(self, table: str, id: str, model: type[M]) -> M | None:
        row = await self.store.get(table, id)
        return model.model_validate(row["data"]) if row else None

    async def list(self, table: str, model: type[M], project_id: str | None = None,
                   status: str | None = None) -> list[M]:
        rows = await self.store.list(table, project_id=project_id, status=status)
        return [model.model_validate(r["data"]) for r in rows]


def _status(data: dict[str, Any]) -> str | None:
    s = data.get("status")
    return str(s) if s is not None else None
