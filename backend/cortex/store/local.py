"""Local fallback backends: SQLite for metadata, a directory for blobs.

Used when InsForge credentials are absent. The UI and /api/health report which
backend is active so the fallback is always disclosed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import Store

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def pg_to_sqlite(sql: str) -> str:
    sql = re.sub(r"(?im)^\s*ALTER TABLE \w+ ENABLE ROW LEVEL SECURITY;\s*$", "", sql)  # Postgres-only
    sql = re.sub(r"\bJSONB\b", "TEXT", sql)
    sql = re.sub(r"\bTIMESTAMPTZ\b", "TEXT", sql)
    return sql.replace("DEFAULT now()", "DEFAULT CURRENT_TIMESTAMP")


class SqliteStore(Store):
    name = "sqlite (local fallback)"

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY)")
        done = {r[0] for r in self._conn.execute("SELECT name FROM schema_migrations")}
        for f in sorted(MIGRATIONS.glob("*.sql")):
            if f.name in done:
                continue
            self._conn.executescript(pg_to_sqlite(f.read_text()))
            self._conn.execute("INSERT INTO schema_migrations VALUES (?)", (f.name,))
        self._conn.commit()

    async def _run(self, fn):
        async with self._lock:
            return await asyncio.to_thread(fn)

    async def put(self, table: str, row: dict[str, Any]) -> None:
        assert self._conn
        created = row.get("created_at") or datetime.now(UTC).isoformat()
        conn = self._conn

        cols = {**{k: v for k, v in row.items() if k not in ("data", "created_at")},
                "data": json.dumps(row["data"]), "created_at": created}
        names = list(cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in names if c not in ("id", "created_at"))

        def go():
            conn.execute(
                f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}", [cols[c] for c in names])
            conn.commit()
        await self._run(go)

    async def get(self, table: str, id: str) -> dict[str, Any] | None:
        assert self._conn
        conn = self._conn
        r = await self._run(lambda: conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (id,)).fetchone())
        return _row(r) if r else None

    async def list(self, table: str, project_id: str | None = None,
                   status: str | None = None) -> list[dict[str, Any]]:
        assert self._conn
        q, args = f"SELECT * FROM {table} WHERE 1=1", []
        if project_id is not None:
            q, args = q + " AND project_id = ?", [*args, project_id]
        if status is not None:
            q, args = q + " AND status = ?", [*args, status]
        conn = self._conn
        rows = await self._run(lambda: conn.execute(q + " ORDER BY created_at, rowid", args).fetchall())
        return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["data"] = json.loads(d["data"])
    return d


class LocalBlobStore:
    name = "local disk (fallback)"

    def __init__(self, root: Path, public_base: str):
        self.root = root
        self.public_base = public_base.rstrip("/")

    async def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError(f"bad key {key!r}")
        return p

    async def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)
        return key

    async def get_bytes(self, key: str) -> bytes:
        return await asyncio.to_thread(self.path(key).read_bytes)

    async def url(self, key: str, expires_s: int = 3600) -> str:
        return f"{self.public_base}/{key}"
