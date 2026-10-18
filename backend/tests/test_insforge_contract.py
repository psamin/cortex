"""Contract test for the InsForge adapters against a strict in-memory fake that
follows InsForge's documented REST shapes (docs/sdks/rest/{database,storage,ai}.mdx).
It checks our request shapes and round-trips; it is not a substitute for a run
against a real InsForge project."""

import json
import re
from urllib.parse import parse_qs, unquote

import httpx
import pytest

from cortex.models import AgentAttempt, AgentCollection, Scenario, StartAgentCollectionRequest
from cortex.store.base import Repo
from cortex.store.insforge import InsForgeBlobStore, InsForgeStore

KEY = "ik_test"


class FakeInsForge:
    def __init__(self):
        self.migrations: list[dict] = []
        self.tables: dict[str, dict[str, dict]] = {}
        self.buckets: dict[str, dict] = {}
        self.objects: dict[tuple[str, str], bytes] = {}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == f"Bearer {KEY}"
        path, q = req.url.path, parse_qs(req.url.query.decode())
        if path == "/api/database/migrations":
            if req.method == "GET":
                return httpx.Response(200, json=self.migrations)
            body = json.loads(req.content)
            assert set(body) == {"version", "name", "sql"} and re.fullmatch(r"[a-z0-9-]+", body["name"])
            assert "BEGIN" not in body["sql"].upper().split()
            self.migrations.append(body)
            for t in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", body["sql"]):
                self.tables.setdefault(t, {})
            return httpx.Response(201, json={"ok": True})
        if m := re.fullmatch(r"/api/database/records/(\w+)", path):
            table = self.tables[m.group(1)]  # KeyError -> unknown table, like PostgREST 404
            if req.method == "POST":
                assert "merge-duplicates" in req.headers.get("prefer", "")
                rows = json.loads(req.content)
                assert isinstance(rows, list), "InsForge requires array bodies"
                for r in rows:
                    table[r["id"]] = {**table.get(r["id"], {}), **r}
                return httpx.Response(201)
            rows = list(table.values())
            for col, cond in q.items():
                if col in ("order", "limit", "offset", "select"):
                    continue
                op, val = cond[0].split(".", 1)
                assert op == "eq"
                rows = [r for r in rows if str(r.get(col)) == val]
            if "order" in q:
                cols = [c.split(".")[0] for c in q["order"][0].split(",")]
                rows.sort(key=lambda r: tuple(str(r.get(c) or "") for c in cols))
            off, lim = int(q.get("offset", ["0"])[0]), int(q.get("limit", ["100"])[0])
            return httpx.Response(200, json=rows[off: off + lim])
        if path == "/api/storage/buckets" and req.method == "POST":
            body = json.loads(req.content)
            assert body["isPublic"] is False
            if body["bucketName"] in self.buckets:
                return httpx.Response(409, json={"error": "bucket already exists"})
            self.buckets[body["bucketName"]] = body
            return httpx.Response(201, json=body)
        if m := re.fullmatch(r"/api/storage/buckets/([^/]+)/download-strategy/objects/(.+)", path):
            return httpx.Response(200, json={"method": "presigned", "url": f"https://s3.example/{unquote(m.group(2))}?sig=x"})
        if m := re.fullmatch(r"/api/storage/buckets/([^/]+)/objects/(.+)", path):
            k = (m.group(1), unquote(m.group(2)))
            if req.method == "PUT":
                assert req.headers["content-type"].startswith("multipart/form-data")
                assert b'name="file"' in req.content
                self.objects[k] = req.content.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
                return httpx.Response(201, json={"key": k[1]})
            return httpx.Response(200, content=self.objects[k])
        return httpx.Response(404, json={"error": f"no route {req.method} {path}"})


@pytest.fixture
def fake():
    return FakeInsForge()


async def test_migrations_records_and_agent_artifacts(fake):
    t = httpx.MockTransport(fake)
    store = InsForgeStore("https://app.insforge.test", KEY, transport=t)
    await store.init()
    assert [m["name"] for m in fake.migrations] == ["cortex-001-init", "cortex-002-autopilot", "cortex-003-agent-demos",
                                                    "cortex-004-enable-rls", "cortex-005-rl",
                                                    "cortex-006-agent-obs", "cortex-007-policy-replays"]
    assert "ENABLE ROW LEVEL SECURITY" in fake.migrations[-1]["sql"]
    await store.init()  # idempotent: history is read first
    assert len(fake.migrations) == 7

    repo = Repo(store)
    from cortex.models import CreateProjectRequest
    from cortex.services.project_service import create_project
    p = create_project(CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    await repo.save("projects", p, status="active")
    col = AgentCollection(project_id=p.id, collector="astra", provider="Claude Code (your agent) via MCP",
                          model="claude-code", request=StartAgentCollectionRequest(num_demos=2))
    await repo.save("agent_collections", col)
    for i, st in enumerate(("success", "failed")):
        await repo.save("agent_attempts", AgentAttempt(
            project_id=p.id, collection_id=col.id, index=i, collector="astra", seed=20_000 + i, scenario=Scenario(),
            status=st, provider=col.provider, model=col.model, usage={"calls": 9, "cost_usd": 0.41}))
    got = await repo.list("agent_attempts", AgentAttempt, project_id=p.id)
    # rows saved in the same clock tick tie on created_at and are ordered by id, so compare by index
    assert {a.index: a.status for a in got} == {0: "success", 1: "failed"}
    assert {a.model for a in got} == {"claude-code"}
    assert len(await repo.list("agent_attempts", AgentAttempt, project_id=p.id, status="failed")) == 1
    assert (await repo.get("projects", p.id, type(p))).name == p.name
    assert fake.tables["projects"][p.id]["project_id"] == p.id  # uniform (id, project_id, status, data) rows


async def test_storage_roundtrip_and_presign(fake):
    t = httpx.MockTransport(fake)
    blobs = InsForgeBlobStore("https://app.insforge.test", KEY, "cortex-assets", transport=t)
    await blobs.init()
    await blobs.init()  # existing bucket is fine
    key = "projects/prj_x/attempts/at_1/trace.json"
    await blobs.put_bytes(key, b'{"trace": []}', "application/json")
    assert await blobs.get_bytes(key) == b'{"trace": []}'
    assert (await blobs.url(key)).startswith("https://s3.example/")


async def test_rejected_migration_fails_loudly(fake):
    """InsForge rejects versions not newer than the latest applied one; that must not be skipped silently."""
    def handler(req):
        if req.url.path == "/api/database/migrations" and req.method == "POST":
            return httpx.Response(400, json={"error": "DATABASE_MIGRATION_ALREADY_EXISTS",
                                             "message": "version already exists or is not newer"})
        return fake(req)
    store = InsForgeStore("https://app.insforge.test", KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="ALREADY_EXISTS"):
        await store.init()


async def test_list_pages_past_one_request(fake):
    """More rows than one records request returns: all of them come back, newest included, in order."""
    store = InsForgeStore("https://app.insforge.test", KEY, transport=httpx.MockTransport(fake))
    await store.init()
    fake.tables["tool_audit_logs"] = {
        f"aud_{i:05d}": {"id": f"aud_{i:05d}", "project_id": "prj_1", "status": "ok", "data": {"i": i},
                         "created_at": f"2026-10-17T00:00:{i // 1000:02d}.{i % 1000:03d}"}
        for i in range(2500)}
    fake.tables["tool_audit_logs"]["aud_other"] = {"id": "aud_other", "project_id": "prj_2", "status": "ok",
                                                   "data": {}, "created_at": "2026-10-17T00:00:00"}
    rows = await store.list("tool_audit_logs", project_id="prj_1")
    assert len(rows) == 2500
    assert [r["data"]["i"] for r in rows] == list(range(2500))
