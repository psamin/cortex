import asyncio
import json

import httpx

from cortex.store.insforge import MIGRATIONS, InsForgeStore, migration_name


def test_a_migration_numbered_below_the_latest_applied_one_is_applied_next_not_skipped():
    # The shared DB already ran another branch's 005 (agent-obs); main now has its own 005 (rl).
    files = sorted(MIGRATIONS.glob("*.sql"))
    applied = [{"name": migration_name(f), "version": f"2026101700{i + 1:04d}"} for i, f in enumerate(files[:4])]
    applied.append({"name": "cortex-005-agent-obs", "version": "20261017000005"})
    posted = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            return httpx.Response(200, json=applied + posted)
        body = json.loads(req.content)
        if int(body["version"]) <= max(int(m["version"]) for m in applied + posted):  # InsForge's rule
            return httpx.Response(409, json={"error": "DATABASE_MIGRATION_ALREADY_EXISTS"})
        posted.append(body)
        return httpx.Response(200, json={})

    asyncio.run(InsForgeStore("https://app.insforge.test", "k", transport=httpx.MockTransport(handler)).init())
    new = [f for f in files[4:]]
    assert [p["name"] for p in posted] == [migration_name(f) for f in new]
    assert [int(p["version"]) for p in posted] == list(range(20261017000006, 20261017000006 + len(new)))
