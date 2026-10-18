"""Agent observability (#105): keys, sessions, start/end call rows, images, scoping, events."""

import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from cortex.config import Settings
from cortex.services import agent_obs, insforge_auth


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def _ctx():
    from cortex.api.app import ctx

    return ctx()


def agent(key=None, session="mcp-sess-1", client="claude-code/2.1.0"):
    h = {"X-Cortex-Actor": "mcp", "X-Cortex-Agent-Session": session, "X-Cortex-Client": client}
    if key:
        h["X-Cortex-Agent-Key"] = key
    return h


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_key_session_and_call_timeline(client):
    k = client.post("/api/agents/keys", json={"label": "laptop claude"}).json()
    assert k["key"].startswith("ck_") and k["status"] == "active"
    listed = client.get("/api/agents/keys").json()
    assert [x["id"] for x in listed] == [k["id"]] and "key" not in listed[0] and "key_hash" not in listed[0]

    q = _ctx().bus.subscribe(agent_obs.channel("local"))
    r = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=agent(k["key"]))
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    r = client.post("/api/tools/update_intake", json={"project_id": pid, "patch": {"task": {"pik": "x"}}},
                    headers=agent(k["key"]))
    assert r.status_code == 400

    events = [e for e, _ in drain(q)]
    assert events[:3] == ["session_start", "tool_start", "tool_end"] and events.count("tool_end") == 2

    (s,) = client.get("/api/agents/sessions").json()
    assert s["client"] == "claude-code" and s["client_version"] == "2.1.0" and s["live"]
    assert s["call_count"] == 2 and s["projects"] == [pid] and s["key_id"] == k["id"]
    calls = client.get(f"/api/agents/sessions/{s['id']}/calls").json()
    assert [(c["tool"], c["status"]) for c in calls] == [("start_intake", "ok"), ("update_intake", "error")]
    assert all(isinstance(c["duration_ms"], int) for c in calls) and "pik" in calls[1]["error"]
    # the services' own audits are folded into the call row, not extra rows
    assert {d["tool"] for d in calls[0]["domain"]} >= {"create_project"}
    audit = client.get(f"/api/projects/{pid}/audit").json()
    assert sum(1 for a in audit if a["tool"] in ("start_intake", "create_project")) == 1

    link = client.post("/api/tools/get_session_link", json={}, headers=agent(k["key"])).json()
    assert link["url"].endswith(f"/studio/agents?session={s['id']}")


def test_images_are_stored_and_approval_is_flagged(client, monkeypatch):
    from cortex.services import agent_orchestrator as orch
    from cortex.services.training_service import ApprovalRequired

    jpg = b"\xff\xd8\xff\xe0fake-jpeg"

    async def snap(ctx, a, actor):
        return {"summary": "frame", "ref_ids": [], "image_b64": base64.b64encode(jpg).decode()}

    async def gated(ctx, a, actor):
        raise ApprovalRequired("approve in the studio")

    monkeypatch.setitem(orch.TOOLS, "t_snap", orch.Tool("t_snap", "", orch.NoArgs, snap))
    monkeypatch.setitem(orch.TOOLS, "t_gated", orch.Tool("t_gated", "", orch.NoArgs, gated))
    q = _ctx().bus.subscribe(agent_obs.channel("local"))
    assert client.post("/api/tools/t_snap", json={}, headers=agent()).status_code == 200
    r = client.post("/api/tools/t_gated", json={}, headers=agent())
    assert r.status_code == 403 and r.json()["approval_required"]

    (s,) = client.get("/api/agents/sessions").json()
    snap_call, gated_call = client.get(f"/api/agents/sessions/{s['id']}/calls").json()
    assert snap_call["images"] and "image_b64" not in snap_call["result_preview"]
    assert asyncio.run(_ctx().blobs.get_bytes(snap_call["images"][0])) == jpg
    assert snap_call["image_urls"][0].endswith(snap_call["images"][0])
    assert gated_call["status"] == "approval_required"
    assert "approval_required" in [e for e, _ in drain(q)]


def test_keys_are_scoped_and_revocable(client):
    ctx = _ctx()
    other = asyncio.run(agent_obs.create_key(ctx, "user_2", "theirs"))
    r = client.post("/api/tools/get_session_link", json={}, headers=agent(other["key"], session="s2"))
    assert r.status_code == 200
    assert client.get("/api/agents/sessions").json() == []  # user_2's session is not the local owner's
    sid = r.json()["session_id"]
    assert client.get(f"/api/agents/sessions/{sid}/calls").status_code == 404
    assert client.delete(f"/api/agents/keys/{other['id']}").status_code == 404  # not yours to revoke

    mine = client.post("/api/agents/keys", json={}).json()
    assert client.delete(f"/api/agents/keys/{mine['id']}").json()["status"] == "revoked"
    r = client.post("/api/tools/get_session_link", json={}, headers=agent(mine["key"]))
    assert r.status_code == 403 and r.json()["code"] == "agent_key_invalid" and not r.json()["approval_required"]
    r = client.post("/api/tools/get_session_link", json={}, headers=agent("ck_made_up"))
    assert r.status_code == 403


def test_agents_cannot_manage_keys_approve_or_drive_hardware(client):
    k = client.post("/api/agents/keys", json={}).json()
    assert client.post("/api/agents/keys", json={}, headers=agent(k["key"])).status_code == 403
    assert client.get("/api/agents/sessions", headers={"X-Cortex-Actor": "mcp"}).status_code == 403
    assert client.post("/api/hardware/disarm", headers=agent(k["key"])).status_code in (403, 409)

    from cortex.api.app import actor

    def req(h):
        return Request({"type": "http", "headers": [(a.lower().encode(), b.encode()) for a, b in h.items()]})
    assert actor(req({})) == "gui"
    assert actor(req({"X-Cortex-Agent-Key": k["key"]})) == "mcp"  # so training approval treats it as an agent
    assert actor(req({"X-Cortex-Agent-Session": "x", "X-Cortex-Actor": "gui"})) == "mcp"


def test_idle_sessions_emit_session_idle(client):
    client.post("/api/tools/get_session_link", json={}, headers=agent())
    q = _ctx().bus.subscribe(agent_obs.channel("local"))
    assert asyncio.run(agent_obs.mark_idle(_ctx(), idle_after_s=0)) == 1
    assert [e for e, _ in drain(q)] == ["session_idle"]
    (s,) = client.get("/api/agents/sessions").json()
    assert s["status"] == "idle" and not s["live"]


def test_secrets_are_redacted_in_the_audit_and_on_the_bus(client, monkeypatch):
    from cortex.services import agent_orchestrator as orch

    async def leak(ctx, a, actor):
        return {"summary": "ok", "view_approval_nonce": "nonce-secret",
                "nested": {"approval_token": "nested-secret"}}

    monkeypatch.setitem(orch.TOOLS, "t_leak", orch.Tool("t_leak", "", orch.NoArgs, leak))
    q = _ctx().bus.subscribe(agent_obs.channel("local"))
    r = client.post("/api/tools/t_leak", json={"approval_token": "op-secret", "note": "keep"}, headers=agent())
    assert r.status_code == 200, r.text
    assert r.json()["view_approval_nonce"] == "nonce-secret"  # the caller still receives it
    (s,) = client.get("/api/agents/sessions").json()
    (call,) = client.get(f"/api/agents/sessions/{s['id']}/calls").json()
    assert call["args"] == {"approval_token": "[redacted]", "note": "keep"}
    assert "nonce-secret" not in call["result_preview"] and "nested-secret" not in call["result_preview"]
    assert "[redacted]" in call["result_preview"]
    for _, payload in drain(q):
        blob = json.dumps(payload, default=str)
        assert "op-secret" not in blob and "nonce-secret" not in blob and "nested-secret" not in blob

    ctx = _ctx()
    asyncio.run(ctx.audit("mcp", "launch_training_job", None, {"approval_token": "sek", "id": "tj"}, "ok", "x"))
    rows = asyncio.run(ctx.store.list("tool_audit_logs"))
    assert rows[-1]["data"]["args"] == {"approval_token": "[redacted]", "id": "tj"}


def test_remote_mcp_key_comparison_rejects_a_mismatch(client, monkeypatch):
    from cortex.config import settings

    monkeypatch.setenv("CORTEX_MCP_KEY", "tunnel-secret")
    settings.cache_clear()
    try:
        remote = {"X-Forwarded-For": "8.8.8.8"}
        assert client.get("/api/health", headers=remote).status_code == 200
        wrong = client.post("/api/tools/get_session_link", json={},
                            headers={**remote, "Authorization": "Bearer wrong"})
        assert wrong.status_code == 403
        ok = client.post("/api/tools/get_session_link", json={},
                         headers={**remote, "Authorization": "Bearer tunnel-secret"})
        assert ok.status_code == 200, ok.text
    finally:
        settings.cache_clear()


def test_me_reports_local_mode(client):
    me = client.get("/api/agents/me").json()
    assert me["mode"] == "local" and me["user"]["id"] == "local" and "single user" in me["note"]


def test_insforge_auth_sign_in_and_verify():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/auth/sessions" and req.method == "POST":
            return httpx.Response(200, json={"user": {"id": "u1", "email": "a@b.c"}, "accessToken": "jwt1"})
        if req.url.path == "/api/auth/sessions/current":
            ok = req.headers.get("authorization") == "Bearer jwt2"
            return httpx.Response(200 if ok else 401, json={"user": {"id": "u1", "email": "a@b.c"}} if ok else {})
        return httpx.Response(404)

    s = Settings(insforge_base_url="https://x.insforge.app", insforge_api_key="ik_x")
    t = httpx.MockTransport(handler)
    out = asyncio.run(insforge_auth.sign_in(s, "a@b.c", "pw", transport=t))
    assert out["access_token"] == "jwt1" and out["user"]["id"] == "u1"
    assert asyncio.run(insforge_auth.current_user(s, "jwt2", transport=t))["id"] == "u1"
    with pytest.raises(insforge_auth.SignInRequired):
        asyncio.run(insforge_auth.current_user(s, "bad", transport=t))
    with pytest.raises(insforge_auth.SignInRequired):
        asyncio.run(insforge_auth.current_user(s, None, transport=t))


def test_verify_agent_key_tool_works_through_the_tools_path(client):
    """The hosted /mcp gate (#149) checks keys via /api/tools, the only route a tunnel or the runner relay reaches."""
    k = client.post("/api/agents/keys", json={}).json()
    ok = client.post("/api/tools/verify_agent_key", json={}, headers=agent(k["key"]))
    assert ok.status_code == 200 and ok.json()["valid"] and ok.json()["key_id"] == k["id"]
    none = client.post("/api/tools/verify_agent_key", json={}, headers=agent())
    assert none.status_code == 403 and none.json()["code"] == "agent_key_invalid"
    client.delete(f"/api/agents/keys/{k['id']}")
    for key in (k["key"], "ck_nope"):
        r = client.post("/api/tools/verify_agent_key", json={}, headers=agent(key))
        assert r.status_code == 403 and r.json()["code"] == "agent_key_invalid"


def test_project_log_shows_every_agent_on_the_project(client):
    """The chat's log mode: MCP sessions and the studio agent, on one project, in order, live via entity events."""
    pid = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=agent()).json()["project_id"]
    q = _ctx().bus.subscribe(pid)
    client.post("/api/tools/get_intake_status", json={"project_id": pid}, headers={"X-Cortex-Actor": "agent"})
    calls = client.get(f"/api/agents/projects/{pid}/calls").json()
    assert [c["tool"] for c in calls] == ["start_intake", "get_intake_status"]
    assert calls[0]["client"] == "claude-code" and calls[1]["actor"] == "agent"
    assert all("image_urls" in c for c in calls)
    assert any(ev == "entity" and d.get("table") == "tool_audit_logs" for ev, d in drain(q))
    assert client.get("/api/agents/projects/prj_missing/calls").json() == []


def test_agent_decisions_land_in_the_project_log(client):
    """log_decision: the agent narrates its plan; the note is a call row in the project's Agent log."""
    pid = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=agent()).json()["project_id"]
    r = client.post("/api/tools/log_decision", headers=agent(),
                    json={"project_id": pid, "kind": "plan", "text": "Collect 10 demos first, then train ACT."})
    assert r.status_code == 200 and r.json()["kind"] == "plan", r.text
    assert client.post("/api/tools/log_decision", headers=agent(), json={"project_id": "prj_nope", "text": "x"}).status_code == 404
    calls = client.get(f"/api/agents/projects/{pid}/calls").json()
    note = [c for c in calls if c["tool"] == "log_decision"]
    assert len(note) == 1 and note[0]["args"]["kind"] == "plan" and "10 demos" in note[0]["args"]["text"]


def test_agent_names_itself_and_is_labelled_everywhere(client):
    """cortex.register_agent: the agent picks a name; its session, later calls and the reply to its user carry it."""
    r = client.post("/api/tools/register_agent", json={"name": "  Atlas ", "purpose": "train the mug task"},
                    headers=agent(session="named-1"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "Atlas" and body["session_named"] and body["tell_user"].startswith("Connected to Cortex as Atlas")
    pid = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=agent(session="named-1")).json()["project_id"]
    (s,) = [x for x in client.get("/api/agents/sessions").json() if x["agent_name"] == "Atlas"]
    assert s["purpose"] == "train the mug task"
    calls = client.get(f"/api/agents/projects/{pid}/calls").json()
    assert calls and calls[-1]["agent_name"] == "Atlas"


def test_studio_chat_agent_works_in_its_chats_project(client):
    """X-Cortex-Project (set by the studio chat's MCP config): calls that leave out project_id stay in that project."""
    pid = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=agent()).json()["project_id"]
    bound = {**agent(session="studio-chat"), "X-Cortex-Project": pid}
    r = client.post("/api/tools/start_intake", json={"prompt": "teach a robot to play jenga"}, headers=bound)
    assert r.status_code == 200 and r.json()["project_id"] == pid, r.text  # reused, not a new project
    assert client.post("/api/tools/get_intake_status", json={}, headers=bound).status_code == 200
    calls = client.get(f"/api/agents/projects/{pid}/calls").json()
    assert [c["tool"] for c in calls][-2:] == ["start_intake", "get_intake_status"]
    other = client.post("/api/tools/start_intake", json={"prompt": "x", "project_id": pid}, headers=bound)
    assert other.json()["project_id"] == pid  # an explicit project_id still wins
