"""The local runner that relays a hosted MCP server's tool calls to this backend (cortex.mcp_runner)."""

import json

import httpx
import pytest

from cortex.mcp_runner import Runner, RunnerError, forward_headers, server_base

SERVER = "https://cortex.example.run"
API = "http://127.0.0.1:8000/api"


def runner(handler, key="k3y") -> Runner:
    return Runner(SERVER, key, API, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_server_base_accepts_mcp_endpoint():
    assert server_base("https://x.run.mcp-use.com/mcp/") == "https://x.run.mcp-use.com"
    assert server_base("http://localhost:3010") == "http://localhost:3010"


def test_forward_headers_drops_auth_and_host():
    h = forward_headers({"Authorization": "Bearer x", "Host": "evil", "X-Cortex-Actor": "mcp",
                         "Content-Type": "application/json", "Cookie": "c"})
    assert h == {"x-cortex-actor": "mcp", "content-type": "application/json"}
    assert forward_headers(None) == {}


def test_missing_key_refuses_to_start():
    with pytest.raises(RunnerError, match="CORTEX_MCP_KEY"):
        Runner(SERVER, "", API)


async def test_poll_sends_key_and_parses_calls():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        assert str(req.url) == f"{SERVER}/relay/poll"
        return httpx.Response(200, json={"calls": [{"id": "c1", "tool": "get_job", "args": {}}, {"junk": 1}]})

    calls = await runner(handler).poll_once()
    assert seen["auth"] == "Bearer k3y"
    assert [c["id"] for c in calls] == ["c1"]


async def test_idle_poll_returns_nothing():
    r = runner(lambda req: httpx.Response(204))
    assert await r.poll_once() == []
    assert r.connected


@pytest.mark.parametrize("status,match", [(401, "rejected the runner key"), (503, "relay disabled"),
                                          (404, "not found")])
async def test_unrecoverable_poll_errors_stop_the_runner(status, match):
    r = runner(lambda req: httpx.Response(status, json={"detail": "x"}))
    with pytest.raises(RunnerError, match=match):
        await r.run()  # run() must raise, not retry forever


async def test_handle_forwards_to_backend_and_returns_raw_response():
    posted = {}

    def handler(req: httpx.Request):
        if req.url.host == "127.0.0.1":
            posted["backend"] = (str(req.url), json.loads(req.content), dict(req.headers))
            return httpx.Response(422, json={"detail": [{"loc": ["project_id"], "msg": "required"}]})
        posted["result"] = (str(req.url), req.headers["authorization"], json.loads(req.content))
        return httpx.Response(200, json={"ok": True})

    await runner(handler).handle({"id": "c7", "tool": "get_next_actions", "args": {"project_id": "prj_1"},
                                  "headers": {"X-Cortex-Actor": "mcp", "Authorization": "Bearer leak"}})
    url, body, headers = posted["backend"]
    assert url == f"{API}/tools/get_next_actions" and body == {"project_id": "prj_1"}
    assert headers["x-cortex-actor"] == "mcp" and "authorization" not in headers
    url, auth, result = posted["result"]
    assert url == f"{SERVER}/relay/result" and auth == "Bearer k3y"
    assert result["id"] == "c7" and result["status"] == 422
    assert json.loads(result["body"])["detail"][0]["msg"] == "required"


async def test_backend_down_becomes_502_with_detail():
    posted = {}

    def handler(req: httpx.Request):
        if req.url.host == "127.0.0.1":
            raise httpx.ConnectError("refused", request=req)
        posted["result"] = json.loads(req.content)
        return httpx.Response(200, json={"ok": True})

    await runner(handler).handle({"id": "c9", "tool": "get_job", "args": {"job_id": "j"}})
    assert posted["result"]["status"] == 502
    assert "could not reach the backend" in json.loads(posted["result"]["body"])["detail"]


async def test_against_the_real_backend(tmp_path, monkeypatch):
    """In-process end to end: the runner forwards a registry call to the FastAPI app."""
    from fastapi.testclient import TestClient

    from cortex.config import settings

    for k, v in {"DATA_DIR": str(tmp_path), "INSFORGE_BASE_URL": "", "INSFORGE_API_KEY": ""}.items():
        monkeypatch.setenv(k, v)
    settings.cache_clear()
    from cortex.api.app import app

    posted = {}
    try:
        with TestClient(app) as tc:
            def handler(req: httpx.Request):
                if req.url.host == "127.0.0.1":
                    r = tc.post(req.url.path, content=req.content, headers=dict(req.headers))
                    return httpx.Response(r.status_code, content=r.content)
                posted[json.loads(req.content)["id"]] = json.loads(req.content)
                return httpx.Response(200, json={"ok": True})

            r = runner(handler)
            await r.handle({"id": "missing", "tool": "no_such_tool", "args": {}})
            await r.handle({"id": "bad", "tool": "get_next_actions", "args": {}})
    finally:
        settings.cache_clear()
    assert posted["missing"]["status"] == 404
    assert posted["bad"]["status"] == 422
