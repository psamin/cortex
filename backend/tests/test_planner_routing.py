import asyncio
from types import SimpleNamespace

import pytest

from cortex.models import StartAgentCollectionRequest
from cortex.services import agent_collection_service as acs
from cortex.services import claude_code_agent as cc


def route(monkeypatch, planner, cli_ok, mcp_ok, engine="auto"):
    monkeypatch.setattr(cc, "available", lambda: None if cli_ok else "the `claude` CLI is not on PATH")

    async def up():
        return mcp_ok
    monkeypatch.setattr(cc, "mcp_up", up)
    ctx = SimpleNamespace(settings=SimpleNamespace(agent_engine=engine))
    return asyncio.run(acs.use_claude_code(ctx, StartAgentCollectionRequest(num_demos=1, planner=planner)))


def test_auto_prefers_claude_code_and_falls_back(monkeypatch):
    assert route(monkeypatch, "auto", True, True) is True
    assert route(monkeypatch, "auto", True, False) is False  # MCP server down -> model route
    assert route(monkeypatch, "auto", True, True, engine="model") is False  # engine pinned to the model route
    assert route(monkeypatch, "model", True, True) is False


def test_explicit_claude_code_fails_loudly_when_unavailable(monkeypatch):
    with pytest.raises(acs.PlannerUnavailable, match="Claude Code"):
        route(monkeypatch, "claude_code", False, True)
