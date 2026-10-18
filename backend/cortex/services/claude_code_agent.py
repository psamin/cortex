"""The studio's agent, powered by the user's own Claude Code.

Each chat message runs `claude -p` headless with ONLY the Cortex MCP server and
only Cortex's tools allowed (no shell, no file edits), resuming one Claude Code
session per project so the conversation carries over. Claude's text and tool
calls stream back to the studio as SSE. Every Cortex tool call goes through the
MCP server, so it lands in the audit log as actor "mcp" and keeps all approval
gates (training, budgets, hardware).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..context import Ctx
from . import agent_orchestrator as orch

MCP_URL = "http://localhost:3000/mcp"
MAX_TURNS = 40
SESSIONS: dict[str, str] = {}  # project id -> Claude Code session id (also persisted on disk)

SYSTEM = """You are the agent inside Cortex, a robot-learning studio. The user talks to you in the Cortex app;
you act through the cortex MCP tools only. Current project: {pid} ("{name}": {goal}). Always pass
project_id="{pid}" to tools that take one.
Style: act, don't ask. Call tools to make progress, then reply in 2-4 short plain sentences about what
happened and what the evidence shows. No JSON, no ids unless the user asks, no markdown tables.
Rules:
- Only report numbers that tools returned. Simulator results are measured; VLM/agent diagnoses are hypotheses.
- Training on the GPU cluster needs the human's approval in the app: propose it, then tell the user to approve.
- Long jobs (collection, analysis, datasets, evaluation) return job ids: wait with short polls
  (cortex.get_job / cortex.get_collection_progress) and report when done.
- Scripted-oracle demos read simulator state; say so when you use them."""


def available() -> str | None:
    """None if Claude Code can run here, else why not."""
    if not shutil.which("claude"):
        return "the `claude` CLI is not on PATH"
    return None


async def mcp_up() -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                             headers={"accept": "application/json, text/event-stream"})
        return r.status_code < 500
    except Exception:
        return False


def _session_file(ctx: Ctx, pid: str) -> Path:
    return ctx.data_dir / "agent_sessions" / f"{pid}.json"


def _load_session(ctx: Ctx, pid: str) -> str | None:
    if pid in SESSIONS:
        return SESSIONS[pid]
    f = _session_file(ctx, pid)
    if f.exists():
        SESSIONS[pid] = json.loads(f.read_text()).get("session_id")
    return SESSIONS.get(pid)


def _save_session(ctx: Ctx, pid: str, sid: str) -> None:
    SESSIONS[pid] = sid
    f = _session_file(ctx, pid)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"session_id": sid}))


def _short_tool(name: str) -> str:
    # Claude Code names MCP tools mcp__<server>__<tool>; ours are cortex.<x> -> cortex_<x>
    return name.split("__")[-1].removeprefix("cortex_")


def _summarize_result(content: Any) -> tuple[str, list[str], bool]:
    """(one-line summary, ref ids, approval_required) from a tool_result content block.
    The Cortex MCP server writes "<summary>\nref_ids: a, b" as its text content."""
    text = ""
    if isinstance(content, list):
        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    elif isinstance(content, str):
        text = content
    try:  # Claude Code hands the model our structuredContent as JSON text
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            return (str(data["summary"])[:240], [str(r) for r in data.get("ref_ids", [])][:12],
                    bool(data.get("approval_required")) or "APPROVAL REQUIRED" in text)
    except ValueError:
        pass
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    refs: list[str] = []
    for ln in lines:
        if ln.startswith("ref_ids:"):
            refs = [r.strip() for r in ln.removeprefix("ref_ids:").split(",") if r.strip()]
    summary = next((ln for ln in lines if not ln.startswith("ref_ids:")), "")
    return summary[:240], refs, "APPROVAL REQUIRED" in text


async def chat(ctx: Ctx, pid: str, message: str, sse) -> AsyncIterator[str]:
    project = await orch.get_project(ctx, pid)
    workdir = ctx.data_dir / "agent_sessions" / pid
    workdir.mkdir(parents=True, exist_ok=True)
    mcp_cfg = workdir / "mcp.json"
    mcp_cfg.write_text(json.dumps({"mcpServers": {"cortex": {"type": "http", "url": MCP_URL}}}))
    cmd = ["claude", "-p", message, "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(mcp_cfg), "--strict-mcp-config",
           "--allowedTools", "mcp__cortex__*", "--max-turns", str(MAX_TURNS),
           "--append-system-prompt", SYSTEM.format(pid=pid, name=project.name, goal=project.spec.goal)]
    if ctx.settings.claude_code_model:
        cmd += ["--model", ctx.settings.claude_code_model]
    if sid := _load_session(ctx, pid):
        cmd += ["--resume", sid]
    yield sse("status", {"state": "thinking", "engine": "claude-code"})
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(workdir), stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024)
    pending: dict[str, str] = {}  # tool_use id -> tool name
    result: dict[str, Any] = {}
    try:
        assert proc.stdout
        async for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "system" and ev.get("subtype") == "init":
                _save_session(ctx, pid, ev["session_id"])
                yield sse("status", {"state": "thinking", "engine": "claude-code", "model": ev.get("model")})
            elif t == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        yield sse("text", {"delta": block["text"]})
                    elif block.get("type") == "tool_use":
                        if not block.get("name", "").startswith("mcp__cortex__"):
                            continue  # Claude Code internals (e.g. ToolSearch) are not shown in the chat
                        name = _short_tool(block.get("name", ""))
                        pending[block.get("id", "")] = name
                        yield sse("tool_call", {"tool": name, "args": block.get("input", {})})
                        yield sse("status", {"state": "running", "tool": name})
            elif t == "user":
                for block in ev.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        if block.get("tool_use_id", "") not in pending:
                            continue
                        name = pending.pop(block.get("tool_use_id", ""))
                        summary, refs, needs_approval = _summarize_result(block.get("content"))
                        yield sse("tool_result", {"tool": name, "summary": summary, "ref_ids": refs,
                                                  "ok": not block.get("is_error")})
                        if needs_approval:
                            yield sse("approval_required", {"kind": "training", "ref_id": pid, "summary": summary,
                                                            "args": {}})
                yield sse("status", {"state": "thinking"})
            elif t == "result":
                result = ev
                if ev.get("session_id"):
                    _save_session(ctx, pid, ev["session_id"])
        await proc.wait()
        if proc.returncode and not result:
            err = (await proc.stderr.read()).decode(errors="replace")[-400:] if proc.stderr else ""
            yield sse("text", {"delta": f"Claude Code exited with an error: {err or proc.returncode}"})
    finally:
        if proc.returncode is None:
            proc.terminate()
    yield sse("done", {"engine": "claude-code", "turns": result.get("num_turns"),
                       "cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms"),
                       "is_error": result.get("is_error")})


def reset(ctx: Ctx, pid: str) -> None:
    SESSIONS.pop(pid, None)
    f = _session_file(ctx, pid)
    if f.exists():
        f.unlink()
