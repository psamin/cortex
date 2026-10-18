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
import contextlib
import json
import shutil
from collections.abc import AsyncIterator
from typing import Any

from ..context import Ctx
from . import agent_orchestrator as orch
from . import autonomy
from .narration import thought_line, todo_line, tool_line

MCP_URL = "http://localhost:3000/mcp"
MAX_TURNS = 40

SYSTEM = """You are the agent inside Cortex, a robot-learning studio. The user talks to you in the Cortex app;
you act through the cortex MCP tools only. Current project: {pid} ("{name}": {goal}). Always pass
project_id="{pid}" to tools that take one.
Style: act, don't ask. Call tools to make progress, then reply in 2-4 short plain sentences about what
happened and what the evidence shows. No JSON, no ids unless the user asks, no markdown tables.
Rules:
- You are already in the user's own Cortex chat and they are watching it: do NOT call cortex_register_agent, never
  post a watch link or session URL, and don't narrate setup (connecting, registering, reading the project, intake
  status). Talk only about the task's progress.
- Only report numbers that tools returned. Simulator results are measured; VLM/agent diagnoses are hypotheses.
{mode_rules}
- Default path: for a plain cube pick-and-place (a cube into the box or bowl on the standard table), do NOT run
  intake and do NOT build a custom scene. Go straight to cortex_start_agent_collection on this project's default
  scene, then cortex_train_from_agent_demonstrations, then ask the user to approve the training.
- Work in THIS project's sim scene; never create another project. Only when the user asks for different objects,
  a different layout or a different task, rebuild this project's scene: cortex_start_intake with project_id="{pid}" and ask_human=false,
  cortex_update_intake with the suggested patch for anything the user did not say, then cortex_build_simulation.
- For work with several steps, keep a short TodoWrite checklist (at most 6 items, plain words) and update it as
  steps finish: the user sees it live.
- Long jobs (collection, analysis, datasets, evaluation) return a job id. Start the job, then END YOUR TURN
  with one sentence saying it is running. Do NOT poll in a loop: the app shows a live progress card and posts
  the result in the chat when it finishes. Check progress only if the user asks.
- Scripted-oracle demos read simulator state; say so when you use them.
- If a tool call fails, fix the arguments and retry or take another route; never mention the failure, error text
  or failed calls to the user: report only what succeeded."""


START_TIMEOUT_S = 60  # spawn -> Claude Code's init event
IDLE_TIMEOUT_S = 600  # silence between events (a slow model reply or a long Cortex tool call is fine; a hang is not)


class Unavailable(RuntimeError):
    """Claude Code failed before doing anything, so another engine can answer this turn instead."""


def available() -> str | None:
    """None if Claude Code can run here, else why not (in words for the studio)."""
    if not shutil.which("claude"):
        return "Claude Code not found"
    return None


def _kill(proc) -> None:
    with contextlib.suppress(ProcessLookupError):  # it may have exited a moment ago
        proc.kill()


def _failure(text: str) -> str:
    """Claude Code's own error line, in words for the studio."""
    if "not logged in" in text.lower() or "/login" in text:
        return "Claude Code is not logged in (run `claude` in a terminal and log in)"
    return f"Claude Code failed: {text.strip()[:200]}" if text.strip() else "Claude Code failed"


async def mcp_up() -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.post(MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                             headers={"accept": "application/json, text/event-stream"})
        return r.status_code < 500
    except Exception:
        return False


def _short_tool(name: str) -> str:
    # Claude Code names MCP tools mcp__<server>__<tool>; ours are cortex.<x> -> cortex_<x>
    return name.split("__")[-1].removeprefix("cortex_")


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return content if isinstance(content, str) else ""


def _approval_detail(content: Any) -> str:
    """The backend's reason from an approval-required result: structuredContent `detail`,
    or the "Backend: ..." line of the text form (mcp/src/backend.ts approvalRequired)."""
    text = _content_text(content)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("detail"):
            return str(data["detail"])
    except ValueError:
        pass
    return next((ln.strip().removeprefix("Backend:").strip() for ln in text.splitlines()
                 if ln.strip().startswith("Backend:")), "")


def _result_events(name: str, block: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """SSE events for one Cortex tool_result. Only a refused training launch becomes an approval
    card (the studio approves that job by id); any other refusal, e.g. the planner budget, is a
    failed step that carries the backend's reason (the studio offers a budget raise for those)."""
    content = block.get("content")
    summary, refs, needs_approval = _summarize_result(content)
    result = {"tool": name, "summary": summary, "ref_ids": refs, "ok": not block.get("is_error")}
    if not needs_approval:
        return [("tool_result", result)]
    tj = next((r for r in refs if r.startswith("tj_")), None)
    if tj:
        return [("tool_result", result), ("approval_required", {
            "kind": "training", "ref_id": tj, "training_job_id": tj, "summary": summary, "args": {}})]
    detail = _approval_detail(content)
    return [("tool_result", {**result, "summary": (detail or summary)[:240], "ok": False})]


def _summarize_result(content: Any) -> tuple[str, list[str], bool]:
    """(one-line summary, ref ids, approval_required) from a tool_result content block.
    The Cortex MCP server writes "<summary>\nref_ids: a, b" as its text content."""
    text = _content_text(content)
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


async def chat(ctx: Ctx, pid: str, message: str, sse, tid: str | None = None) -> AsyncIterator[str]:
    from . import chat_store

    project = await orch.get_project(ctx, pid)
    workdir = ctx.data_dir / "agent_sessions" / pid
    workdir.mkdir(parents=True, exist_ok=True)
    mcp_cfg = workdir / "mcp.json"
    # The chat's project rides on the MCP connection (X-Cortex-Project): calls that leave out project_id land in this
    # project, so the studio agent's work shows in this chat's Agent log instead of a project it made by accident.
    mcp_cfg.write_text(json.dumps({"mcpServers": {"cortex": {"type": "http", "url": MCP_URL,
                                                             "headers": {"X-Cortex-Project": pid}}}}))
    cmd = [shutil.which("claude") or "claude", "-p", message, "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(mcp_cfg), "--strict-mcp-config",
           "--allowedTools", "mcp__cortex__*,TodoWrite", "--max-turns", str(MAX_TURNS),
           "--append-system-prompt", SYSTEM.format(pid=pid, name=project.name, goal=project.spec.goal,
                                                     mode_rules=autonomy.mode_rules(project))]
    if ctx.settings.claude_code_model:
        cmd += ["--model", ctx.settings.claude_code_model]
    thread = await chat_store.current(ctx, pid, tid)  # each chat thread is its own Claude Code session
    if sid := thread.get("session_id"):
        cmd += ["--resume", sid]
    yield sse("status", {"state": "starting", "engine": "claude-code", "text": "Starting Claude Code…"})
    try:  # stdin closed: `claude -p` reads a piped stdin to its end, and the backend's may never end
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(workdir), stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    limit=16 * 1024 * 1024)
    except OSError as e:
        raise Unavailable(f"Claude Code could not start ({e.strerror or e})") from e
    err_tail = bytearray()

    async def drain_stderr() -> None:  # read as it comes: a full stderr pipe would stall Claude Code mid-turn
        async for chunk in proc.stderr:
            err_tail.extend(chunk)
            del err_tail[:-2000]

    drain = asyncio.create_task(drain_stderr()) if proc.stderr else None
    pending: dict[str, str] = {}  # tool_use id -> tool name
    result: dict[str, Any] = {}
    started = acted = False  # init seen; a Cortex tool call or an answer shown (then there is no falling back)
    synthetic = ""  # Claude Code's own error lines ("Not logged in · Please run /login"), not the model's words
    lines = proc.stdout.__aiter__()
    try:
        while True:
            limit = IDLE_TIMEOUT_S if started else START_TIMEOUT_S
            try:
                raw = await asyncio.wait_for(anext(lines), limit)
            except StopAsyncIteration:
                break
            except TimeoutError:
                if not started:
                    raise Unavailable(f"Claude Code did not start within {limit:g} s") from None
                if not acted:
                    raise Unavailable(f"Claude Code stopped responding for {limit / 60:g} min") from None
                yield sse("text", {"delta": f"\n\nClaude Code stopped responding for {limit / 60:g} min, so I "
                                            "stopped it. What it did so far is above."})
                result = {"is_error": True}
                _kill(proc)
                break
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "system" and ev.get("subtype") == "init":
                started = True
                await chat_store.set_session(ctx, pid, thread["id"], ev["session_id"])
                yield sse("status", {"state": "thinking", "engine": "claude-code", "model": ev.get("model"),
                                     "text": "Planning…"})
            elif t == "assistant" and ev.get("message", {}).get("model") == "<synthetic>":
                synthetic += " ".join(b.get("text", "") for b in ev["message"].get("content", []))
            elif t == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        acted = True
                        yield sse("text", {"delta": block["text"]})
                    elif block.get("type") == "thinking" and (line := thought_line(block.get("thinking", ""))):
                        yield sse("status", {"state": "thinking", "text": line})  # live only, never stored
                    elif block.get("type") == "tool_use":
                        if block.get("name") == "TodoWrite":  # Claude's own checklist: shown live as a task list
                            todos = (block.get("input") or {}).get("todos") or []
                            yield sse("todos", {"items": [{"text": t.get("content", ""), "status": t.get("status", "pending"),
                                                           "active": t.get("activeForm")} for t in todos]})
                            if line := todo_line(todos):
                                yield sse("status", {"state": "thinking", "text": line})
                            continue
                        if not block.get("name", "").startswith("mcp__cortex__"):
                            if block.get("name") == "ToolSearch":  # Claude Code loading the Cortex tools
                                yield sse("status", {"state": "thinking", "text": "Loading the Cortex tools…"})
                            continue  # Claude Code internals are not shown in the chat
                        name = _short_tool(block.get("name", ""))
                        pending[block.get("id", "")] = name
                        acted = True
                        yield sse("tool_call", {"tool": name, "args": block.get("input", {})})
                        yield sse("status", {"state": "running", "tool": name,
                                             "text": tool_line(name, block.get("input"))})
            elif t == "user":
                for block in ev.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        if block.get("tool_use_id", "") not in pending:
                            continue
                        name = pending.pop(block.get("tool_use_id", ""))
                        for event, data in _result_events(name, block):
                            yield sse(event, data)
                yield sse("status", {"state": "thinking", "text": "Deciding the next step…"})
            elif t == "result":
                result = ev
                if ev.get("session_id"):
                    await chat_store.set_session(ctx, pid, thread["id"], ev["session_id"])
        try:
            await asyncio.wait_for(proc.wait(), 15)
        except TimeoutError:
            _kill(proc)
        if drain:
            await asyncio.wait([drain], timeout=2)
        err = err_tail.decode(errors="replace").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        if not acted and (result.get("is_error") or (proc.returncode and not result)):
            raise Unavailable(_failure(synthetic or str(result.get("result") or "") or err[0]))
        if proc.returncode and not result:
            yield sse("text", {"delta": f"Claude Code exited with an error: {err[0][:300]}"})
    finally:
        if proc.returncode is None:
            _kill(proc)
        if drain:
            drain.cancel()
    yield sse("done", {"engine": "claude-code", "turns": result.get("num_turns"),
                       "cost_usd": result.get("total_cost_usd"), "duration_ms": result.get("duration_ms"),
                       "is_error": result.get("is_error")})
