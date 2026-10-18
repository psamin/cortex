"""Failure diagnosis by the user's own Claude Code.

When the project's planner choice (#68) resolves to Claude Code, each failed rollout's keyframe
contact sheet (the same one vlm_diagnostics sends to a model route) is written to a work dir and
`claude -p` runs headless with ONLY the Read tool, no MCP servers, no shell and no edits. It answers
with the same fixed-taxonomy JSON, parsed by vlm_diagnostics.parse_hypotheses. Results are stored
as unverified hypotheses; a failed or timed-out run records an error, never invented hypotheses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..context import Ctx
from ..models import EpisodeDiagnosis, Project
from . import vlm_diagnostics as vlm

TIMEOUT_S = 180
CONCURRENCY = 2  # parallel `claude -p` runs per analysis
LABEL = "claude-code"

INSTRUCTIONS = """The contact sheet is the image file ./{image} in the current directory. Open it with the Read
tool, then answer with ONLY the JSON object described below (no prose, no code fence).

"""


def diagnoser(ctx: Ctx, project: Project) -> tuple[str | None, str]:
    """('claude_code' | 'model' | None, label or reason) from the project's planner choice:
    auto prefers Claude Code when the `claude` CLI is here, else the model route."""
    from . import claude_code_agent as cc

    want = project.infra.planner
    cc_ok = cc.available() is None and ctx.settings.agent_engine in ("auto", "claude_code")
    if want in ("auto", "claude_code") and cc_ok:
        return "claude_code", "Claude Code (your agent)"
    if want == "claude_code":
        return None, f"Claude Code was chosen but {cc.available() or 'the agent engine is pinned to the model'}"
    if ctx.model:
        return "model", ctx.vlm_source
    return None, "Claude Code is not available and no self-hosted model endpoint is configured"


async def diagnose_all(ctx: Ctx, project: Project, targets: list[tuple[str, list[np.ndarray], dict[str, Any]]],
                       workdir: Path) -> list[EpisodeDiagnosis]:
    workdir.mkdir(parents=True, exist_ok=True)
    model = ctx.settings.claude_code_model
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(i: int, ref: str, frames: list[np.ndarray], tel: dict[str, Any]) -> EpisodeDiagnosis:
        async with sem:
            return await diagnose(ref, frames, project.spec.goal, tel, workdir, f"target{i:02d}", model)

    return list(await asyncio.gather(*(one(i, *t) for i, t in enumerate(targets))))


async def diagnose(ref_id: str, frames: list[np.ndarray], task: str, telemetry: dict[str, Any], workdir: Path,
                   name: str, model: str | None = None) -> EpisodeDiagnosis:
    label = f"{LABEL} ({model})" if model else LABEL
    image = f"{name}.jpg"
    (workdir / image).write_bytes(vlm.contact_sheet(frames))
    prompt = INSTRUCTIONS.format(image=image) + vlm.PROMPT.format(
        task=task, n=len(frames), telemetry=json.dumps(telemetry),
        taxonomy="\n".join(f"- {k}: {v}" for k, v in vlm.TAXONOMY.items()))
    try:
        reply = await run_claude(prompt, workdir, model)
        return EpisodeDiagnosis(episode_id=ref_id, model=label, hypotheses=vlm.parse_hypotheses(reply))
    except Exception as e:  # recorded on the analysis, never silently dropped
        return EpisodeDiagnosis(episode_id=ref_id, model=label, hypotheses=[], error=f"{type(e).__name__}: {e}"[:300])


def command(prompt: str, model: str | None = None) -> list[str]:
    from .claude_code_agent import executable

    exe = executable() or "claude"  # full path so Windows finds claude.cmd
    cmd = [exe, "-p", prompt, "--output-format", "json", "--strict-mcp-config", "--allowedTools", "Read",
           "--max-turns", "4"]
    return cmd + (["--model", model] if model else [])


async def run_claude(prompt: str, workdir: Path, model: str | None = None, timeout_s: float | None = None) -> str:
    """The reply text of one headless `claude -p` run (its final `result`), within timeout_s (default TIMEOUT_S)."""
    timeout_s = timeout_s or TIMEOUT_S
    # stdin closed: `claude -p` reads a piped stdin to its end, and the backend's may never end
    proc = await asyncio.create_subprocess_exec(*command(prompt, model), cwd=str(workdir),
                                                stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Claude Code did not answer within {timeout_s:g} s") from None
    if proc.returncode:
        raise RuntimeError(f"Claude Code exited {proc.returncode}: {err.decode(errors='replace').strip()[-200:]}")
    data = json.loads(out)
    if data.get("is_error") or data.get("subtype", "success") != "success":
        raise RuntimeError(f"Claude Code stopped ({data.get('subtype')}): {str(data.get('result', ''))[:200]}")
    return str(data.get("result", ""))  # parse_hypotheses extracts the JSON object from the reply
