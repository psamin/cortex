"""Infrastructure choices for a project: each option with live availability and, when unavailable,
the reason in plain words. The studio's Setup panel and the agent read the same answer."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..context import Ctx
from ..hardware import setup as hw
from ..models import GPU_CHOICES, InfraChoice, Project
from . import agent_collection_service as acs
from . import claude_code_agent as cc

_REACH: dict[str, tuple[bool, float]] = {}  # host -> (reachable, checked_at)


def opt(id: str, label: str, available: bool, reason: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"id": id, "label": label, "available": available, "reason": None if available else reason, **kw}


async def futurama_reachable(host: str, timeout: float = 2.0) -> bool:
    """Can this machine open the cluster's SSH port right now (GT VPN up)? Cached for 60 s."""
    hit = _REACH.get(host)
    if hit and time.time() - hit[1] < 60:
        return hit[0]
    name = host.split("@")[-1]
    try:  # resolve an ~/.ssh/config alias such as "futurama"
        p = await asyncio.create_subprocess_exec("ssh", "-G", host, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), 3)
        name = next((ln.split()[1] for ln in out.decode().splitlines() if ln.startswith("hostname ")), name)
        _, w = await asyncio.wait_for(asyncio.open_connection(name, 22), timeout)
        w.close()
        ok = True
    except (OSError, TimeoutError, StopIteration):
        ok = False
    _REACH[host] = (ok, time.time())
    return ok


async def options(ctx: Ctx, project: Project) -> dict[str, Any]:
    s = ctx.settings
    cc_ok = cc.available() is None and await cc.mcp_up()
    route = acs.resolve_planner(ctx)
    problem = (await acs.openrouter_balance_problem(ctx) or await acs.selfhosted_reachable(ctx)) if route else None
    host = s.futurama_ssh_host
    reach = await futurama_reachable(host) if host else False
    hwst = hw.status(ctx.data_dir, s.cortex_enable_hardware)
    return {
        "choice": project.infra.model_dump(),
        "planner": [
            opt("auto", "Automatic (your Claude Code when it's running, else the model route)", True),
            opt("claude_code", "Claude Code (your agent)", cc_ok,
                cc.available() or "the Cortex MCP server (http://localhost:3000/mcp) isn't running"),
            opt("model", route["label"] if route else "Model route", bool(route) and not problem,
                problem or "no model is configured (self-hosted endpoint, OpenRouter or InsForge key)"),
        ],
        "training_backend": [
            opt("auto", "Automatic (Futurama when configured, else run it yourself)", True),
            opt("futurama_slurm", "Futurama GPU cluster (GT, Slurm)", bool(host) and reach,
                "FUTURAMA_SSH_HOST is not set in .env" if not host else "can't reach Futurama: is the GT VPN on?",
                partition=s.futurama_partition),
            opt("manual", "Run lerobot-train yourself (Cortex prints the command)", True),
        ],
        "gpus": [opt(k, v, True) for k, v in GPU_CHOICES.items()],
        "walltime_min": {"min": 15, "max": 24 * 60, "note": "Futurama's general partition allows up to 1 day"},
        "evaluation": [
            opt("local_mujoco", "MuJoCo on this machine (parallel conditions)", True),
            opt("futurama", "MuJoCo on Futurama", False, "not built yet; see issue #17"),
        ],
        "robot": [
            opt("sim", "Simulated SO-101 (MuJoCo)", True),
            opt("real_so101", "Real SO-101 arm (Robot page)", bool(hwst["ready_for_motion"]),
                "; ".join(hwst["missing"]) or None),
        ],
    }


async def set_choice(ctx: Ctx, project: Project, choice: InfraChoice, actor: str = "gui") -> Project:
    project.infra = choice
    await ctx.repo.save("projects", project, status="active")
    await ctx.audit(actor, "set_infra", project.id, choice.model_dump(), "ok", "infrastructure choices saved")
    return project
