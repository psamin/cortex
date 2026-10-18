"""Live GPU inference for trained policies, from the studio (issue #157).

A Futurama GPU job (infra/futurama/serve_policy.sbatch) loads the newest checkpoint per task and
streams the policy running on held-out episodes. The cluster firewall blocks the login node from
reaching compute-node ports, so the tunnel goes through the GPU node itself:

    ssh -N -J <login> -L 8765:127.0.0.1:<port> <user>@<node>

`start` submits the job, waits for its READY line and opens that tunnel; `status` reports what the
server serves; the API proxies the MJPEG stream so the studio shows it in place.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ..context import Ctx

log = logging.getLogger(__name__)
STATE: dict[str, Any] = {"job": None, "tunnel": None, "phase": "stopped", "error": None}


def url(ctx: Ctx) -> str:
    return ctx.settings.live_policy_url.rstrip("/")


async def _sh(*args: str, timeout: float = 60) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await asyncio.wait_for(p.communicate(), timeout)
    return p.returncode or 0, out.decode(errors="replace")


async def status(ctx: Ctx) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            info = (await c.get(f"{url(ctx)}/policies.json")).json()
        STATE["phase"] = "ready"
        return {"available": True, "phase": "ready", **info}
    except (httpx.HTTPError, ValueError):
        reason = STATE["error"] or {
            "starting": "The GPU job is starting (installing and preparing held-out episodes, ~5–10 min).",
            "stopped": "No live GPU server is running. Start one to run the trained policies on a GPU.",
        }.get(STATE["phase"], "The live GPU server isn't reachable.")
        return {"available": False, "phase": STATE["phase"], "reason": reason, "job": STATE["job"]}


async def start(ctx: Ctx) -> dict[str, Any]:
    s = ctx.settings
    if not s.futurama_ssh_host:
        raise ValueError("FUTURAMA_SSH_HOST is not set in .env, so Cortex can't start a GPU job.")
    if (await status(ctx))["available"] or STATE["phase"] == "starting":
        return await status(ctx)
    d = s.futurama_policy_dir
    rc, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.futurama_ssh_host,
                        f"cd {d} && rm -f live/server.json && sbatch --parsable -J cortex-serve "
                        f"serve_policy.sbatch $HOME/{d.removeprefix('~/')}")
    job = next((ln.strip() for ln in out.splitlines() if ln.strip().isdigit()), None)
    if rc != 0 or not job:
        raise RuntimeError(f"sbatch failed: {out[-300:]}. Is the GT VPN on?")
    STATE.update(job=job, phase="starting", error=None)
    asyncio.get_running_loop().create_task(_connect_when_ready(ctx, job))
    return await status(ctx)


async def _connect_when_ready(ctx: Ctx, job: str) -> None:
    s, d = ctx.settings, ctx.settings.futurama_policy_dir
    for _ in range(80):  # up to ~20 min: queue + install + dataset prep
        await asyncio.sleep(15)
        rc, out = await _sh("ssh", "-o", "BatchMode=yes", s.futurama_ssh_host,
                            f"grep -c 'READY port' {d}/slurm-cortex-serve-{job}.out 2>/dev/null; "
                            f"cat {d}/live/server.json 2>/dev/null; squeue -j {job} -h -o %T")
        lines = [ln for ln in out.splitlines() if ln.strip() and not ln.lstrip().startswith("*")]
        if not any(ln.strip() in ("RUNNING", "PENDING", "CONFIGURING") for ln in lines):
            STATE.update(phase="stopped", error="The GPU job ended before the server was ready.")
            return
        info = next((json.loads(ln) for ln in lines if ln.startswith("{")), None)
        if info and any(ln.strip() == "1" for ln in lines):
            await open_tunnel(ctx, info["node"], int(info["port"]))
            return
    STATE.update(phase="stopped", error="The GPU server didn't become ready within 20 minutes.")


async def open_tunnel(ctx: Ctx, node: str, port: int) -> None:
    s = ctx.settings
    close_tunnel()
    local = int(url(ctx).rsplit(":", 1)[-1])
    user = await _remote_user(s.futurama_ssh_host)
    target = f"{user}@{node}" if user else node
    STATE["tunnel"] = await asyncio.create_subprocess_exec(
        "ssh", "-N", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
        "-J", s.futurama_ssh_host, "-L", f"{local}:127.0.0.1:{port}", f"{target}.{s.futurama_node_domain}"
        if s.futurama_node_domain else target, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    STATE["phase"] = "ready"


async def _remote_user(host: str) -> str | None:
    """The cluster username: from user@host, else from the ssh config (FUTURAMA_SSH_HOST is often an alias,
    whose User does not apply to the compute node's own hostname)."""
    if "@" in host:
        return host.split("@")[0]
    _, out = await _sh("ssh", "-G", host, timeout=10)
    return next((ln.split()[1] for ln in out.splitlines() if ln.startswith("user ")), None)


def close_tunnel() -> None:
    t = STATE.get("tunnel")
    if t is not None and t.returncode is None:
        t.terminate()
    STATE["tunnel"] = None


async def stop(ctx: Ctx) -> dict[str, Any]:
    close_tunnel()
    if STATE["job"] and ctx.settings.futurama_ssh_host:
        await _sh("ssh", "-o", "BatchMode=yes", ctx.settings.futurama_ssh_host, f"scancel {STATE['job']}")
    STATE.update(job=None, phase="stopped", error=None)
    return await status(ctx)
