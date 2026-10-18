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
STATE: dict[str, Any] = {"job": None, "tunnel": None, "phase": "stopped", "step": None, "detail": None,
                         "error": None}
# What the studio shows while a GPU comes up, in order.
STEPS = {"request": "Requesting a GPU on Georgia Tech's Futurama cluster",
         "queued": "Waiting for a free GPU",
         "install": "Installing LeRobot on the GPU node",
         "prepare": "Preparing held-out episodes",
         "load": "Loading the policies onto the GPU",
         "connect": "Connecting to the GPU server",
         "ready": "Live"}


def url(ctx: Ctx) -> str:
    return ctx.settings.live_policy_url.rstrip("/")


async def _sh(*args: str, timeout: float = 60) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await asyncio.wait_for(p.communicate(), timeout)
    return p.returncode or 0, out.decode(errors="replace")


def _step(step: str, detail: str | None = None) -> None:
    STATE.update(step=step, detail=detail or STEPS[step])


async def status(ctx: Ctx) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            info = (await c.get(f"{url(ctx)}/policies.json")).json()
        STATE["phase"] = "ready"
        _step("ready")
        return {"available": True, "phase": "ready", "step": "ready", "steps": STEPS, **info}
    except (httpx.HTTPError, ValueError):
        reason = STATE["error"] or (STATE["detail"] if STATE["phase"] == "starting" else None) or (
            "No GPU is allocated. Opening this tab requests one on Futurama." if STATE["phase"] == "stopped"
            else "The live GPU server isn't reachable.")
        return {"available": False, "phase": STATE["phase"], "step": STATE["step"], "steps": STEPS,
                "reason": reason, "error": STATE["error"], "job": STATE["job"]}


async def start(ctx: Ctx) -> dict[str, Any]:
    """Attach to a GPU server that is already allocated, or request one through Slurm."""
    s = ctx.settings
    if not s.futurama_ssh_host:
        raise ValueError("FUTURAMA_SSH_HOST is not set in .env, so Cortex can't start a GPU job.")
    if (await status(ctx))["available"] or STATE["phase"] == "starting":
        return await status(ctx)
    STATE.update(phase="starting", error=None)
    _step("request")
    d = s.futurama_policy_dir
    rc, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.futurama_ssh_host,
                        f"cd {d} && j=$(squeue -u $USER -n cortex-serve -h -o %i | head -1); "
                        f"if [ -n \"$j\" ]; then echo ATTACH $j; else rm -f live/server.json && "
                        f"sbatch --parsable -J cortex-serve --gres={s.live_policy_gres} "
                        f"serve_policy.sbatch $HOME/{d.removeprefix('~/')}; fi")
    attach = next((ln.split()[1] for ln in out.splitlines() if ln.startswith("ATTACH ")), None)
    job = attach or next((ln.strip() for ln in out.splitlines() if ln.strip().isdigit()), None)
    if rc != 0 or not job:
        STATE.update(phase="stopped", error=f"Couldn't submit the GPU job: {out.strip()[-200:]}. Is the GT VPN on?")
        raise RuntimeError(STATE["error"])
    STATE["job"] = job
    _step("queued", f"Found GPU job {job} already allocated; connecting" if attach
          else f"Submitted Slurm job {job}; waiting for a free GPU")
    asyncio.get_running_loop().create_task(_connect_when_ready(ctx, job))
    return await status(ctx)


async def _poll(ctx: Ctx, job: str) -> tuple[str, str, dict[str, Any] | None]:
    """(Slurm state and reason, latest progress line, server.json) for the serve job."""
    s, d = ctx.settings, ctx.settings.futurama_policy_dir
    _, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.futurama_ssh_host,
                       f"squeue -j {job} -h -o '%T %R'; echo @@; "
                       f"grep -E '^\\[live\\]|READY port' {d}/slurm-cortex-serve-{job}.out 2>/dev/null | tail -1; "
                       f"echo @@; cat {d}/live/server.json 2>/dev/null")
    text = "\n".join(ln for ln in out.splitlines() if not ln.lstrip().startswith("*"))
    state, progress, info = ([p.strip() for p in text.split("@@")] + ["", "", ""])[:3]
    try:
        server = json.loads(info) if info else None
    except ValueError:
        server = None
    return state, progress, server


def narrate(job: str, state: str, progress: str) -> tuple[str, str]:
    """The step and the line the studio shows for this poll."""
    if state.startswith(("PENDING", "CONFIGURING")):
        why = state.split(" ", 1)[1].strip("()") if " " in state else ""
        return "queued", f"Waiting for a free GPU (Slurm job {job} queued{': ' + why if why else ''})"
    if "READY port" in progress:
        return "connect", STEPS["connect"]
    if progress.startswith("[live] "):
        line = progress.removeprefix("[live] ")
        step = "install" if line.startswith("Installing") else "prepare" if line.startswith("Preparing") else "load"
        return step, line
    return "install", f"GPU allocated (job {job}); starting up"


async def _connect_when_ready(ctx: Ctx, job: str) -> None:
    for _ in range(120):  # up to ~20 min: queue + install + dataset prep
        try:
            state, progress, server = await _poll(ctx, job)
        except (TimeoutError, OSError) as e:
            log.warning("live policy poll failed: %s", e)
            await asyncio.sleep(10)
            continue
        if not state:
            STATE.update(phase="stopped", error=f"GPU job {job} ended before the server was ready.")
            return
        _step(*narrate(job, state, progress))
        if STATE["step"] == "connect" and server and str(server.get("job")) == str(job):
            await open_tunnel(ctx, server["node"], int(server["port"]))
            return
        await asyncio.sleep(10)
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
    STATE.update(job=None, phase="stopped", step=None, detail=None, error=None)
    return await status(ctx)
