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
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..context import Ctx

log = logging.getLogger(__name__)
COOLDOWN_S = 600  # after a failed request, wait before submitting again (#221: protect cluster access)
POLL_S = 10  # how often a starting job is checked
SSH_TIMEOUT_S = 60  # an ssh that hasn't finished by then is killed and counts as unreachable
START_DEADLINE_S = 20 * 60  # queue + install + dataset prep
TUNNEL_WAIT_S = 30  # two ssh handshakes (login node, then the GPU node) and the server's first answer
UNREACHABLE = 255  # ssh's exit status when it can't connect (also given to a timed-out command)
ENDED = {"COMPLETED", "COMPLETING", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
         "BOOT_FAIL", "DEADLINE"}
_CONNECT: dict[str, asyncio.Task | None] = {"task": None}  # the poller of the job being started
_STREAM: dict[str, asyncio.Task | None] = {"pump": None}  # the stream playing (the server plays one at a time)
STATE: dict[str, Any] = {"job": None, "tunnel": None, "phase": "stopped", "step": None, "detail": None, "failed_at": 0.0,
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


async def _sh(*args: str, timeout: float | None = None) -> tuple[int, str]:
    """(exit status, stdout + stderr). A command still running after `timeout` is killed and reported as
    unreachable, so a hung ssh never leaves the studio waiting."""
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    timeout = timeout or SSH_TIMEOUT_S
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        try:
            p.kill()
        except ProcessLookupError:
            pass
        await p.wait()
        return UNREACHABLE, f"no answer within {timeout:.0f} s"
    return p.returncode or 0, out.decode(errors="replace")


def _clean(out: str) -> str:
    """ssh output without the login node's banner (lines of asterisks) or blank lines."""
    return "\n".join(ln for ln in out.splitlines() if ln.strip() and not ln.lstrip().startswith("*")).strip()


def _unreachable(out: str) -> str:
    return f"Can't reach Futurama ({_clean(out)[-160:] or 'no answer'}). Is the GT VPN on?"


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
            "Not connected to a GPU server. Click Request a GPU to attach to one running on Futurama, or "
            "to start one." if STATE["phase"] == "stopped"
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
    wait = COOLDOWN_S - (time.time() - STATE["failed_at"])
    if wait > 0:
        raise RuntimeError(f"The last GPU request failed, so Cortex waits {int(wait // 60) + 1} more min before "
                           "asking the cluster again (this protects your Futurama access).")
    STATE.update(phase="starting", error=None, job=None)
    _step("request")
    d = s.futurama_policy_dir
    rc, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.futurama_ssh_host,
                        f"cd {d} && j=$(squeue -u $USER -n cortex-serve -h -o %i | head -1); "
                        f"if [ -n \"$j\" ]; then echo ATTACH $j; else rm -f live/server.json && "
                        f"sbatch --parsable -J cortex-serve --gres={s.live_policy_gres} "
                        f"serve_policy.sbatch $HOME/{d.removeprefix('~/')}; fi")
    out = _clean(out)
    attach = next((ln.split()[1] for ln in out.splitlines() if ln.startswith("ATTACH ")), None)
    job = attach or next((ln.strip() for ln in out.splitlines() if ln.strip().isdigit()), None)
    if rc != 0 or not job:
        if rc == UNREACHABLE:  # the cluster was never asked (VPN off, ssh hung): nothing to cool down from
            STATE.update(phase="stopped", step=None, error=_unreachable(out))
        else:
            STATE.update(phase="stopped", step=None, error=f"Couldn't submit the GPU job: {out[-200:]}",
                         failed_at=time.time())
        raise RuntimeError(STATE["error"])
    STATE["job"] = job
    _step("queued", f"Found GPU job {job} already allocated; connecting" if attach
          else f"Submitted Slurm job {job}; waiting for a free GPU")
    _cancel_connect()
    _CONNECT["task"] = asyncio.get_running_loop().create_task(_connect_when_ready(ctx, job))
    return await status(ctx)


def _cancel_connect() -> None:
    t = _CONNECT["task"]
    if t is not None and not t.done() and t is not asyncio.current_task():
        t.cancel()
    _CONNECT["task"] = None


async def _poll(ctx: Ctx, job: str) -> tuple[int, str, str, dict[str, Any] | None]:
    """(ssh exit status, Slurm state and reason, latest progress line, server.json) for the serve job."""
    s, d = ctx.settings, ctx.settings.futurama_policy_dir
    rc, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", s.futurama_ssh_host,
                       f"squeue -j {job} -h -o '%T %R'; echo @@; "
                       f"grep -E '^\\[live\\]|READY port' {d}/slurm-cortex-serve-{job}.out 2>/dev/null | tail -1; "
                       f"echo @@; cat {d}/live/server.json 2>/dev/null")
    if rc == UNREACHABLE:
        return rc, _clean(out), "", None
    state, progress, info = ([p.strip() for p in _clean(out).split("@@")] + ["", "", ""])[:3]
    try:
        server = json.loads(info) if info else None
    except ValueError:
        server = None
    return rc, state, progress, server


def job_ended(state: str) -> bool:
    """squeue no longer has the job (once a job leaves the queue, `squeue -j` says "Invalid job id"), or shows
    it finishing."""
    return not state or "Invalid job id" in state or state.split()[0] in ENDED


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
    """Narrate the job's progress until its server is up, then open the tunnel. Ends quietly when the job is
    stopped or replaced (STATE["job"] changes)."""
    d = ctx.settings.futurama_policy_dir
    deadline = time.monotonic() + START_DEADLINE_S
    while time.monotonic() < deadline:
        try:
            rc, state, progress, server = await _poll(ctx, job)
        except OSError as e:
            rc, state, progress, server = UNREACHABLE, str(e), "", None
        if STATE["job"] != job:
            return
        if rc == UNREACHABLE or state.startswith("slurm_load_jobs error") and "Invalid job id" not in state:
            # the VPN dropped or Slurm isn't answering: say so, and keep trying (the job may be fine)
            STATE["detail"] = (f"Can't reach Futurama right now ({state[-120:] or 'no answer'}); still trying. "
                               f"Is the GT VPN on? (GPU job {job})")
        elif job_ended(state):
            last = progress.removeprefix("[live] ")
            STATE.update(phase="stopped", job=None, failed_at=time.time(),
                         error=f"GPU job {job} ended before the server was ready"
                               f"{f' (last step: {last})' if last else ''}. Its log on Futurama: "
                               f"{d}/slurm-cortex-serve-{job}.out")
            return
        else:
            _step(*narrate(job, state, progress))
            if STATE["step"] == "connect" and server and str(server.get("job")) == str(job):
                await open_tunnel(ctx, server["node"], int(server["port"]))
                return
        await asyncio.sleep(POLL_S)
    if STATE["job"] == job:  # the job is still there: Try again attaches to it, Stop cancels it
        STATE.update(phase="stopped", error=f"GPU job {job} isn't ready after {START_DEADLINE_S // 60} minutes "
                                            f"({STATE['detail']}). Try again to keep waiting.")


async def open_tunnel(ctx: Ctx, node: str, port: int) -> None:
    """Open the tunnel and wait for the server to answer through it: until then the studio keeps saying
    "Connecting". A tunnel that fails (port in use, node unreachable) is reported, not left as a dead end."""
    s = ctx.settings
    close_tunnel()
    local = int(url(ctx).rsplit(":", 1)[-1])
    user = await _remote_user(s.futurama_ssh_host)
    target = f"{user}@{node}" if user else node
    t = STATE["tunnel"] = await asyncio.create_subprocess_exec(
        "ssh", "-N", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
        "-J", s.futurama_ssh_host, "-L", f"{local}:127.0.0.1:{port}", f"{target}.{s.futurama_node_domain}"
        if s.futurama_node_domain else target, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    deadline = time.monotonic() + TUNNEL_WAIT_S
    while time.monotonic() < deadline:
        try:
            await asyncio.wait_for(t.wait(), 0.5)
        except TimeoutError:
            pass
        if t.returncode is not None:  # ssh gave up: say why (its stderr, without the login banner)
            why = _clean((await t.stderr.read()).decode(errors="replace")) if t.stderr else ""
            STATE.update(phase="stopped", tunnel=None,
                         error=f"Couldn't open the tunnel to {node} ({why[-200:] or f'ssh exited {t.returncode}'}).")
            return
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                (await c.get(f"{url(ctx)}/policies.json")).raise_for_status()
        except httpx.HTTPError:
            continue
        STATE["phase"] = "ready"
        _step("ready")
        return
    close_tunnel()
    STATE.update(phase="stopped", error=f"The tunnel to {node} is open but the GPU server didn't answer within "
                                        f"{TUNNEL_WAIT_S} s. Try again to reconnect.")


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
        try:
            t.terminate()
        except ProcessLookupError:  # it exited on its own
            pass
    STATE["tunnel"] = None


async def relay(ctx: Ctx, path: str, rec: Any) -> AsyncIterator[bytes]:
    """Proxy one MJPEG stream from the GPU server (`path`, e.g. /sim/jenga/7), feeding `rec` as it passes.

    The server plays one stream at a time and makes the next one wait. So a new stream (a second viewer, or
    "Run on a new layout" while a run plays) ends the one playing, instead of showing a blank image until it
    finishes; the ended run is saved as partial."""
    q: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=None)) as c:
                async with c.stream("GET", f"{url(ctx)}{path}") as r:
                    async for chunk in r.aiter_raw():
                        q.put_nowait(chunk)
            rec.finished = True
        except Exception as e:  # noqa: BLE001 - a dropped tunnel or server: the stream just ends
            log.info("live stream %s ended: %r", path, e)
        finally:
            q.put_nowait(None)

    end_stream()
    task = _STREAM["pump"] = asyncio.get_running_loop().create_task(pump())
    try:
        while (chunk := await q.get()) is not None:
            rec.feed(chunk)
            yield chunk
    finally:
        task.cancel()
        if _STREAM["pump"] is task:
            _STREAM["pump"] = None


def end_stream() -> None:
    """End the stream playing, if any (its viewer sees the last frame; the run is saved as partial)."""
    t = _STREAM["pump"]
    if t is not None and not t.done():
        t.cancel()
    _STREAM["pump"] = None


async def stop(ctx: Ctx) -> dict[str, Any]:
    """Close the tunnel and cancel the serve job. After a backend restart the job id is unknown, so the job
    is cancelled by its name (cortex-serve)."""
    job = STATE["job"]
    _cancel_connect()
    end_stream()
    close_tunnel()
    STATE.update(job=None, phase="stopped", step=None, detail=None, error=None)
    if ctx.settings.futurama_ssh_host:
        rc, out = await _sh("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ctx.settings.futurama_ssh_host,
                            f"scancel {job}" if job else "scancel -u $USER -n cortex-serve")
        if rc != 0:
            STATE["error"] = (f"Couldn't cancel the GPU job on Futurama ({_clean(out)[-160:] or 'no answer'}). "
                              f"Cancel it there: scancel {job or '-u $USER -n cortex-serve'}")
    return await status(ctx)
