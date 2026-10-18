"""Autonomous mode: the studio chat runs the whole workflow without waiting on the human.

Every project has `autopilot.autonomy`, either "autonomous" (the default) or "manual".

Autonomous:
- When background work finishes after a studio chat turn (demos, a dataset, training, an evaluation),
  Cortex wakes the studio agent with the results, and it continues with the next step: collect, build
  the dataset, train, evaluate, improve. The chain ends when a turn ends with nothing running and no
  unseen results. It also ends after MAX_AUTO_TURNS wake-ups in a row, when the user stops it, or when
  the project switches to manual.
- Training the agent proposes is approved and started under the project's standing approval when that
  approval covers it. This is the rule Autopilot uses: auto-approve is on, the GPU is free unless paid jobs
  are allowed, and the job is at most max_gpu_hours_per_job.

Two gates hold in every mode (PRODUCT_SPEC section 5):
- spend outside the standing approval waits for a human click: paid GPUs, more GPU-hours, a used-up
  planner budget;
- real-robot motion always needs a human, and nothing here can start it.

Manual: the agent does one step per turn, and training waits for an approval click.

A step that is cut off (the backend restarted mid-turn, or the agent process crashed or timed out) is picked up
again on its own in autonomous mode, after a pause that grows each time. After MAX_RECOVERIES in a row the chain
pauses, and the studio shows a Continue button (agent_runs.json keeps this across restarts).

One backend runs the chain. After SIGTERM, uvicorn keeps the old process alive while a studio tab holds its event
stream, so at a restart the old backend can still be running next to the new one. The new one claims the data dir
(backend.owner) and takes the cut-off turn over; the old one then starts no turn and writes no state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from typing import Any

from ..context import Ctx
from ..models import JobStatus, Project, TrainingJob
from . import training_service
from .jobs import INTERRUPTED

log = logging.getLogger(__name__)
ACTOR = "autopilot"  # the standing approval (training_service.approve accepts it; the caller checks coverage)
APPROVER = "standing approval (autonomous mode)"
MAX_AUTO_TURNS = 24  # wake-ups in a row without a user message (a full round is about 4)
WAKE_DELAY_S = 1.5  # let sibling job lines land before waking the agent
DATASET_KINDS = ("dataset", "train_from_agent")  # a failure here ends the chain: no improvised training data
MAX_RECOVERIES = 3  # cut-off steps picked up again in a row; then the chain pauses for a Continue click
RECOVERY_BACKOFF_S = (5.0, 30.0, 120.0)

WAKE_PROMPT = ("[Cortex, autonomous mode] Background work finished: {reason}. Continue the workflow toward the goal "
               "on your own. If the target is met, or the next step needs the human (spend outside the standing "
               "approval, or real-robot motion), say so in one sentence and stop.")
RESUME_PROMPT = ("[Cortex, autonomous mode] Picking the workflow up again: {reason}. Look at what this project already "
                 "has (scene, demos, datasets, training, running jobs) and continue toward the goal from where it "
                 "stopped; don't redo finished work. If the target is met, or the next step needs the human (spend "
                 "outside the standing approval, or real-robot motion), say so in one sentence and stop.")

_LOCKS: dict[str, asyncio.Lock] = {}
_TASKS: dict[str, asyncio.Task] = {}  # wake-up turns in flight, per project
_TURNS: dict[str, asyncio.Task] = {}  # user turns in flight, per project (they outlive the studio's request)
_LIVE: dict[str, dict[str, Any]] = {}  # turns running in this process: {pid: {thread, auto, status}}
_STOPPED: set[str] = set()  # projects whose turn the user stopped: that cancel is not a cut-off
_AUTO_TURNS: dict[str, int] = {}  # consecutive wake-ups per project
_NOTIFIED: set[str] = set()  # training jobs already reported as needing the human
_STATES: dict[str, dict[str, Any]] = {}  # per data dir: {pid: {"thread": tid | None, "pending": [{text, at}]}}
_RUNS: dict[str, dict[str, Any]] = {}  # per data dir: {pid: {"turn": {thread, auto, at, agent_pid}, "attempts", "paused"}}


def lock(pid: str) -> asyncio.Lock:
    """One studio turn at a time per project (a wake-up waits for the user's turn, and the other way round)."""
    return _LOCKS.setdefault(pid, asyncio.Lock())


def is_autonomous(project: Project) -> bool:
    return project.autopilot.autonomy == "autonomous"


def standing_approval(project: Project, tj: TrainingJob) -> tuple[bool, str]:
    """(covered, why). The same rule Autopilot applies before it approves training on its own."""
    pol, est = project.autopilot, tj.estimate
    if tj.backend == "manual":
        return False, "no GPU backend is connected, so run its command bundle and register the checkpoint"
    if not pol.auto_approve_free_training:
        return False, "auto-approve is off in this project's approvals"
    if est.paid and not pol.allow_paid_jobs:
        return False, f"it runs on a paid GPU (~${est.est_cost_usd:.2f}) and paid jobs aren't allowed"
    if est.gpu_hours > pol.max_gpu_hours_per_job:
        return False, f"~{est.gpu_hours} GPU-hours is over the {pol.max_gpu_hours_per_job} GPU-hour cap"
    cost = f"${est.est_cost_usd:.2f}" if est.paid else "no charge"
    return True, f"~{est.gpu_hours} GPU-hours, {cost}"


CLUSTER_CLICK = "it would run on the shared Futurama cluster, and cluster jobs always wait for your click"


def cluster_needs_click(ctx: Ctx, tj: TrainingJob) -> bool:
    """Cluster training is never submitted on a standing approval unless CORTEX_AUTONOMY_CLUSTER is on (#221)."""
    return tj.backend == "futurama_slurm" and not ctx.settings.cortex_autonomy_cluster


def mode_rules(project: Project) -> str:
    """The studio agent's instructions for the project's mode (system-prompt lines)."""
    pol = project.autopilot
    if not is_autonomous(project):
        return ("- MANUAL MODE (the user's choice): do one stage per turn, then stop and suggest the next step.\n"
                "- Training on the GPU cluster needs the human's approval in the app: propose it, then tell the "
                "user to approve.")
    return (f"- AUTONOMOUS MODE (the user's choice): run the whole loop without waiting on the user. Don't ask "
            f"questions or offer options; pick sensible defaults and keep going: collect demos in sim, build the "
            f"dataset, train, evaluate, then improve the weakest condition. Stop when every condition reaches "
            f"{pol.target_success_rate:.0%} or after {pol.max_iterations} improvement rounds.\n"
            f"- Training inside the project's standing approval (at most {pol.max_gpu_hours_per_job} GPU-hours, "
            f"{'paid GPUs allowed' if pol.allow_paid_jobs else 'free GPUs only'}) is approved and started by Cortex "
            f"as soon as you propose it, so don't ask the user to approve it. Spend outside it, a used-up planner "
            f"budget, and any real-robot motion need the human: the app shows them a card, so say in one sentence "
            f"what needs them and stop.\n"
            f"- When you start a long job, end your turn: Cortex wakes you with the result and you continue.")


def notice(tone: str, text: str, ref_ids: list[str] | None = None) -> dict[str, Any]:
    """A chat row written by Cortex: tone "auto" (it acted for you) or "needs_you" (a gate)."""
    return {"kind": "notice", "tone": tone, "text": text, "ref_ids": ref_ids or []}


async def approve_if_covered(ctx: Ctx, pid: str, tj_id: str) -> dict[str, Any] | None:
    """Autonomous projects only: approve and start a proposed training job the standing approval covers.
    Returns the chat notice to show, or None when there is nothing to say."""
    from .agent_orchestrator import get_project

    project = await get_project(ctx, pid)
    tj = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
    if not is_autonomous(project) or tj is None or tj.status != JobStatus.proposed:
        return None
    covered, why = standing_approval(project, tj)
    if covered and cluster_needs_click(ctx, tj):
        covered, why = False, CLUSTER_CLICK
    if not covered:
        if tj.id in _NOTIFIED:
            return None
        _NOTIFIED.add(tj.id)
        return notice("needs_you", f"Training needs you: {why}. Approve it on the card to continue.", [tj.id])
    try:
        await training_service.approve(ctx, tj.id, APPROVER, ACTOR, None)
        tj = await training_service.launch(ctx, tj.id, ACTOR)
    except Exception as e:  # e.g. the cluster refused the submit: the human decides what next
        log.warning("standing approval launch of %s failed: %s", tj.id, e)
        return notice("needs_you", f"Training {tj.id} was approved but didn't start: {e}", [tj.id])
    return notice("auto", f"Approved {tj.policy_type.upper()} training under your standing approval ({why}) and "
                          f"started it on {'Futurama' if tj.backend == 'futurama_slurm' else tj.backend}.", [tj.id])


# ------------------------------------------------------------------ the wake-up chain

def _state(ctx: Ctx) -> dict[str, Any]:
    key = str(ctx.data_dir)
    if key not in _STATES:
        try:
            _STATES[key] = json.loads((ctx.data_dir / "autonomy.json").read_text())
        except (OSError, ValueError):
            _STATES[key] = {}
    return _STATES[key]


def _flush(ctx: Ctx) -> None:
    if superseded(ctx):  # its cache is stale: the newer backend's state stands
        return
    try:
        (ctx.data_dir / "autonomy.json").write_text(json.dumps(_state(ctx)))
    except OSError as e:  # the chain still works in this process
        log.warning("could not save autonomy state: %s", e)


def armed_thread(ctx: Ctx, pid: str) -> str | None:
    """The chat thread whose chain is waiting on background work, if any."""
    return (_state(ctx).get(pid) or {}).get("thread")


def disarm(ctx: Ctx, pid: str) -> None:
    if _state(ctx).pop(pid, None) is not None:
        _flush(ctx)


def _active(ctx: Ctx, pid: str) -> bool:
    return any(j.status in ("queued", "running") for j in ctx.jobs.list(pid))


def turn_started(ctx: Ctx, pid: str, auto: bool) -> None:
    """The agent is about to see the project as it is now, so results so far are no longer pending."""
    if not auto:
        _AUTO_TURNS.pop(pid, None)
    st = _state(ctx).get(pid)
    if st and st.get("pending"):
        st["pending"] = []
        _flush(ctx)


async def turn_ended(ctx: Ctx, pid: str, tid: str, last_result_at: float, failed: bool = False) -> None:
    """Arm the chain if work is still running or results landed after the agent's last tool result.
    `failed`: the turn stopped before it finished (the agent crashed, timed out or hit its turn limit), so it is
    picked up again (see recover). Otherwise end the chain, unless proposed training waits for a human click."""
    from . import autopilot
    from .agent_orchestrator import get_project

    project = await get_project(ctx, pid)
    if not is_autonomous(project) or pid in autopilot._running:  # Autopilot runs its own loop
        disarm(ctx, pid)
        return
    if failed:
        await recover(ctx, pid, tid, "the last step stopped before it finished")
        return
    if (r := _runs(ctx).get(pid)) and r.get("attempts"):  # a step finished: earlier retries no longer count
        r["attempts"] = 0
        _save_runs(ctx)
    st = _state(ctx).get(pid) or {}
    unseen = [p for p in st.get("pending", []) if p["at"] > last_result_at]
    if not unseen and not _active(ctx, pid):
        if await ctx.repo.list("training_jobs", TrainingJob, project_id=pid, status=JobStatus.proposed):
            # training waits for a click: stay armed, so the training job wakes the agent when it ends, wherever
            # it was approved (the chat card, the Training page, Autopilot)
            _state(ctx)[pid] = {"thread": tid, "pending": []}
            _flush(ctx)
            return
        disarm(ctx, pid)
        return
    _state(ctx)[pid] = {"thread": tid, "pending": unseen}
    _flush(ctx)
    if not _active(ctx, pid):
        _schedule_wake(ctx, pid)


async def job_done(ctx: Ctx, job, entry: dict[str, Any]) -> dict[str, Any] | None:
    """A background job finished (its chat line is `entry`). In an autonomous chain: start training the job
    proposed, if the standing approval covers it, and wake the agent once nothing is running.
    Returns a notice row for the chat, if any."""
    from . import autopilot
    from .agent_orchestrator import get_project

    pid = job.project_id
    if ctx.jobs.closing or pid in autopilot._running or superseded(ctx):
        return None
    if job.status == "cancelled":  # the user stopped work: stop the chain too
        disarm(ctx, pid)
        return None
    armed, in_turn = armed_thread(ctx, pid), lock(pid).locked()
    if not armed and not in_turn:
        return None  # not part of a studio chain (e.g. started from a manual page or by another agent)
    try:
        if not is_autonomous(await get_project(ctx, pid)):
            disarm(ctx, pid)
            return None
    except Exception:  # the project is gone
        disarm(ctx, pid)
        return None
    if job.status == "failed" and job.kind in DATASET_KINDS and job.error_detail != INTERRUPTED:
        # a woken agent would improvise a dataset (prj_e5e9c205d5f3 took another task's demos) and the standing
        # approval would start training on it: the human decides what the policy learns from. A restart is not
        # such a failure: the agent runs the same step again.
        disarm(ctx, pid)
        return notice("needs_you", f"Stopped the automatic steps: the dataset step failed ({job.error or 'no detail'})."
                                   " Nothing was trained. Tell me which demos to use, or collect more.")
    st = _state(ctx).setdefault(pid, {"thread": armed, "pending": []})
    st.setdefault("pending", []).append({"text": entry.get("text", ""), "at": time.time()})
    _flush(ctx)
    row = None
    tj = next((r for r in entry.get("ref_ids", []) if r.startswith("tj_")), None)
    if tj and job.status == "succeeded":
        row = await approve_if_covered(ctx, pid, tj)
    if armed and not in_turn and not _active(ctx, pid):
        _schedule_wake(ctx, pid)
    return row


def _schedule_wake(ctx: Ctx, pid: str, delay: float | None = None) -> None:
    t = _TASKS.get(pid)
    if t and not t.done() and t is not asyncio.current_task():  # an automatic turn may chain the next one
        return
    _TASKS[pid] = asyncio.get_running_loop().create_task(_wake(ctx, pid, WAKE_DELAY_S if delay is None else delay))


def relay(ctx: Ctx, pid: str, chunk: str, tid: str | None) -> None:
    """Publish one SSE chunk of a turn nobody streams (an automatic turn, or the studio left) on the project stream."""
    event, _, data = chunk.partition("\ndata: ")
    try:
        payload = json.loads(data)
    except ValueError:
        return
    ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": event.removeprefix("event: "),
                                  "data": payload})


async def _wake(ctx: Ctx, pid: str, delay: float) -> None:
    """Run one agent turn on the armed thread, published live on the project stream."""
    from . import agent
    from .agent_orchestrator import get_project

    await asyncio.sleep(delay)
    async with lock(pid):
        if superseded(ctx):  # a newer backend took the chain over during the wait
            return
        st = _state(ctx).get(pid) or {}
        tid, pending = st.get("thread"), st.get("pending") or []
        if not tid or not pending or _active(ctx, pid):
            return  # a user turn saw the results meanwhile, or more work started
        try:
            autonomous = is_autonomous(await get_project(ctx, pid))
        except Exception:  # the project is gone
            disarm(ctx, pid)
            return
        if not autonomous:  # switched to manual meanwhile
            disarm(ctx, pid)
            return
        if await agent.engine(ctx) == "rules":  # the rule-based planner can't plan a next step: say so, don't vanish
            await pause(ctx, pid, tid, "no agent can plan the next step here (start Claude Code's Cortex MCP server, "
                                       "or set a model key)")
            return
        n = _AUTO_TURNS.get(pid, 0)
        if n >= MAX_AUTO_TURNS:
            await pause(ctx, pid, tid, f"{n} automatic steps ran in a row")
            return
        _AUTO_TURNS[pid] = n + 1
        reason = "; ".join(p["text"] for p in pending if p.get("text")) or "background work"
        prompt = RESUME_PROMPT if any(p.get("resume") for p in pending) else WAKE_PROMPT
        try:
            async for chunk in agent.turn(ctx, pid, prompt.format(reason=reason), tid, auto=reason):
                relay(ctx, pid, chunk, tid)
        except asyncio.CancelledError:
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "done", "data": {"stopped": True}})
            raise
        except Exception as e:
            log.exception("autonomous turn failed")
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "error",
                                          "data": {"message": f"The automatic step failed: {e}"}})
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "done", "data": {}})


def stop(ctx: Ctx, pid: str) -> bool:
    """The user pressed stop: end the chain, a pause, and any turn in flight. True if a turn was stopped."""
    disarm(ctx, pid)
    _AUTO_TURNS.pop(pid, None)
    if (r := _runs(ctx).get(pid)) is not None:
        r.pop("paused", None)
        r["attempts"] = 0
        _save_runs(ctx)
    stopped = False
    for t in (_TASKS.pop(pid, None), _TURNS.pop(pid, None)):
        if t and not t.done():
            _STOPPED.add(pid)
            t.cancel()
            stopped = True
    # the studio's live line ends now, also when nothing was left to cancel (a turn the restart had ended)
    ctx.bus.publish(pid, "chat", {"kind": "live", "event": "done", "data": {"stopped": True}})
    return stopped


# ------------------------------------------------------------------ cut-off turns: resume, bounded retries, pause

OWNER = "backend.owner"  # the pid of the backend that runs this data dir's chains


def claim(ctx: Ctx) -> None:
    """Backend start: this process runs the chains from now on (see the module docstring)."""
    try:
        (ctx.data_dir / OWNER).write_text(str(os.getpid()))
    except OSError as e:
        log.warning("could not claim the data dir for this backend: %s", e)


def superseded(ctx: Ctx) -> bool:
    """A newer backend runs this data dir and is alive: this one is shutting down, so it leaves the work to it."""
    try:
        owner = int((ctx.data_dir / OWNER).read_text())
    except (OSError, ValueError):
        return False
    if owner == os.getpid():
        return False
    try:
        os.kill(owner, 0)
    except ProcessLookupError:
        return False
    except OSError:  # e.g. EPERM: it exists
        pass
    return True


def _runs(ctx: Ctx) -> dict[str, Any]:
    key = str(ctx.data_dir)
    if key not in _RUNS:
        try:
            _RUNS[key] = json.loads((ctx.data_dir / "agent_runs.json").read_text())
        except (OSError, ValueError):
            _RUNS[key] = {}
    return _RUNS[key]


def _save_runs(ctx: Ctx) -> None:
    if superseded(ctx):
        return
    try:
        (ctx.data_dir / "agent_runs.json").write_text(json.dumps(_runs(ctx)))
    except OSError as e:  # resuming after a restart needs it; this process still works without it
        log.warning("could not save agent run state: %s", e)


def track_turn(pid: str, task: asyncio.Task) -> asyncio.Task:
    """A user turn running in its own task (so Stop can end it after the studio's request is gone)."""
    _TURNS[pid] = task
    task.add_done_callback(lambda t: _TURNS.pop(pid, None) if _TURNS.get(pid) is t else None)
    return task


def run_started(ctx: Ctx, pid: str, tid: str, auto: bool) -> None:
    """A turn starts. It stays marked on disk until it ends, so the next start finds a turn a restart cut off."""
    _LIVE[pid] = {"thread": tid, "auto": auto, "status": None}
    r = _runs(ctx).setdefault(pid, {})
    r["turn"] = {"thread": tid, "auto": auto, "at": time.time()}
    if not auto:  # the user is back: a pause and earlier retries no longer apply
        r.pop("paused", None)
        r["attempts"] = 0
    _save_runs(ctx)


def agent_process(ctx: Ctx, pid: str, os_pid: int) -> None:
    """The turn's Claude Code process, so the next start can end it if a hard restart left it running."""
    if (r := _runs(ctx).get(pid)) and r.get("turn"):
        r["turn"]["agent_pid"] = os_pid
        _save_runs(ctx)


def set_status(pid: str, text: str | None) -> None:
    if text and pid in _LIVE:
        _LIVE[pid]["status"] = text


def was_cut_off(pid: str) -> bool:
    """A cancelled turn was cut off (the backend is stopping, however abruptly), unless the user stopped it."""
    return pid not in _STOPPED


def run_finished(ctx: Ctx, pid: str, cut_off: bool) -> None:
    """The turn ended. `cut_off`: the backend is stopping mid-turn, so the mark stays for the next start."""
    _LIVE.pop(pid, None)
    _STOPPED.discard(pid)
    r = _runs(ctx).get(pid)
    if r is not None and not cut_off and r.pop("turn", None) is not None:
        _save_runs(ctx)


async def recover(ctx: Ctx, pid: str, tid: str, why: str) -> None:
    """A step was cut off: pick it up again after a pause that grows each time. After MAX_RECOVERIES in a row the
    chain pauses until someone presses Continue or sends a message."""
    if superseded(ctx):  # the newer backend ended this step and picks it up itself
        return
    r = _runs(ctx).setdefault(pid, {})
    n = int(r.get("attempts") or 0)
    if n >= MAX_RECOVERIES:
        await pause(ctx, pid, tid, f"{why}, {n + 1} times in a row")
        return
    r["attempts"] = n + 1
    _save_runs(ctx)
    st = _state(ctx).setdefault(pid, {"thread": tid, "pending": []})
    st["thread"] = tid
    st.setdefault("pending", []).append({"text": why, "at": time.time(), "resume": True})
    _flush(ctx)
    _schedule_wake(ctx, pid, RECOVERY_BACKOFF_S[min(n, len(RECOVERY_BACKOFF_S) - 1)])


async def pause(ctx: Ctx, pid: str, tid: str, why: str) -> None:
    """Stop the automatic steps until Continue is pressed (or a message is sent), and say why in the chat."""
    from . import agent

    disarm(ctx, pid)
    r = _runs(ctx).setdefault(pid, {})
    r["paused"], r["attempts"] = {"thread": tid, "reason": why, "at": time.time()}, 0
    _save_runs(ctx)
    row = notice("needs_you", f"Paused: {why}. Press Continue to pick up from here.")
    await agent._append_log(ctx, pid, row, tid)
    ctx.bus.publish(pid, "chat", {**row, "thread": tid})
    ctx.bus.publish(pid, "chat", {"kind": "paused", "thread": tid, "reason": why})


async def resume(ctx: Ctx, pid: str) -> bool:
    """The Continue button: pick the workflow up again. False when a turn is already running."""
    from . import chat_store

    if pid in _LIVE or ((t := _TURNS.get(pid)) and not t.done()):
        return False
    r = _runs(ctx).setdefault(pid, {})
    paused = r.pop("paused", None) or {}
    r["attempts"] = 0
    _save_runs(ctx)
    _AUTO_TURNS.pop(pid, None)
    tid = paused.get("thread") or armed_thread(ctx, pid) or (await chat_store.current(ctx, pid))["id"]
    _state(ctx)[pid] = {"thread": tid, "pending": [{"text": "Continue was pressed", "at": time.time(),
                                                    "resume": True}]}
    _flush(ctx)
    _schedule_wake(ctx, pid, 0)
    return True


def status(ctx: Ctx, pid: str) -> dict[str, Any]:
    """What the studio shows: a turn running in this backend (and its live line), one about to resume, a chain
    waiting on background work, or a pause that needs a Continue click."""
    live, user, wake = _LIVE.get(pid), _TURNS.get(pid), _TASKS.get(pid)
    return {"running": bool(live) or bool(user and not user.done()),
            "thread": (live or {}).get("thread"), "status": (live or {}).get("status"),
            "resuming": bool(wake and not wake.done()) and not live,
            "waiting": armed_thread(ctx, pid) is not None,
            "paused": (_runs(ctx).get(pid) or {}).get("paused")}


def _end_orphan(os_pid: Any) -> None:
    """A Claude Code process a hard restart left running would act next to the resumed turn: end it."""
    if not os_pid or os.name != "posix":
        return
    try:
        cmd = subprocess.run(["ps", "-p", str(int(os_pid)), "-o", "command="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    if "claude" in cmd and "--mcp-config" in cmd:  # still our agent, not a reused pid
        with contextlib.suppress(OSError):
            os.kill(int(os_pid), signal.SIGTERM)
            log.info("ended Claude Code process %s left over from before the restart", os_pid)


async def resume_after_restart(ctx: Ctx) -> int:
    """Backend start: pick up what the restart cut off. A turn still marked as running ends in its chat, and in
    autonomous mode the agent picks the workflow up again (bounded, see recover). Jobs the restart ended report in
    their armed chain, which wakes as for any finished job. Returns how many turns resume."""
    from . import agent, autopilot
    from .agent_orchestrator import get_project

    n = 0
    for pid, r in list(_runs(ctx).items()):
        cut = r.pop("turn", None)
        if not cut:
            continue
        _save_runs(ctx)
        _end_orphan(cut.get("agent_pid"))
        tid = cut.get("thread")
        try:
            project = await get_project(ctx, pid)
        except Exception:  # the project is gone
            _runs(ctx).pop(pid, None)
            _save_runs(ctx)
            continue
        if not tid:
            continue
        await agent._append_log(ctx, pid, {"kind": "done"}, tid)  # the cut-off turn ends: its steps stop spinning
        if is_autonomous(project) and pid not in autopilot._running:
            await recover(ctx, pid, tid, "the backend restarted during the last step")
            n += 1
        else:
            await agent._append_log(ctx, pid, notice("needs_you", "Cortex restarted during this step. Send a message "
                                                                  "to continue."), tid)
    for job in ctx.jobs.interrupted:
        rerun = job.idempotency_key and any(j.idempotency_key == job.idempotency_key and j.status in ("queued", "running")
                                            for j in ctx.jobs.list(job.project_id))
        if armed_thread(ctx, job.project_id) and not rerun:  # e.g. a Slurm poller re-attached: that job reports
            await agent.post_job_done(ctx, job)
    for pid, st in list(_state(ctx).items()):  # chains whose wake-up the restart cut off
        if st.get("pending") and not _active(ctx, pid):
            _schedule_wake(ctx, pid)
    return n
