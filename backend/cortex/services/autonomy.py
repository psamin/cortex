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
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..context import Ctx
from ..models import JobStatus, Project, TrainingJob
from . import training_service

log = logging.getLogger(__name__)
ACTOR = "autopilot"  # the standing approval (training_service.approve accepts it; the caller checks coverage)
APPROVER = "standing approval (autonomous mode)"
MAX_AUTO_TURNS = 24  # wake-ups in a row without a user message (a full round is about 4)
WAKE_DELAY_S = 1.5  # let sibling job lines land before waking the agent

WAKE_PROMPT = ("[Cortex, autonomous mode] Background work finished: {reason}. Continue the workflow toward the goal "
               "on your own. If the target is met, or the next step needs the human (spend outside the standing "
               "approval, or real-robot motion), say so in one sentence and stop.")

_LOCKS: dict[str, asyncio.Lock] = {}
_TASKS: dict[str, asyncio.Task] = {}  # wake-up turns in flight, per project
_AUTO_TURNS: dict[str, int] = {}  # consecutive wake-ups per project
_NOTIFIED: set[str] = set()  # training jobs already reported as needing the human
_STATES: dict[str, dict[str, Any]] = {}  # per data dir: {pid: {"thread": tid | None, "pending": [{text, at}]}}


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


async def turn_ended(ctx: Ctx, pid: str, tid: str, last_result_at: float) -> None:
    """Arm the chain if work is still running or results landed after the agent's last tool result.
    Otherwise end it."""
    from . import autopilot
    from .agent_orchestrator import get_project

    project = await get_project(ctx, pid)
    if not is_autonomous(project) or pid in autopilot._running:  # Autopilot runs its own loop
        disarm(ctx, pid)
        return
    st = _state(ctx).get(pid) or {}
    unseen = [p for p in st.get("pending", []) if p["at"] > last_result_at]
    if not unseen and not _active(ctx, pid):
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
    if ctx.jobs.closing or pid in autopilot._running:
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


def _schedule_wake(ctx: Ctx, pid: str) -> None:
    t = _TASKS.get(pid)
    if t and not t.done() and t is not asyncio.current_task():  # an automatic turn may chain the next one
        return
    _TASKS[pid] = asyncio.get_running_loop().create_task(_wake(ctx, pid))


async def _wake(ctx: Ctx, pid: str) -> None:
    """Run one agent turn on the armed thread, published live on the project stream."""
    from . import agent
    from .agent_orchestrator import get_project

    await asyncio.sleep(WAKE_DELAY_S)
    async with lock(pid):
        st = _state(ctx).get(pid) or {}
        tid, pending = st.get("thread"), st.get("pending") or []
        if not tid or not pending or _active(ctx, pid):
            return  # a user turn saw the results meanwhile, or more work started
        if not is_autonomous(await get_project(ctx, pid)) or await agent.engine(ctx) == "rules":
            # switched to manual meanwhile, or the rule-based planner (it can't plan a next step)
            disarm(ctx, pid)
            return
        n = _AUTO_TURNS.get(pid, 0)
        if n >= MAX_AUTO_TURNS:
            disarm(ctx, pid)
            row = notice("needs_you", f"Paused after {n} automatic steps in a row. Send a message to keep going.")
            await agent._append_log(ctx, pid, row, tid)
            ctx.bus.publish(pid, "chat", {**row, "thread": tid})
            return
        _AUTO_TURNS[pid] = n + 1
        reason = "; ".join(p["text"] for p in pending if p.get("text")) or "background work"
        try:
            async for chunk in agent.turn(ctx, pid, WAKE_PROMPT.format(reason=reason), tid, auto=reason):
                event, _, data = chunk.partition("\ndata: ")
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue
                ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": event.removeprefix("event: "),
                                              "data": payload})
        except asyncio.CancelledError:
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "done", "data": {"stopped": True}})
            raise
        except Exception as e:
            log.exception("autonomous turn failed")
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "error",
                                          "data": {"message": f"The automatic step failed: {e}"}})
            ctx.bus.publish(pid, "chat", {"kind": "live", "thread": tid, "event": "done", "data": {}})


def stop(ctx: Ctx, pid: str) -> bool:
    """The user pressed stop: end the chain and any wake-up turn in flight. True if a turn was stopped."""
    disarm(ctx, pid)
    _AUTO_TURNS.pop(pid, None)
    t = _TASKS.pop(pid, None)
    if t and not t.done():
        t.cancel()
        return True
    return False
