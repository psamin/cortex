"""In-process job runner with explicit states, idempotency keys and a per-project
event bus (fanned out to SSE clients). Durable results live in the entity
tables; a job only tracks the work in flight."""

from __future__ import annotations

import asyncio
import json
import os
import logging
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import WorkerFailed, describe
from ..models import new_id, now

log = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    kind: str
    project_id: str
    status: JobState = "queued"
    progress: float = 0.0
    message: str = ""
    result_ref: str | None = None
    error: str | None = None  # written for people (see cortex.errors)
    error_detail: str | None = None  # technical text for internal errors
    idempotency_key: str | None = None
    created_at: str = Field(default_factory=lambda: now().isoformat())


class EventBus:
    def __init__(self) -> None:
        self.subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def publish(self, project_id: str, event: str, data: Any) -> None:
        for q in list(self.subs[project_id]):
            if q.qsize() < 1000:
                q.put_nowait((event, data))

    def subscribe(self, project_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subs[project_id].add(q)
        return q

    def unsubscribe(self, project_id: str, q: asyncio.Queue) -> None:
        self.subs[project_id].discard(q)


TERMINAL = ("succeeded", "failed", "cancelled")
MAX_KEPT = 500  # finished jobs remembered (in memory and on disk); unfinished ones are always kept


class JobManager:
    """Jobs run in this process, but their state is written to disk so a backend restart
    reports in-flight jobs as interrupted instead of forgetting them."""

    def __init__(self, bus: EventBus, persist: Path | None = None):
        self.bus = bus
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.procs: dict[str, list[asyncio.subprocess.Process]] = defaultdict(list)
        self.persist = persist
        self.on_finished: list[Callable[[Job], None]] = []
        self.closing = False
        self.stopping: set[str] = set()  # jobs a user asked to stop: their worker may die first and raise
        if persist and persist.exists():
            for row in json.loads(persist.read_text() or "[]"):
                j = Job.model_validate(row)
                if j.status not in TERMINAL:
                    j.status, j.error = "failed", "It stopped because Cortex restarted. Run it again."
                    j.error_detail = "interrupted by a backend restart"
                self.jobs[j.id] = j
            self._save()

    def _save(self) -> None:
        keep = {j.id for j in sorted(self.jobs.values(), key=lambda j: j.created_at)[-MAX_KEPT:]}
        self.jobs = {i: j for i, j in self.jobs.items() if i in keep or j.status not in TERMINAL}
        self.tasks = {i: t for i, t in self.tasks.items() if i in self.jobs}
        if self.persist:
            tmp = self.persist.with_suffix(".tmp")
            try:  # a snapshot for the next start: failing to write it must never end a job (#125)
                tmp.write_text(json.dumps([j.model_dump() for j in self.jobs.values()]))
                os.replace(tmp, self.persist)
            except OSError as e:  # e.g. Windows refusing the replace while another process has the file open
                log.warning("could not save job states to %s: %s", self.persist, e)

    def stop_requested(self, job_id: str) -> bool:
        """True once ``cancel`` was called: a worker that fails from here on was stopped, not broken."""
        return job_id in self.stopping

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, project_id: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.project_id == project_id]

    def update(self, job: Job, **kw: Any) -> None:
        was = job.status
        for k, v in kw.items():
            setattr(job, k, v)
        self.bus.publish(job.project_id, "job", job.model_dump())
        if "status" in kw or was != job.status:
            self._save()
        if job.status in TERMINAL and was not in TERMINAL:
            for cb in self.on_finished:
                try:
                    cb(job)
                except Exception:  # a listener must never break the job
                    log.exception("job finish listener failed")

    def log(self, job: Job, line: str) -> None:
        self.bus.publish(job.project_id, "log", {"job_id": job.id, "line": line})

    def submit(self, kind: str, project_id: str, fn: Callable[[Job], Awaitable[str | None]],
               idempotency_key: str | None = None) -> Job:
        if idempotency_key:
            for j in self.jobs.values():
                if j.idempotency_key == idempotency_key and j.status in ("queued", "running", "succeeded"):
                    return j
        job = Job(kind=kind, project_id=project_id, idempotency_key=idempotency_key)
        self.jobs[job.id] = job
        self._save()
        self.bus.publish(project_id, "job", job.model_dump())

        async def run() -> None:
            try:  # from the first line: an error here used to end the task unseen, the job "running" forever (#125)
                self.update(job, status="running")
                ref = await fn(job)
                self.update(job, status="succeeded", progress=1.0, result_ref=ref)
            except asyncio.CancelledError:
                if self.closing:
                    raise  # stays "running" on disk; the next start reports it as interrupted by the restart
                self.update(job, status="cancelled", message="stopped by user")
            except Exception as e:  # surfaced to the UI and the agent, in words people can act on
                if job.id in self.stopping:  # the stop ended the worker before the task saw the cancel
                    self.update(job, status="cancelled", message="stopped by user")
                    return
                log.exception("job %s failed", job.id)
                msg, detail = describe(e)
                self.update(job, status="failed", error=msg, error_detail=detail)
            finally:
                self.procs.pop(job.id, None)
                self.stopping.discard(job.id)

        self.tasks[job.id] = asyncio.create_task(run())
        return job

    @staticmethod
    async def _terminate(procs: list[asyncio.subprocess.Process]) -> None:
        for p in procs:
            if p.returncode is None:
                p.terminate()
        for p in procs:  # all signalled first, so the grace periods overlap
            if p.returncode is None:
                try:
                    await asyncio.wait_for(p.wait(), 5)
                except TimeoutError:
                    p.kill()
                    await p.wait()

    async def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is not None and job.status not in TERMINAL:
            self.stopping.add(job_id)  # before the worker dies, so its failure is read as the stop
        await self._terminate(list(self.procs.get(job_id, [])))
        t = self.tasks.get(job_id)
        if t and not t.done():
            t.cancel()
        return job

    async def shutdown(self) -> None:
        """Backend stopping: end worker processes (they would outlive us) and in-process tasks."""
        self.closing = True
        procs = [p for ps in self.procs.values() for p in ps]  # taken now: cancelled tasks drop their entries
        tasks = [t for t in self.tasks.values() if not t.done()]
        for t in tasks:  # first, so a killed worker isn't recorded as a failed job
            t.cancel()
        await self._terminate(procs)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_worker(self, job: Job, module: str, config: dict[str, Any], cfg_path: Path,
                         on_event: Callable[[dict[str, Any]], Any] | None = None,
                         env: dict[str, str] | None = None) -> None:
        """Run `python -m <module> cfg.json` and relay its JSON-line events.
        `on_event` may be sync or async; async handlers are awaited in order."""
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(config))
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", module, str(cfg_path), cwd=str(BACKEND_DIR),
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.procs[job.id].append(proc)
        err_tail: list[str] = []

        async def drain_err() -> None:
            assert proc.stderr
            async for raw in proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line and not any(s in line for s in ("objc[", "torchcodec", "libav", "dylib")):
                    err_tail.append(line)
                    del err_tail[:-20]

        err_task = asyncio.create_task(drain_err())
        result_error, user_facing = None, False
        assert proc.stdout
        async for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                self.log(job, raw.decode(errors="replace").rstrip())
                continue
            if ev.get("type") == "result" and not ev.get("ok"):
                result_error, user_facing = ev.get("error"), bool(ev.get("user_facing"))
                self.log(job, ev.get("trace", ""))
            if on_event:
                r = on_event(ev)
                if asyncio.iscoroutine(r):
                    await r
        rc = await proc.wait()
        await err_task
        if rc != 0:
            if result_error:
                raise WorkerFailed(result_error, user_facing)
            raise WorkerFailed(f"worker exited {rc}: {' | '.join(err_tail[-5:])}")
