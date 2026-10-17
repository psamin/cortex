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
    error: str | None = None
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


class JobManager:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.procs: dict[str, list[asyncio.subprocess.Process]] = defaultdict(list)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, project_id: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.project_id == project_id]

    def update(self, job: Job, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(job, k, v)
        self.bus.publish(job.project_id, "job", job.model_dump())

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
        self.bus.publish(project_id, "job", job.model_dump())

        async def run() -> None:
            self.update(job, status="running")
            try:
                ref = await fn(job)
                self.update(job, status="succeeded", progress=1.0, result_ref=ref)
            except asyncio.CancelledError:
                self.update(job, status="cancelled", message="stopped by user")
            except Exception as e:  # surfaced to the UI and the agent
                log.exception("job %s failed", job.id)
                self.update(job, status="failed", error=f"{type(e).__name__}: {e}")

        self.tasks[job.id] = asyncio.create_task(run())
        return job

    async def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        for p in self.procs.get(job_id, []):
            if p.returncode is None:
                p.terminate()
                try:
                    await asyncio.wait_for(p.wait(), 5)
                except TimeoutError:
                    p.kill()
                    await p.wait()
        t = self.tasks.get(job_id)
        if t and not t.done():
            t.cancel()
        return job

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
        result_error = None
        assert proc.stdout
        async for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                self.log(job, raw.decode(errors="replace").rstrip())
                continue
            if ev.get("type") == "result" and not ev.get("ok"):
                result_error = ev.get("error")
                self.log(job, ev.get("trace", ""))
            if on_event:
                r = on_event(ev)
                if asyncio.iscoroutine(r):
                    await r
        rc = await proc.wait()
        await err_task
        if rc != 0:
            raise RuntimeError(result_error or f"worker exited {rc}: {' | '.join(err_tail[-5:])}")
