"""Readiness: the checks a dataset must pass before training (docs/STACK_SPEC.md 2.1).

The problems that waste a training run rarely show in the loss: idle frames teach the policy to wait, a gripper
that never moves teaches it not to grasp, streams out of sync or a source recorded in other units make the
targets wrong. Each check here is a pure function of the recorded arrays, so it runs the same for sim and real
demos and is unit-testable. Readiness only reports: it never edits or drops data.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..context import Ctx
from ..errors import NotFound
from ..models import DatasetVersion, Episode, EpisodeSource, EpisodeStatus, Project
from .recording_service import list_episodes, load_npz

log = logging.getLogger(__name__)

Severity = Literal["info", "warn", "error"]
Verdict = Literal["ready", "ready_with_warnings", "blocked"]

IDLE_EDGE_S = 1.0  # idle this long at the start or end of a demo is worth trimming
JUMP_FACTOR = 8.0  # a step this many times the median step is a glitch
LAG_FLAG = 3  # frames by which actions may trail the recorded state before it is a sync problem
LAG_SEARCH = 10
UNIT_RATIO = (30.0, 100.0)  # degrees / radians = 57.3
FEW_DEMOS = 10
IDLE_DATA_FRAC = 0.4
MAX_EPISODES = 300  # beyond this, a seeded sample is checked
PICK_WORDS = re.compile(r"\b(pick|grasp|grab|place|put|lift|move|stack|sort|insert)\b", re.I)


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    episode_ids: list[str] = []
    evidence: dict = {}


class ReadinessReport(BaseModel):
    verdict: Verdict
    episodes_checked: int
    findings: list[Finding]
    summary: str
    dataset_version_id: str | None = None


# ------------------------------------------------------------------ per-episode

def _idle_eps(action: np.ndarray) -> float:
    """What counts as 'not moving', relative to how far the joints travel (works in radians or degrees)."""
    span = np.percentile(action, 99, axis=0) - np.percentile(action, 1, axis=0)
    return max(1e-3, 0.002 * float(np.median(span)))


def _edge_idle(steps: np.ndarray, eps: float) -> tuple[int, int]:
    moving = np.where(steps >= eps)[0]
    if not len(moving):
        return len(steps), 0
    return int(moving[0]), int(len(steps) - 1 - moving[-1])


def best_lag(action: np.ndarray, state: np.ndarray) -> int | None:
    """Shift (in frames) at which the recorded state best follows the commanded action, summed over joints.
    Positive is normal (state follows the command); negative means the actions trail the state."""
    n = min(len(action), len(state))
    if n < 4 * LAG_SEARCH or action.shape[1:] != state.shape[1:]:
        return None
    a, s = action[:n] - action[:n].mean(0), state[:n] - state[:n].mean(0)
    if float(np.abs(a).max()) < 1e-6 or float(np.abs(s).max()) < 1e-6:
        return None
    best, score = None, -np.inf
    for lag in range(-LAG_SEARCH, LAG_SEARCH + 1):
        x, y = (a[: n - lag], s[lag:]) if lag >= 0 else (a[-lag:], s[: n + lag])
        num = float((x * y).sum())
        den = float(np.sqrt((x * x).sum() * (y * y).sum())) + 1e-12
        if num / den > score:
            best, score = lag, num / den
    return best


def episode_findings(ep_id: str, action: np.ndarray, state: np.ndarray | None, fps: int,
                     gripper: int | None) -> list[Finding]:
    a = np.nan_to_num(np.asarray(action, dtype=np.float64))
    out: list[Finding] = []
    if a.ndim != 2 or len(a) < 3:
        return out
    steps = np.abs(np.diff(a, axis=0)).max(axis=1)
    eps = _idle_eps(a)
    head, tail = _edge_idle(steps, eps)
    for code, n, where in (("idle_head", head, "start"), ("idle_tail", tail, "end")):
        if n / fps >= IDLE_EDGE_S:
            out.append(Finding(code=code, severity="warn", episode_ids=[ep_id],
                               message=f"{n / fps:.1f} s without motion at the {where}",
                               evidence={"seconds": round(n / fps, 2), "frames": n}))
    if gripper is not None and gripper < a.shape[1]:
        g = a[:, gripper]
        span = float(np.percentile(g, 99) - np.percentile(g, 1))
        joints = np.percentile(a, 99, axis=0) - np.percentile(a, 1, axis=0)
        if span < 0.05 * float(joints.max()):
            out.append(Finding(code="gripper_unused", severity="warn", episode_ids=[ep_id],
                               message="the gripper never opens or closes",
                               evidence={"gripper_range": round(span, 4)}))
    # glitches are judged on the arm joints: a gripper is meant to snap open or shut in a frame or two
    arm = a[:, [j for j in range(a.shape[1]) if j != gripper]] if gripper is not None and a.shape[1] > 1 else a
    steps = np.abs(np.diff(arm, axis=0)).max(axis=1)
    moving = steps[steps >= eps]
    if len(moving) >= 5:
        med = float(np.median(moving))
        worst = float(steps.max())
        if worst > JUMP_FACTOR * med and worst > 4 * eps:
            out.append(Finding(code="action_jump", severity="warn", episode_ids=[ep_id],
                               message=f"one step moves {worst / med:.0f}x the usual amount",
                               evidence={"max_step": round(worst, 4), "median_step": round(med, 4),
                                         "frame": int(np.argmax(steps)) + 1}))
    if state is not None:
        lag = best_lag(a, np.asarray(state, dtype=np.float64))
        if lag is not None and lag <= -LAG_FLAG:
            out.append(Finding(code="action_state_lag", severity="warn", episode_ids=[ep_id],
                               message=f"actions trail the recorded state by {-lag} frames",
                               evidence={"lag_frames": lag}))
    return out


# ------------------------------------------------------------------ dataset-level

def _group(e: Episode) -> str:
    return "real" if e.source in (EpisodeSource.imported, EpisodeSource.real_teleop) else "sim"


def dataset_findings(items: list[tuple[Episode, np.ndarray]], wanted_conditions: list[str] | None) -> list[Finding]:
    out: list[Finding] = []
    fps = Counter(e.fps for e, _ in items)
    if len(fps) > 1:
        common = fps.most_common(1)[0][0]
        out.append(Finding(code="fps_mismatch", severity="error",
                           message="episodes were recorded at " + ", ".join(f"{f} fps ({n})" for f, n in fps.items()),
                           episode_ids=[e.id for e, _ in items if e.fps != common], evidence=dict(fps)))
    dims = Counter(a.shape[1] if a.ndim == 2 else 0 for _, a in items)
    if len(dims) > 1:
        common = dims.most_common(1)[0][0]
        out.append(Finding(code="shape_mismatch", severity="error",
                           message="action size differs: " + ", ".join(f"{d} values ({n})" for d, n in dims.items()),
                           episode_ids=[e.id for e, a in items if a.shape[1] != common], evidence=dict(dims)))
    # units: compare how far the joints travel in each source (degrees vs radians is ~57x)
    spans: dict[str, list[float]] = defaultdict(list)
    for e, a in items:
        if a.ndim == 2 and len(a) > 2:
            spans[_group(e)].append(float(np.median(np.percentile(a, 99, 0) - np.percentile(a, 1, 0))))
    if len(spans) == 2 and all(spans.values()):
        r = float(np.median(spans["real"])) / max(float(np.median(spans["sim"])), 1e-9)
        if UNIT_RATIO[0] <= r <= UNIT_RATIO[1] or UNIT_RATIO[0] <= 1 / max(r, 1e-9) <= UNIT_RATIO[1]:
            big = "real" if r > 1 else "sim"
            out.append(Finding(code="unit_mismatch", severity="error",
                               message=f"{big} demos move about {max(r, 1 / r):.0f}x further than the others: "
                                       "likely degrees mixed with radians",
                               episode_ids=[e.id for e, _ in items if _group(e) == big],
                               evidence={"ratio": round(r, 2)}))
    lengths = np.array([len(a) for _, a in items], dtype=float)
    if len(lengths) >= 5:
        med = float(np.median(lengths))
        mad = float(np.median(np.abs(lengths - med))) or 1.0
        odd = [e.id for (e, _), n in zip(items, lengths) if abs(n - med) > 3 * mad * 1.4826]
        if odd:
            out.append(Finding(code="length_outlier", severity="info", episode_ids=odd,
                               message=f"{len(odd)} episode{'s' if len(odd) > 1 else ''} far from the "
                                       f"median length ({med:.0f} frames)", evidence={"median_frames": med}))
    if wanted_conditions:
        have = Counter(e.condition.key for e, _ in items if _group(e) == "sim")
        thin = {c: have.get(c, 0) for c in wanted_conditions if have.get(c, 0) < FEW_DEMOS}
        if thin:
            out.append(Finding(code="few_demos", severity="warn",
                               message="under 10 demos for " + ", ".join(f"{c} ({n})" for c, n in thin.items()),
                               evidence=thin))
    idle, total = 0, 0
    for _, a in items:
        if a.ndim == 2 and len(a) > 2:
            steps = np.abs(np.diff(a, axis=0)).max(axis=1)
            idle += int((steps < _idle_eps(a)).sum())
            total += len(steps)
    if total and idle / total > IDLE_DATA_FRAC:
        out.append(Finding(code="mostly_idle_data", severity="warn",
                           message=f"{idle / total:.0%} of all steps are idle",
                           evidence={"idle_frac": round(idle / total, 3)}))
    return out


def verdict_of(findings: list[Finding]) -> Verdict:
    sev = {f.severity for f in findings}
    return "blocked" if "error" in sev else "ready_with_warnings" if "warn" in sev else "ready"


def summarize(verdict: Verdict, n: int, findings: list[Finding]) -> str:
    head = {"ready": "Ready", "ready_with_warnings": "Ready, with warnings", "blocked": "Blocked"}[verdict]
    shown = [f for f in findings if f.severity != "info"]
    if not shown:
        return f"{head}: {n} episodes checked, nothing to fix."
    # "idle start (3 episodes)": how many episodes each problem touches, not how many findings
    parts = [WORDS.get(f.code, f.code.replace("_", " ")) + (f" ({len(f.episode_ids)} episode"
             f"{'s' if len(f.episode_ids) != 1 else ''})" if f.episode_ids else "") for f in shown]
    return f"{head}: {n} episodes checked; " + ", ".join(parts)


WORDS = {"idle_head": "idle start", "idle_tail": "idle end", "gripper_unused": "gripper never moves",
         "action_jump": "glitch step", "action_state_lag": "actions out of sync", "fps_mismatch": "mixed frame rates",
         "shape_mismatch": "mixed action sizes", "unit_mismatch": "mixed units", "few_demos": "thin conditions",
         "mostly_idle_data": "mostly idle data", "unreadable": "unreadable file", "no_episodes": "no episodes"}


def _gripper_index(project: Project, dims: int) -> int | None:
    """SO-101 actions are 5 joints then the gripper; only checked for tasks that handle an object."""
    return dims - 1 if dims == 6 and PICK_WORDS.search(project.spec.goal or "") else None


def _merge(per_episode: list[Finding]) -> list[Finding]:
    """One finding per code across episodes, so a report lists 'idle_head x14', not 14 lines."""
    by: dict[str, list[Finding]] = defaultdict(list)
    for f in per_episode:
        by[f.code].append(f)
    out = []
    for code, fs in by.items():
        ids = [i for f in fs for i in f.episode_ids]
        if len(fs) == 1:
            out.append(fs[0])
            continue
        out.append(Finding(code=code, severity=max((f.severity for f in fs), key=["info", "warn", "error"].index),
                           message=f"{len(ids)} episodes: {fs[0].message}" + (" (and similar)" if len(fs) > 1 else ""),
                           episode_ids=ids, evidence={"episodes": len(ids)}))
    return out


async def check(ctx: Ctx, project: Project, episode_ids: list[str] | None = None,
                dataset_version_id: str | None = None) -> ReadinessReport:
    """Readiness of a dataset version's episodes, the given episodes, or (default) every accepted episode."""
    eps = await list_episodes(ctx, project.id)
    if dataset_version_id:
        ds = await ctx.repo.get("dataset_versions", dataset_version_id, DatasetVersion)
        if ds is None or ds.project_id != project.id:
            raise NotFound(dataset_version_id)
        wanted = {ln.episode_id for ln in ds.links}
        eps = [e for e in eps if e.id in wanted]
    elif episode_ids is not None:
        wanted = set(episode_ids)
        eps = [e for e in eps if e.id in wanted]
    else:
        eps = [e for e in eps if e.status == EpisodeStatus.accepted]
    if len(eps) > MAX_EPISODES:
        rng = np.random.default_rng(0)
        eps = [eps[i] for i in sorted(rng.choice(len(eps), MAX_EPISODES, replace=False))]
    items: list[tuple[Episode, np.ndarray]] = []
    per_episode: list[Finding] = []
    for e in eps:
        try:
            z = load_npz(ctx, e)
        except (OSError, KeyError, ValueError) as err:  # a missing or unreadable file is a finding, not a crash
            per_episode.append(Finding(code="unreadable", severity="error", episode_ids=[e.id],
                                       message="episode file missing or unreadable",
                                       evidence={"error": f"{type(err).__name__}"}))
            continue
        a = np.asarray(z.get("action", np.zeros((0, 0))))
        items.append((e, a))
        per_episode += episode_findings(e.id, a, z.get("state"), e.fps or 30,
                                        _gripper_index(project, a.shape[1] if a.ndim == 2 else 0))
    wanted_conditions = sorted({c.condition.key for c in project.plan.cells}) if project.plan.cells else None
    findings = dataset_findings(items, wanted_conditions) + _merge(per_episode)
    findings.sort(key=lambda f: ["error", "warn", "info"].index(f.severity))
    v = verdict_of(findings)
    if not eps:
        v, findings = "blocked", [Finding(code="no_episodes", severity="error", message="no episodes to check")]
    return ReadinessReport(verdict=v, episodes_checked=len(eps), findings=findings,
                           summary=summarize(v, len(eps), findings), dataset_version_id=dataset_version_id)
