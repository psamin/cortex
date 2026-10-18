"""Find where the current learned policy is weak, from measured simulator evaluations.

Pure function over evaluation records (``simulation_runner.list_evaluations`` shape) and
failure clusters. Only checkpoint evaluations count: the scripted expert is a reference,
not the policy being improved.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..models import FailureCluster
from .stats import start_bin, start_xy
from .types import StartCluster, Weakness


class NoPolicyEvaluation(ValueError):
    """There is no evaluation of a learned checkpoint to find weaknesses in."""


def latest_learned(evals: list[dict[str, Any]], checkpoint_id: str | None = None) -> dict[str, Any]:
    learned = [e for e in evals if e["run"].policy.kind == "checkpoint"
               and (checkpoint_id is None or e["run"].policy.checkpoint_id == checkpoint_id)]
    learned = [e for e in learned if any(b.result for b in e["branches"])]
    if not learned:
        who = f"checkpoint {checkpoint_id}" if checkpoint_id else "a trained checkpoint"
        raise NoPolicyEvaluation(f"no finished evaluation of {who}; run evaluate_learned_policy first")
    return learned[-1]


def find_weaknesses(evals: list[dict[str, Any]], clusters: list[FailureCluster], target: float = 0.8,
                    checkpoint_id: str | None = None, include_passing: bool = False) -> list[Weakness]:
    ev = latest_learned(evals, checkpoint_id)
    run = ev["run"]
    out: list[Weakness] = []
    for b in ev["branches"]:
        r = b.result
        if r is None or r.n == 0:
            continue
        gap = max(0.0, target - r.success_rate)
        if gap == 0 and not include_passing:
            continue
        modes = Counter(e.failure_mode or "other" for e in r.episodes if not e.success)
        width = r.ci95[1] - r.ci95[0]
        gap_hi = max(0.0, target - r.ci95[1])  # the gap that holds even at the optimistic end
        bins: dict[tuple[str, str], list[int]] = {}
        for e in r.episodes:
            if not e.success:
                x, y = start_xy(b.spec.condition.cube_region, tuple(b.spec.cube_offset), e.seed)
                bins.setdefault((start_bin(x, y), e.failure_mode or "other"), []).append(e.seed)
        starts = sorted((StartCluster(start_bin=k[0], failure_mode=k[1], count=len(v), seeds=v)
                           for k, v in bins.items()), key=lambda c: -c.count)
        refs = {f"{run.id}/{b.id}/seed{e.seed}" for e in r.episodes}
        vlm = [ev_text for cl in clusters if cl.source == "vlm"
               and (b.spec.condition.key in cl.conditions or refs & set(cl.episode_ids))
               for ev_text in cl.evidence[:2]]
        out.append(Weakness(
            key=b.spec.name, branch_id=b.id, evaluation_id=run.id, checkpoint_id=run.policy.checkpoint_id,
            policy_label=run.policy_label, condition=b.spec.condition, cube_offset=tuple(b.spec.cube_offset),
            light_scale=b.spec.light_scale, n=r.n, successes=r.successes, success_rate=r.success_rate,
            ci95=tuple(r.ci95), target=target, gap=round(gap, 4), failure_modes=dict(modes),
            dominant_failure=modes.most_common(1)[0][0] if modes else None,
            gap_at_upper=round(gap_hi, 4), start_clusters=starts[:6],
            # Confident gap first, then the raw gap as a small tie-break.
            score=round(gap_hi * max(0.0, 1.0 - width) + 0.01 * gap, 4), vlm_hypotheses=vlm[:4],
            evidence=[f"{run.id}/{b.id}"]))
    out.sort(key=lambda w: (-w.score, w.success_rate))
    return out
