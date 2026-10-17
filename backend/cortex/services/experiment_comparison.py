"""Side-by-side comparison of evaluation runs, branch by branch."""

from __future__ import annotations

import math

from ..context import Ctx
from ..models import ExperimentComparison
from . import simulation_runner as sim


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value of a pooled two-proportion z-test."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k2 / n2 - k1 / n1) / se
    return math.erfc(abs(z) / math.sqrt(2))


async def compare(ctx: Ctx, project_id: str, evaluation_ids: list[str] | None = None) -> ExperimentComparison:
    evals = await sim.list_evaluations(ctx, project_id)
    if evaluation_ids:
        evals = [e for e in evals if e["run"].id in evaluation_ids]
    rows, deltas = [], []
    by_branch: dict[str, list] = {}
    for ev in evals:
        for b in ev["branches"]:
            r = b.result
            rows.append({"evaluation_id": ev["run"].id, "evaluation": ev["run"].name, "policy": ev["run"].policy_label,
                         "executor": ev["run"].executor, "branch": b.spec.name, "status": b.status,
                         "n": r.n if r else 0, "successes": r.successes if r else 0,
                         "success_rate": r.success_rate if r else None, "ci95": r.ci95 if r else None,
                         "mean_reward": r.mean_reward if r else None,
                         "seeds": b.seeds})
            if r:
                by_branch.setdefault(b.spec.name, []).append((ev["run"], r))
    for name, items in by_branch.items():
        for (ra, a), (rb, b) in zip(items, items[1:]):
            p = two_proportion_p(a.successes, a.n, b.successes, b.n)
            deltas.append({"branch": name, "from": ra.id, "to": rb.id, "from_policy": ra.policy_label,
                           "to_policy": rb.policy_label, "delta_success": round(b.success_rate - a.success_rate, 3),
                           "p_value": round(p, 4), "significant": p < 0.05})
    if not rows:
        summary = "No evaluations yet."
    elif deltas:
        parts = [f"{d['branch']}: {d['delta_success']:+.0%} (p={d['p_value']:.2f}{', significant' if d['significant'] else ', not significant'})"
                 for d in deltas]
        summary = "Change vs previous run — " + "; ".join(parts)
    else:
        parts = [f"{r['branch']} {r['successes']}/{r['n']}" for r in rows if r["success_rate"] is not None]
        summary = "Single run: " + ", ".join(parts)
    return ExperimentComparison(evaluation_ids=[e["run"].id for e in evals], rows=rows, deltas=deltas, summary=summary)
