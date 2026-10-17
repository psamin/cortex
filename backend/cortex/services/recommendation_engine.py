"""Rank which demonstrations to collect next.

For each condition cell c of the collection grid:

  coverage_gap  = max(0, target_c - effective_c) / target_c
                  effective_c counts accepted demos weighted by quality score
  failure_rate  = posterior mean of Beta(1 + failures_c, 1 + successes_c)
                  from simulation evaluations touching c (0.5 prior if none)
  uncertainty   = posterior std / std of Beta(1,1), so 1.0 = no evidence
  vlm_signal    = share of VLM failure hypotheses that mention c (capped at 1)
  diversity     = 1 - max similarity to cells already picked this round
                  (greedy, maximal-marginal-relevance style)

  score = w_gap*gap + w_fail*failure_rate*(1-uncertainty) + w_unc*uncertainty
          + w_vlm*vlm_signal + w_div*diversity

Cells are picked greedily until the remaining demo budget runs out. A
recommendation is "evidence_backed" when it rests on counted coverage or on at
least MIN_TRIALS evaluation episodes; if it rests mainly on the prior or on VLM
hypotheses it is labelled "hypothesis".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import Condition, Evidence, PlanCell, Recommendation

BETA11_STD = math.sqrt(1 / 12)
MIN_TRIALS = 5
DEFAULT_WEIGHTS = {"gap": 1.0, "fail": 1.2, "unc": 0.3, "vlm": 0.3, "div": 0.4}


@dataclass
class CellEvidence:
    condition: Condition
    target: int
    accepted: int = 0
    effective: float = 0.0  # quality-weighted accepted demos
    episode_ids: list[str] = field(default_factory=list)
    eval_successes: int = 0
    eval_trials: int = 0
    eval_ids: list[str] = field(default_factory=list)
    vlm_mentions: int = 0
    vlm_episode_ids: list[str] = field(default_factory=list)
    low_quality_ids: list[str] = field(default_factory=list)


def beta_stats(successes: int, trials: int) -> tuple[float, float]:
    """Mean failure rate and std of the Beta(1+f, 1+s) posterior."""
    a, b = 1 + (trials - successes), 1 + successes
    mean = a / (a + b)
    std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return mean, std


def similarity(a: Condition, b: Condition) -> float:
    regions = ["left", "center", "right"]
    lights = ["dim", "normal", "bright"]
    dr = abs(regions.index(a.cube_region) - regions.index(b.cube_region)) / 2
    dl = abs(lights.index(a.lighting) - lights.index(b.lighting)) / 2
    return 1 - (dr + dl) / 2


def score_cell(c: CellEvidence, picked: list[Condition], total_vlm: int,
               w: dict[str, float]) -> tuple[float, dict[str, float]]:
    gap = max(0.0, c.target - c.effective) / c.target if c.target else 0.0
    fail, std = beta_stats(c.eval_successes, c.eval_trials)
    unc = std / BETA11_STD
    vlm = min(1.0, c.vlm_mentions / total_vlm) if total_vlm else 0.0
    div = 1 - max((similarity(c.condition, p) for p in picked), default=0.0)
    comp = {"coverage_gap": gap, "failure_rate": fail, "uncertainty": unc,
            "vlm_signal": vlm, "diversity": div}
    s = (w["gap"] * gap + w["fail"] * fail * (1 - unc) + w["unc"] * unc
         + w["vlm"] * vlm + w["div"] * div)
    return s, comp


def recommend_demonstrations(
    project_id: str,
    cells: list[CellEvidence],
    budget_remaining: int,
    max_items: int = 4,
    weights: dict[str, float] | None = None,
) -> list[Recommendation]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_vlm = sum(c.vlm_mentions for c in cells)
    remaining = list(cells)
    picked: list[Condition] = []
    out: list[Recommendation] = []
    budget = budget_remaining

    while remaining and budget > 0 and len(out) < max_items:
        scored = [(score_cell(c, picked, total_vlm, w), c) for c in remaining]
        (s, comp), best = max(scored, key=lambda t: (t[0][0], t[1].condition.key))
        if s <= 0.05:
            break
        remaining.remove(best)
        fail_driven = best.eval_trials >= MIN_TRIALS and comp["failure_rate"] > 0.3
        need = math.ceil(max(0.0, best.target - best.effective))
        if fail_driven:
            need = max(need, 3)
        need = min(max(need, 0), budget)
        if need == 0:
            continue
        budget -= need
        picked.append(best.condition)
        out.append(_build(project_id, best, need, s, comp, fail_driven))
    return out


def _build(project_id: str, c: CellEvidence, count: int, score: float,
           comp: dict[str, float], fail_driven: bool) -> Recommendation:
    cond = c.condition
    ev: list[Evidence] = [Evidence(
        kind="coverage",
        description=f"{c.accepted} accepted demos in {cond.key} (quality-weighted {c.effective:.1f}) vs target {c.target}",
        ref_ids=c.episode_ids, value=comp["coverage_gap"])]
    if c.eval_trials:
        ev.append(Evidence(
            kind="evaluation",
            description=f"policy failed {c.eval_trials - c.eval_successes}/{c.eval_trials} simulated trials in this condition",
            ref_ids=c.eval_ids, value=1 - c.eval_successes / c.eval_trials))
    if c.low_quality_ids:
        ev.append(Evidence(kind="quality",
                           description=f"{len(c.low_quality_ids)} demos here were down-weighted by quality checks",
                           ref_ids=c.low_quality_ids))
    if c.vlm_mentions:
        ev.append(Evidence(kind="vlm",
                           description=f"{c.vlm_mentions} VLM failure hypotheses reference this condition (unverified)",
                           ref_ids=c.vlm_episode_ids, value=comp["vlm_signal"]))

    has_counted_gap = comp["coverage_gap"] > 0 and c.target > 0
    backed = has_counted_gap or c.eval_trials >= MIN_TRIALS
    where = {"left": "on the left", "center": "in the center", "right": "on the right"}[cond.cube_region]
    light = "" if cond.lighting == "normal" else f" under {cond.lighting} lighting"
    why = ("closes a coverage gap" if comp["coverage_gap"] >= comp["failure_rate"] * (1 - comp["uncertainty"])
           else "targets an observed failure mode")
    return Recommendation(
        project_id=project_id, action="collect",
        title=f"Collect {count} more demonstration{'s' if count != 1 else ''} with the cube {where}{light} — {why}",
        condition=cond, count=count, score=round(score, 4),
        components={k: round(v, 4) for k, v in comp.items()},
        evidence=ev, support="evidence_backed" if backed else "hypothesis")


def cells_from_plan(plan_cells: list[PlanCell]) -> dict[str, CellEvidence]:
    return {pc.condition.key: CellEvidence(condition=pc.condition, target=pc.target) for pc in plan_cells}
