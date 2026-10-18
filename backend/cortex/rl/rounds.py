"""Closing the loop (docs/RL_DESIGN.md section 7): pure helpers for an RL round.

* ``select_targeted``: how many of a sandbox's demos fit under the targeted-data cap, measured on the train
  split exactly as ``dataset_service.split_episodes`` will make it.
* ``paired_compare``: old vs new checkpoint on matched held-out seeds, every condition, exact McNemar each.
* ``needs_human_demos``: the round stopping rule (no paired improvement on the weak condition twice running).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ..models import Episode, SimulationBranch
from ..services.augment import MAX_AUGMENTED_SHARE
from ..services.dataset_service import split_episodes
from .stats import mcnemar_exact

# Targeted demos stay at most this share of the train split (RL_DESIGN section 6: 30-40%), the same ceiling the
# dataset builder applies to augmented copies, to limit interference with the other conditions.
MAX_TARGETED_SHARE = MAX_AUGMENTED_SHARE
ALPHA = 0.05  # McNemar significance level for "improved" / "regressed"
POSSIBLE_REGRESSION = 0.10  # a drop this large is flagged even when n is too small for significance
RL_PLANNER = "rl_sandbox:"


def is_targeted(e: Episode) -> bool:
    return str(e.provenance.get("planner", "")).startswith(RL_PLANNER)


def select_targeted(base: list[Episode], new: list[Episode], eval_fraction: float, split_seed: int,
                    cap: float = MAX_TARGETED_SHARE) -> tuple[list[Episode], dict[str, Any]]:
    """The largest prefix of ``new`` whose addition keeps every targeted episode (earlier rounds' ones in ``base``
    included) at or below ``cap`` of the train split. Returns the chosen episodes and the measured shares."""
    for k in range(len(new), -1, -1):
        eps = base + new[:k]
        split = split_episodes(eps, eval_fraction, split_seed)
        train = [e for e in eps if split[e.id] == "train"]
        targeted = sum(is_targeted(e) for e in train)
        if k == 0 or (train and targeted <= cap * len(train) + 1e-9):
            return new[:k], {"cap": cap, "offered": len(new), "added": k, "train_episodes": len(train),
                             "targeted_train_episodes": targeted,
                             "targeted_share": round(targeted / len(train), 4) if train else 0.0}
    raise AssertionError("unreachable")


class ConditionComparison(BaseModel):
    """One condition, old vs new checkpoint on the same seeds (measured)."""

    kind: Literal["measured"] = "measured"
    name: str
    condition: str
    is_target: bool = False
    n: int
    seeds: list[int]
    old_successes: int
    new_successes: int
    old_rate: float
    new_rate: float
    delta: float
    discordant: tuple[int, int]  # (old only, new only)
    p_value: float
    verdict: Literal["improved", "regressed", "no_significant_change"]
    possible_regression: bool = False  # a drop >= POSSIBLE_REGRESSION that is not (yet) significant


def paired_compare(old: list[SimulationBranch], new: list[SimulationBranch], target_name: str | None = None,
                   alpha: float = ALPHA) -> list[ConditionComparison]:
    """Branches are matched by name and episodes by seed; unmatched seeds are left out (never imputed)."""
    by_name = {b.spec.name: b for b in new if b.result}
    out = []
    for ob in old:
        nb = by_name.get(ob.spec.name)
        if ob.result is None or nb is None:
            continue
        o = {e.seed: e.success for e in ob.result.episodes}
        n = {e.seed: e.success for e in nb.result.episodes}
        seeds = sorted(set(o) & set(n))
        if not seeds:
            continue
        ok, nk = sum(o[s] for s in seeds), sum(n[s] for s in seeds)
        only_old = sum(o[s] and not n[s] for s in seeds)
        only_new = sum(n[s] and not o[s] for s in seeds)
        p = mcnemar_exact(only_old, only_new)
        delta = (nk - ok) / len(seeds)
        verdict = ("improved" if p < alpha and delta > 0 else "regressed" if p < alpha and delta < 0
                   else "no_significant_change")
        out.append(ConditionComparison(
            name=ob.spec.name, condition=ob.spec.condition.key, is_target=ob.spec.name == target_name, n=len(seeds),
            seeds=seeds, old_successes=ok, new_successes=nk, old_rate=round(ok / len(seeds), 4),
            new_rate=round(nk / len(seeds), 4), delta=round(delta, 4), discordant=(only_old, only_new),
            p_value=round(p, 6), verdict=verdict,
            possible_regression=verdict != "regressed" and delta <= -POSSIBLE_REGRESSION))
    return out


def needs_human_demos(target_verdicts: list[str]) -> bool:
    """RL_DESIGN section 6: stop RL on a condition when the re-evaluated policy failed to improve it (paired test)
    in the last 2 consecutive rounds. ``target_verdicts``: oldest first, this round last."""
    return len(target_verdicts) >= 2 and all(v != "improved" for v in target_verdicts[-2:])
