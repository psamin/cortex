"""Candidate-action generation strategies for speculation.

  llm_rank     top-K actions by the fast policy's normalized scores
  llm_sample   distinct actions among stochastic proposals (padded by rank)
  uniform      all legal actions (K ignored)
  heuristic    observation-only heuristic action first, then by rank
"""
from __future__ import annotations

from dreamcache.policies.heuristic import ObsBFSPolicy


def generate(strategy: str, k: int, decision, legal, proposals=None, obs=None):
    ranked = sorted(legal, key=lambda a: -decision.probs.get(a, 0.0))
    # the fast action always leads so ties keep the baseline choice
    ranked.remove(decision.action)
    ranked.insert(0, decision.action)
    if strategy == "uniform":
        return list(legal)
    if strategy == "llm_rank":
        return ranked[:k]
    if strategy == "llm_sample":
        out = [decision.action]
        for a in proposals or []:
            if a not in out:
                out.append(a)
        for a in ranked:
            if len(out) >= k:
                break
            if a not in out:
                out.append(a)
        return out[:k]
    if strategy == "heuristic":
        h = ObsBFSPolicy(seed=0).act(obs)
        out = [decision.action] + ([h] if h != decision.action and h in legal else [])
        out += [a for a in ranked if a not in out]
        return out[:k]
    raise ValueError(strategy)
