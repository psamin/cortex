"""Difficulty score = alpha*entropy + beta*disagreement + gamma*risk (weights configurable)."""
from __future__ import annotations

from dataclasses import dataclass

from dreamcache.detection.disagreement import disagreement
from dreamcache.detection.risk import observable_risk
from dreamcache.detection.uncertainty import normalized_entropy


@dataclass
class DetectorConfig:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    n_proposals: int = 3

    @property
    def needs_proposals(self) -> bool:
        return self.beta > 0 and self.n_proposals > 1


def signals(decision, obs, legal, family, proposals=None) -> dict[str, float]:
    return {
        "entropy": normalized_entropy(decision.probs),
        "disagreement": disagreement([decision.action] + list(proposals), len(legal))
        if proposals else 0.0,
        "risk": observable_risk(obs, decision.action, family),
    }


def difficulty(sig: dict[str, float], cfg: DetectorConfig) -> float:
    w = cfg.alpha + cfg.beta + cfg.gamma
    return (cfg.alpha * sig["entropy"] + cfg.beta * sig["disagreement"]
            + cfg.gamma * sig["risk"]) / (w or 1.0)
