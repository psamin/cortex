"""Shared data types (frozen contract, see CONTRIBUTING.md §2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Cost:
    """Compute accounting for one decision. Latency and work are kept separate."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    llm_seconds: float = 0.0     # model-server busy time (sum over calls; proxy for GPU-s)
    cache_hits: int = 0
    malformed: int = 0
    sim_steps: int = 0           # speculative simulator steps (excl. the real step)
    sim_cpu_seconds: float = 0.0
    branches: int = 0
    wall_seconds: float = 0.0    # end-to-end decision latency

    def __iadd__(self, o: "Cost") -> "Cost":
        for f in fields(self):
            if f.name != "wall_seconds":
                setattr(self, f.name, getattr(self, f.name) + getattr(o, f.name))
        return self

    def as_dict(self):
        return asdict(self)


@dataclass
class Decision:
    action: int
    probs: dict[int, float]            # normalized candidate scores (NOT calibrated)
    cost: Cost = field(default_factory=Cost)
