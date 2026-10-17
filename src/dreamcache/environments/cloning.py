"""Simulator cloning for sandboxed speculation.

MiniGrid envs are pure Python objects; ``copy.deepcopy`` of the unwrapped env copies
grid, objects (including moving obstacles and their cur_pos), agent pos/dir,
carrying, step_count and the numpy Generator state. Correctness is verified in
tests/environments/test_cloning.py rather than assumed.
"""
from __future__ import annotations

import copy

import numpy as np


def clone_env(env):
    """Independent deep copy of the (unwrapped) environment, RNG state included."""
    return copy.deepcopy(env.unwrapped)


def reseed(sim, seed: int) -> None:
    """Give a sandbox its own RNG stream (for independent stochastic futures)."""
    sim.np_random = np.random.default_rng(seed)


def state_fingerprint(sim) -> tuple:
    """Hashable summary of all mutable state used to compare clones."""
    u = sim.unwrapped if hasattr(sim, "unwrapped") else sim
    grid = tuple(u.grid.encode().flatten().tolist())
    carrying = None if u.carrying is None else (u.carrying.type, u.carrying.color)
    rng = u.np_random.bit_generator.state
    return (tuple(u.agent_pos), int(u.agent_dir), grid, carrying, int(u.step_count),
            repr(rng))


def sim_step(sim, action: int):
    """Step an unwrapped env clone. Returns (obs, reward, terminated, truncated)."""
    obs, reward, terminated, truncated, _ = sim.step(action)
    return obs, float(reward), bool(terminated), bool(truncated)
