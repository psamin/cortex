"""Bootstrap confidence intervals and paired comparisons over common seeds."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(a: dict, b: dict, n=10_000, seed=0):
    """a, b: seed -> value. Returns mean(a-b), 95% CI, two-sided bootstrap p, n_pairs."""
    common = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in common], float)
    if len(d) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    # sign-flip / centered bootstrap p-value for H0: mean diff = 0
    centered = bs - d.mean()
    p = float(np.mean(np.abs(centered) >= abs(d.mean())))
    return {"diff": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "p": p, "n": len(d)}
