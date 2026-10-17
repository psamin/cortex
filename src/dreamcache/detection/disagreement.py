import math
from collections import Counter


def disagreement(actions: list[int], n_legal: int) -> float:
    """1 - mode share, rescaled so that the most spread-out vote is 1."""
    n = len(actions)
    if n <= 1:
        return 0.0
    mode = Counter(actions).most_common(1)[0][1]
    lo = math.ceil(n / n_legal) / n
    return (1 - mode / n) / (1 - lo) if lo < 1 else 0.0
