import math


def normalized_entropy(probs: dict[int, float]) -> float:
    """H(p)/log(n) over normalized candidate scores (uncalibrated)."""
    n = len(probs)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return h / math.log(n)
