"""Image-only augmentations. They change pixels, never actions or states, so
action labels stay valid. Augmented copies are linked to their source episode
and are never counted as new demonstrations."""

from __future__ import annotations

import numpy as np

from ..models import Augmentation


def apply(frames: np.ndarray, aug: Augmentation, seed: int = 0) -> np.ndarray:
    """frames: (T, H, W, 3) uint8 -> same shape uint8. Same transform on every
    frame of an episode so the augmented video stays temporally consistent."""
    x = frames.astype(np.float32)
    p = aug.params
    if aug.type == "brightness":
        x = x * p.get("factor", 1.0)
    elif aug.type == "contrast":
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        x = (x - mean) * p.get("factor", 1.0) + mean
    elif aug.type == "color_temperature":
        # shift > 0 warms (more red, less blue), < 0 cools. Range ~[-1, 1].
        s = p.get("shift", 0.0)
        x = x * np.array([1 + 0.2 * s, 1.0, 1 - 0.2 * s], dtype=np.float32)
    elif aug.type == "shadow":
        x = x * shadow_mask(frames.shape[1], frames.shape[2], p.get("strength", 0.5), seed)[None, :, :, None]
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(aug.type)
    return np.clip(x, 0, 255).astype(np.uint8)


def shadow_mask(h: int, w: int, strength: float, seed: int) -> np.ndarray:
    """A soft-edged half-plane shadow at a seeded random angle and offset."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(-0.3, 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    d = (np.cos(theta) * (xx / w - 0.5) + np.sin(theta) * (yy / h - 0.5)) - off
    soft = 1 / (1 + np.exp(-d * 25))
    return (1 - strength * soft).astype(np.float32)


PRESETS: dict[str, list[Augmentation]] = {
    "lighting_robustness": [
        Augmentation(type="brightness", params={"factor": 0.55}),
        Augmentation(type="brightness", params={"factor": 1.35}),
        Augmentation(type="color_temperature", params={"shift": 0.6}),
        Augmentation(type="shadow", params={"strength": 0.45}),
    ],
}
