"""Image-only augmentations. They change pixels, never actions or states, so
action labels stay valid. Augmented copies are linked to their source episode
and are never counted as new demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

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
    elif aug.type == "hue_saturation":
        # saturation: 0 = grey, 1 = unchanged; hue: rotation in degrees (YIQ chroma plane)
        gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        x = gray[..., None] + (x - gray[..., None]) * p.get("saturation", 1.0)
        x = _rotate_hue(x, p.get("hue_deg", 0.0))
    elif aug.type == "blur":
        x = _box_blur(x, int(p.get("radius", 1)))
    elif aug.type == "noise":
        x = x + np.random.default_rng(seed).normal(0.0, p.get("sigma", 5.0), x.shape).astype(np.float32)
    elif aug.type == "camera_jitter":
        # the same small translation on every frame (a slightly moved camera), edges replicated
        dx, dy = int(p.get("dx", 0)), int(p.get("dy", 0))
        x = np.roll(np.pad(x, ((0, 0), (abs(dy),) * 2, (abs(dx),) * 2, (0, 0)), mode="edge"), (dy, dx), axis=(1, 2))
        x = x[:, abs(dy):abs(dy) + frames.shape[1], abs(dx):abs(dx) + frames.shape[2]]
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(aug.type)
    return np.clip(x, 0, 255).astype(np.uint8)


def _rotate_hue(x: np.ndarray, deg: float) -> np.ndarray:
    if not deg:
        return x
    to_yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)
    m = np.linalg.inv(to_yiq) @ rot @ to_yiq
    return x @ m.T.astype(np.float32)


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """Separable (2r+1)-tap box blur over H and W, edges replicated."""
    if r <= 0:
        return x
    for axis in (1, 2):
        pad = [(0, 0)] * 4
        pad[axis] = (r, r)
        c = np.cumsum(np.pad(x, pad, mode="edge"), axis=axis, dtype=np.float32)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = x.shape[axis]
        x = (np.take(c, np.arange(2 * r + 1, n + 2 * r + 1), axis=axis)
             - np.take(c, np.arange(0, n), axis=axis)) / (2 * r + 1)
    return x


def shadow_mask(h: int, w: int, strength: float, seed: int) -> np.ndarray:
    """A soft-edged half-plane shadow at a seeded random angle and offset."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(-0.3, 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    d = (np.cos(theta) * (xx / w - 0.5) + np.sin(theta) * (yy / h - 0.5)) - off
    soft = 1 / (1 + np.exp(-d * 25))
    return (1 - strength * soft).astype(np.float32)


# ------------------------------------------------------------------ evidence-sized plans (#102)

LIGHTING_GAP = 0.2  # baseline - dim success that triggers lighting augmentation (n >= 5 each)
MAX_AUGMENTED_SHARE = 0.4  # augmented copies stay at most this share of the training split
ASSUMED_DIM_RATIO = 0.55  # used only when no dim-light demos were recorded to measure it


@dataclass
class AugmentEvidence:
    """What was measured. Every field comes from stored telemetry or a human label."""

    lighting_gap: float = 0.0  # baseline - dim success rate in the latest learned-policy evaluation
    dim_brightness_ratio: float | None = None  # median frame brightness, dim / normal demos
    visibility_confirmed: bool = False  # a human confirmed a VLM "poor_visibility" cluster

    def as_components(self) -> dict[str, float]:
        out = {"lighting_gap": round(self.lighting_gap, 3), "visibility_confirmed": float(self.visibility_confirmed)}
        if self.dim_brightness_ratio is not None:
            out["dim_brightness_ratio"] = round(self.dim_brightness_ratio, 3)
        return out

    @classmethod
    def from_components(cls, c: dict[str, float]) -> AugmentEvidence:
        return cls(lighting_gap=c.get("lighting_gap", 0.0), dim_brightness_ratio=c.get("dim_brightness_ratio"),
                   visibility_confirmed=bool(c.get("visibility_confirmed", 0.0)))


def plan_augmentation(ev: AugmentEvidence, strength: float = 1.0) -> list[Augmentation]:
    """Transforms chosen by the evidence and sized by it. `strength` > 1 widens the plan after an
    augmented round failed to help (Autopilot); the share cap is applied by the dataset builder."""
    plan: list[Augmentation] = []
    if ev.lighting_gap >= LIGHTING_GAP:
        sev = float(np.clip(ev.lighting_gap / 0.5 * strength, 0.2, 1.5))  # gap 0.2 -> 0.4, 0.5+ -> 1.0
        target = ev.dim_brightness_ratio or ASSUMED_DIM_RATIO
        darkest = max(0.15, target * (1 - 0.15 * sev))  # cover the measured dimming, plus a margin
        views = 1 + round(2 * min(sev, 1.0))  # 1-3 brightness levels between ~normal and darkest
        top = min(0.9, (1 + darkest) / 2)
        for f in np.linspace(darkest, top, views):
            plan.append(Augmentation(type="brightness", params={"factor": round(float(f), 3)}))
        plan.append(Augmentation(type="shadow", params={"strength": round(min(0.8, 0.25 + 0.35 * sev), 3)}))
        plan.append(Augmentation(type="color_temperature", params={"shift": round(min(1.0, 0.3 + 0.4 * sev), 3)}))
    if ev.visibility_confirmed:
        plan += [Augmentation(type="contrast", params={"factor": round(max(0.4, 0.75 - 0.15 * (strength - 1)), 3)}),
                 Augmentation(type="blur", params={"radius": 1.0 if strength <= 1 else 2.0}),
                 Augmentation(type="noise", params={"sigma": round(6.0 * strength, 2)})]
    share = min(1.0, MAX_AUGMENTED_SHARE / (1 - MAX_AUGMENTED_SHARE) / max(1, len(plan)))
    return [a.model_copy(update={"params": {**a.params, "fraction": round(share, 4)}}) for a in plan]


def describe(plan: list[Augmentation]) -> str:
    return ", ".join(f"{a.type}({', '.join(f'{k}={v:g}' for k, v in a.params.items() if k != 'fraction')})"
                     for a in plan)


PRESETS: dict[str, list[Augmentation]] = {
    "lighting_robustness": [
        Augmentation(type="brightness", params={"factor": 0.55}),
        Augmentation(type="brightness", params={"factor": 1.35}),
        Augmentation(type="color_temperature", params={"shift": 0.6}),
        Augmentation(type="shadow", params={"strength": 0.45}),
    ],
}
