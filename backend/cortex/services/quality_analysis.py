"""Deterministic per-episode quality checks on recorded trajectories.

Pure numpy so it is unit-testable and runs the same for sim and real data.
Every flag carries the measured value and the threshold that tripped it.
"""

from __future__ import annotations

import numpy as np

from ..models import QualityFlag, QualityReport

# Thresholds. Blur is compared against the dataset median, since absolute
# Laplacian variance depends on resolution and scene texture.
OVEREXPOSED_PX = 245
UNDEREXPOSED_PX = 12
EXPOSURE_FRAC_WARN = 0.25
DARK_MEAN_WARN = 45.0
LOW_LIGHT_MEAN = 110.0  # dimmer than typical training light: not broken, but a distribution shift
BRIGHT_MEAN_WARN = 215.0
BLUR_REL_WARN = 0.5
GAP_FACTOR = 1.5
IDLE_EPS = 1e-3
IDLE_FRAC_WARN = 0.6
MIN_DURATION_S = 1.0


def laplacian_var(gray: np.ndarray) -> float:
    """Sharpness on a contrast-normalized image, so dark or low-contrast frames
    are not mistaken for blurry ones."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6) * 50.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def frame_stats(frames: np.ndarray, max_samples: int = 16) -> dict[str, float]:
    """frames: (T, H, W, 3) uint8."""
    idx = np.linspace(0, len(frames) - 1, num=min(max_samples, len(frames))).astype(int)
    sample = frames[idx]
    gray = sample.mean(axis=-1)
    return {
        "brightness_mean": float(gray.mean()),
        "overexposed_frac": float((gray >= OVEREXPOSED_PX).mean()),
        "underexposed_frac": float((gray <= UNDEREXPOSED_PX).mean()),
        "blur_laplacian_var": float(np.median([laplacian_var(g) for g in gray])),
    }


def analyze_episode(
    frames: np.ndarray,
    timestamps: np.ndarray,
    actions: np.ndarray,
    fps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    expected_frames: int | None = None,
    blur_reference: float | None = None,
) -> QualityReport:
    flags: list[QualityFlag] = []
    m: dict[str, float] = {}

    n = len(timestamps)
    m["num_frames"] = float(n)
    m["duration_s"] = float(timestamps[-1] - timestamps[0]) if n > 1 else 0.0
    if n != len(frames) or n != len(actions):
        flags.append(QualityFlag(code="length_mismatch", severity="error",
                                 message="frames/actions/timestamps have different lengths",
                                 evidence={"timestamps": n, "frames": len(frames), "actions": len(actions)}))

    # --- timing
    dt = np.diff(timestamps) if n > 1 else np.array([])
    period = 1.0 / fps
    gaps = dt[dt > GAP_FACTOR * period]
    missing = int(np.sum(np.round(gaps / period) - 1)) if len(gaps) else 0
    m["max_gap_s"] = float(dt.max()) if len(dt) else 0.0
    m["missing_frames"] = float(missing)
    m["nonmonotonic_steps"] = float(np.sum(dt <= 0)) if len(dt) else 0.0
    if m["nonmonotonic_steps"]:
        flags.append(QualityFlag(code="timestamps_nonmonotonic", severity="error",
                                 message=f"{int(m['nonmonotonic_steps'])} non-increasing timestamps",
                                 evidence={"count": m["nonmonotonic_steps"]}))
    if missing:
        flags.append(QualityFlag(code="missing_frames", severity="warn",
                                 message=f"{missing} frames missing (max gap {m['max_gap_s']:.3f}s at {fps} fps)",
                                 evidence={"missing": missing, "max_gap_s": m["max_gap_s"]}))
    if expected_frames is not None and n < expected_frames:
        m["truncated_frac"] = 1 - n / expected_frames
    if m["duration_s"] < MIN_DURATION_S:
        flags.append(QualityFlag(code="too_short", severity="error",
                                 message=f"episode lasts {m['duration_s']:.2f}s",
                                 evidence={"duration_s": m["duration_s"], "min_s": MIN_DURATION_S}))

    # --- exposure / blur
    if len(frames):
        m.update(frame_stats(frames))
        if m["overexposed_frac"] > EXPOSURE_FRAC_WARN or m["brightness_mean"] > BRIGHT_MEAN_WARN:
            flags.append(QualityFlag(code="overexposed", severity="warn",
                                     message=f"{m['overexposed_frac']:.0%} of pixels saturated",
                                     evidence={k: m[k] for k in ("overexposed_frac", "brightness_mean")}))
        if m["underexposed_frac"] > EXPOSURE_FRAC_WARN or m["brightness_mean"] < DARK_MEAN_WARN:
            flags.append(QualityFlag(code="underexposed", severity="warn",
                                     message=f"mean brightness {m['brightness_mean']:.0f}/255",
                                     evidence={k: m[k] for k in ("underexposed_frac", "brightness_mean")}))
        elif m["brightness_mean"] < LOW_LIGHT_MEAN:
            flags.append(QualityFlag(code="low_light", severity="info",
                                     message=f"dim scene: mean brightness {m['brightness_mean']:.0f}/255",
                                     evidence={"brightness_mean": m["brightness_mean"], "threshold": LOW_LIGHT_MEAN}))
        if blur_reference and m["blur_laplacian_var"] < BLUR_REL_WARN * blur_reference:
            flags.append(QualityFlag(code="blurry", severity="warn",
                                     message="sharpness below half the dataset median",
                                     evidence={"laplacian_var": m["blur_laplacian_var"],
                                               "dataset_median": blur_reference}))

    # --- actions
    a = np.asarray(actions, dtype=np.float64)
    nan = int(np.isnan(a).sum())
    m["action_nan"] = float(nan)
    if nan:
        flags.append(QualityFlag(code="action_nan", severity="error",
                                 message=f"{nan} NaN action values", evidence={"count": nan}))
    a = np.nan_to_num(a)
    oob = (a < action_low - 1e-6) | (a > action_high + 1e-6)
    m["action_out_of_range_frac"] = float(oob.any(axis=1).mean()) if len(a) else 0.0
    if m["action_out_of_range_frac"] > 0:
        joints = np.where(oob.any(axis=0))[0].tolist()
        flags.append(QualityFlag(code="action_out_of_range", severity="error",
                                 message=f"actions exceed joint limits on joints {joints}",
                                 evidence={"frac_steps": m["action_out_of_range_frac"], "joints": joints}))
    if len(a) > 2:
        da = np.abs(np.diff(a, axis=0)).max(axis=1)
        m["idle_frac"] = float((da < IDLE_EPS).mean())
        m["jerk_mean"] = float(np.abs(np.diff(a, n=2, axis=0)).mean())
        if m["idle_frac"] > IDLE_FRAC_WARN:
            flags.append(QualityFlag(code="mostly_idle", severity="warn",
                                     message=f"robot idle for {m['idle_frac']:.0%} of steps",
                                     evidence={"idle_frac": m["idle_frac"]}))

    penalty = sum({"info": 0.0, "warn": 0.15, "error": 0.4}[f.severity] for f in flags)
    return QualityReport(metrics=m, flags=flags, score=max(0.0, 1.0 - penalty))
