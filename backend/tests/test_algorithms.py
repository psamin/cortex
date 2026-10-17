import numpy as np
import pytest

from cortex.models import Augmentation, Condition
from cortex.services import augment
from cortex.services.quality_analysis import analyze_episode
from cortex.services.recommendation_engine import (
    CellEvidence, beta_stats, recommend_demonstrations,
)

LOW, HIGH = -np.ones(6), np.ones(6)


def clean_episode(T=50, fps=10, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.integers(60, 200, size=(T, 32, 32, 3), dtype=np.uint8)
    ts = np.arange(T) / fps
    actions = np.cumsum(rng.normal(0, 0.01, size=(T, 6)), axis=0)
    return frames, ts, actions


def codes(report):
    return {f.code for f in report.flags}


def test_clean_episode_has_no_flags():
    f, ts, a = clean_episode()
    r = analyze_episode(f, ts, a, 10, LOW, HIGH)
    assert codes(r) == set() and r.score == 1.0


def test_missing_frames_detected():
    f, ts, a = clean_episode()
    keep = np.r_[0:20, 25:50]  # drop 5 frames
    r = analyze_episode(f[keep], ts[keep], a[keep], 10, LOW, HIGH)
    assert "missing_frames" in codes(r)
    assert r.metrics["missing_frames"] == 5  # t=1.9 -> 2.5 is 6 periods: 5 frames missing


def test_exposure_and_action_range():
    f, ts, a = clean_episode()
    dark = (f * 0.1).astype(np.uint8)
    a[10, 2] = 3.0
    r = analyze_episode(dark, ts, a, 10, LOW, HIGH)
    assert {"underexposed", "action_out_of_range"} <= codes(r)
    assert r.score < 1


def test_blur_relative_to_dataset():
    f, ts, a = clean_episode()
    blurred = np.repeat(np.repeat(f[:, ::4, ::4], 4, axis=1), 4, axis=2)
    sharp = analyze_episode(f, ts, a, 10, LOW, HIGH).metrics["blur_laplacian_var"]
    r = analyze_episode(blurred, ts, a, 10, LOW, HIGH, blur_reference=sharp)
    assert "blurry" in codes(r)


def test_beta_posterior():
    mean, std = beta_stats(0, 0)
    assert mean == pytest.approx(0.5)
    mean, _ = beta_stats(successes=1, trials=10)
    assert mean == pytest.approx(10 / 12)


def cell(region, light="normal", target=4, eff=4.0, s=0, t=0):
    return CellEvidence(condition=Condition(cube_region=region, lighting=light), target=target,
                        accepted=int(eff), effective=eff, eval_successes=s, eval_trials=t)


def test_coverage_gap_ranked_first_and_evidence_backed():
    cells = [cell("center"), cell("left", eff=1.0), cell("right")]
    recs = recommend_demonstrations("prj_x", cells, budget_remaining=10)
    assert recs[0].condition.cube_region == "left"
    assert recs[0].count == 3
    assert recs[0].support == "evidence_backed"
    assert "left" in recs[0].title


def test_observed_failures_beat_untested_cells():
    cells = [cell("center", s=9, t=10), cell("right", s=2, t=10), cell("left")]
    recs = recommend_demonstrations("prj_x", cells, budget_remaining=10)
    assert recs[0].condition.cube_region == "right"
    assert recs[0].count >= 3
    assert any(e.kind == "evaluation" for e in recs[0].evidence)


def test_budget_respected_and_prior_only_is_hypothesis():
    cells = [cell("left", eff=0.0, target=10), cell("right", eff=0.0, target=10)]
    recs = recommend_demonstrations("prj_x", cells, budget_remaining=6)
    assert sum(r.count for r in recs) <= 6
    full = [cell("left", s=0, t=2)]  # covered, too few trials to count as evidence
    recs = recommend_demonstrations("prj_x", full, budget_remaining=5, weights={"fail": 5})
    assert all(r.support == "hypothesis" for r in recs)


def test_augmentations_preserve_shape_and_change_pixels():
    f, _, _ = clean_episode(T=4)
    for a in augment.PRESETS["lighting_robustness"]:
        out = augment.apply(f, a, seed=1)
        assert out.shape == f.shape and out.dtype == np.uint8
        assert not np.array_equal(out, f)
    same = augment.apply(f, Augmentation(type="brightness", params={"factor": 1.0}))
    assert np.array_equal(same, f)


def test_dim_frames_are_not_called_blurry():
    f, ts, a = clean_episode()
    sharp = analyze_episode(f, ts, a, 10, LOW, HIGH).metrics["blur_laplacian_var"]
    dim = (f.astype(np.float32) * 0.4).astype(np.uint8)
    r = analyze_episode(dim, ts, a, 10, LOW, HIGH, blur_reference=sharp)
    assert "blurry" not in codes(r) and "low_light" in codes(r)
