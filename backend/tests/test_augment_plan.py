"""Adaptive augmentation (#102): transforms, evidence-sized plans, the augmented-share cap, and the
recommendation built from stored evaluations, demos and VLM clusters."""

import numpy as np
import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    Augmentation, BranchResult, BranchSpec, Condition, CreateProjectRequest, Episode, EpisodeSource, EpisodeStatus,
    EvaluationRun, FailureCluster, JobStatus, PolicyRef, QualityReport, SimulationBranch,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import augment, recommendation_service
from cortex.services.augment import MAX_AUGMENTED_SHARE, AugmentEvidence, plan_augmentation
from cortex.services.dataset_service import allocate_views

RNG = np.random.default_rng(0)
FRAMES = RNG.integers(30, 220, size=(5, 24, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("aug", [
    Augmentation(type="hue_saturation", params={"saturation": 0.5, "hue_deg": 25}),
    Augmentation(type="blur", params={"radius": 2}),
    Augmentation(type="noise", params={"sigma": 8}),
    Augmentation(type="camera_jitter", params={"dx": 3, "dy": -2}),
])
def test_new_transforms_change_only_pixels_and_are_deterministic(aug):
    a, b = augment.apply(FRAMES, aug, seed=7), augment.apply(FRAMES, aug, seed=7)
    assert a.shape == FRAMES.shape and a.dtype == np.uint8
    assert np.array_equal(a, b) and not np.array_equal(a, FRAMES)


def test_identity_parameters_leave_frames_unchanged():
    for aug in (Augmentation(type="hue_saturation", params={"saturation": 1.0, "hue_deg": 0}),
                Augmentation(type="blur", params={"radius": 0}),
                Augmentation(type="camera_jitter", params={"dx": 0, "dy": 0})):
        assert np.abs(augment.apply(FRAMES, aug).astype(int) - FRAMES).max() <= 1


def test_camera_jitter_is_a_translation():
    out = augment.apply(FRAMES, Augmentation(type="camera_jitter", params={"dx": 3, "dy": 2}))
    assert np.array_equal(out[:, 2:, 3:], FRAMES[:, :-2, :-3])


def brightness(plan):
    return sorted(a.params["factor"] for a in plan if a.type == "brightness")


def test_no_evidence_no_plan():
    assert plan_augmentation(AugmentEvidence(lighting_gap=0.1)) == []


def test_bigger_gap_means_more_and_darker_views():
    small = plan_augmentation(AugmentEvidence(lighting_gap=0.2, dim_brightness_ratio=0.5))
    big = plan_augmentation(AugmentEvidence(lighting_gap=0.6, dim_brightness_ratio=0.5))
    assert len(brightness(big)) > len(brightness(small))
    assert min(brightness(big)) < min(brightness(small)) <= 0.5  # covers the measured dimming
    shadow = {len(p): next(a.params["strength"] for a in p if a.type == "shadow") for p in (small, big)}
    assert shadow[len(big)] > shadow[len(small)]


def test_measured_dimming_sets_the_brightness_range():
    darker = plan_augmentation(AugmentEvidence(lighting_gap=0.4, dim_brightness_ratio=0.3))
    milder = plan_augmentation(AugmentEvidence(lighting_gap=0.4, dim_brightness_ratio=0.7))
    assert min(brightness(darker)) < 0.3 < min(brightness(milder)) < 0.7


def test_confirmed_visibility_adds_appearance_transforms():
    plan = plan_augmentation(AugmentEvidence(visibility_confirmed=True))
    assert {a.type for a in plan} == {"contrast", "blur", "noise"}


def test_strength_widens_the_plan():
    ev = AugmentEvidence(lighting_gap=0.3, dim_brightness_ratio=0.5)
    assert min(brightness(plan_augmentation(ev, strength=1.5))) < min(brightness(plan_augmentation(ev)))


@pytest.mark.parametrize("ev", [AugmentEvidence(lighting_gap=0.6, visibility_confirmed=True),
                                AugmentEvidence(lighting_gap=0.25)])
def test_augmented_copies_stay_within_the_share_cap(ev):
    plan = plan_augmentation(ev)
    for train in (3, 8, 20, 57):
        copies = sum(allocate_views([a.params["fraction"] for a in plan], train))
        assert copies / (train + copies) <= MAX_AUGMENTED_SHARE + 1e-9


def test_evidence_round_trips_through_recommendation_components():
    ev = AugmentEvidence(lighting_gap=0.35, dim_brightness_ratio=0.42, visibility_confirmed=True)
    assert AugmentEvidence.from_components(ev.as_components()) == ev


# ------------------------------------------------------------------ recommendation from stored evidence

async def project_with_eval(tmp_path, base_k: int, dim_k: int, n: int = 10):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    for i, (light, b) in enumerate([("normal", 120.0), ("normal", 130.0), ("dim", 52.0), ("dim", 48.0)]):
        await ctx.repo.save("episodes", Episode(
            project_id=p.id, index=i, source=EpisodeSource.sim_scripted, condition=Condition(lighting=light),
            seed=i, num_frames=10, fps=30, success=True, status=EpisodeStatus.accepted,
            quality=QualityReport(score=0.9, flags=[], metrics={"brightness_mean": b})))
    ev = EvaluationRun(project_id=p.id, policy=PolicyRef(kind="checkpoint", checkpoint_id="ck_1"),
                       policy_label="act", executor="local", name="it1", status=JobStatus.succeeded)
    for name, light, k in (("baseline", "normal", base_k), ("changed_lighting", "dim", dim_k)):
        b = SimulationBranch(evaluation_id=ev.id, project_id=p.id, executor="local", seeds=list(range(n)),
                             status=JobStatus.succeeded, spec=BranchSpec(name=name, condition=Condition(lighting=light)),
                             result=BranchResult(policy_name="act", wall_s=1, n=n, successes=k, success_rate=k / n,
                                                 ci95=(0, 1), mean_reward=0, episodes=[]))
        ev.branch_ids.append(b.id)
        await ctx.repo.save("simulation_branches", b, project_id=p.id)
    await ctx.repo.save("evaluation_runs", ev)
    return ctx, p


async def augment_rec(ctx, p):
    recs = await recommendation_service.recommend(ctx, p)
    return next((r for r in recs if r.action == "augment"), None)


async def test_measured_lighting_gap_gives_a_sized_evidence_backed_plan(tmp_path):
    ctx, p = await project_with_eval(tmp_path, base_k=9, dim_k=3)
    r = await augment_rec(ctx, p)
    assert r and r.support == "evidence_backed"
    assert r.components["lighting_gap"] == 0.6 and r.components["dim_brightness_ratio"] == 0.4  # 50/125 measured
    assert "brightness(factor=" in r.title and {e.kind for e in r.evidence} >= {"evaluation", "quality"}


async def test_no_gap_no_augmentation(tmp_path):
    ctx, p = await project_with_eval(tmp_path, base_k=8, dim_k=7)
    assert await augment_rec(ctx, p) is None


async def test_unconfirmed_visibility_is_only_a_hypothesis(tmp_path):
    ctx, p = await project_with_eval(tmp_path, base_k=8, dim_k=7)
    cl = FailureCluster(project_id=p.id, analysis_id="an_1", category="vlm:poor_visibility", source="vlm",
                        episode_ids=["ep_x"], conditions=["center/dim"], evidence=[], uncertainty=0.4)
    await ctx.repo.save("failure_clusters", cl, status="current")
    r = await augment_rec(ctx, p)
    assert r and r.support == "hypothesis" and "unverified" in r.title
    cl.verified_by_human = True
    await ctx.repo.save("failure_clusters", cl, status="current")
    r = await augment_rec(ctx, p)
    assert r.support == "evidence_backed" and "blur(" in r.title


def test_view_allocation():
    assert allocate_views([1.0, 1.0], 6) == [6, 6]  # manual presets keep "every train episode"
    assert allocate_views([0.25, 0.25, 0.25], 10) == [3, 2, 2]  # budget floor(7.5) = 7, largest remainder
    assert allocate_views([0.1] * 8, 3) == [1, 1, 0, 0, 0, 0, 0, 0]  # tiny dataset: budget 2, no overshoot
    assert allocate_views([0.3], 0) == [0]
