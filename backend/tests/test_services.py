import pytest

from cortex.models import Condition, CreateProjectRequest, Episode, EpisodeSource, Project
from cortex.services import project_service
from cortex.services.dataset_service import split_episodes
from cortex.services.experiment_comparison import two_proportion_p
from cortex.services.recommendation_service import region_of
from cortex.services.simulation_runner import wilson
from cortex.store.base import Repo
from cortex.store.local import SqliteStore


async def test_sqlite_store_roundtrip_and_upsert(tmp_path):
    store = SqliteStore(tmp_path / "t.db")
    await store.init()
    await store.init()  # migrations are idempotent
    repo = Repo(store)
    p = project_service.create_project(CreateProjectRequest(task_prompt="Pick the cube and put it in the bowl"))
    await repo.save("projects", p, status="active")
    p.name = "renamed"
    await repo.save("projects", p, status="active")
    got = await repo.get("projects", p.id, Project)
    assert got and got.name == "renamed"
    assert len(await repo.list("projects", Project)) == 1


def test_project_spec_and_plan():
    p = project_service.create_project(CreateProjectRequest(task_prompt="I want my SO-101 to pick up a cube "
                                                                        "and place it in a bowl", demo_budget=24))
    assert p.spec.objects == ["cube", "bowl"]
    assert {c.condition.cube_region for c in p.plan.cells} == {"left", "center", "right"}
    assert p.plan.budget_total <= 24 + 2


def _ep(i, region="center", light="normal"):
    return Episode(project_id="p", index=i, source=EpisodeSource.sim_scripted,
                   condition=Condition(cube_region=region, lighting=light), seed=i, num_frames=10, fps=30,
                   success=True)


def test_split_is_per_condition_deterministic_and_complete():
    eps = [_ep(i, r) for i, r in enumerate(["left"] * 60 + ["center"] * 60 + ["right"] * 2)]
    for i, e in enumerate(eps):
        e.id = f"ep_{i:03d}"  # fixed ids: the split is a pure function of them
    a = split_episodes(eps, 0.2, seed=0)
    b = split_episodes(eps, 0.2, seed=0)
    assert a == b and set(a) == {e.id for e in eps}
    by_region = {r: [a[e.id] for e in eps if e.condition.cube_region == r] for r in ("left", "center", "right")}
    # each episode is held out with probability eval_fraction (a stable hash, #368), so the share is approximate
    assert 4 <= by_region["left"].count("eval") <= 24 and 4 <= by_region["center"].count("eval") <= 24
    assert by_region["right"].count("eval") == 0  # too few to hold out
    assert split_episodes(eps, 0.2, seed=1) != a  # the seed still picks a different split


def test_stats():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == (0.0, 1.0)
    assert two_proportion_p(2, 10, 9, 10) < 0.05
    assert two_proportion_p(5, 10, 5, 10) == pytest.approx(1.0)


def test_shifted_branch_maps_to_left_region():
    assert region_of("center", (0.0, 0.0)) == "center"
    assert region_of("center", (0.04, 0.06)) == "left"


def _vlm_diag(ref: str):
    from cortex.models import EpisodeDiagnosis, Hypothesis
    return EpisodeDiagnosis(episode_id=ref, model="m", hypotheses=[
        Hypothesis(category="grasp_miss", confidence=0.7, rationale="jaws close beside the cube")])


def test_vlm_clusters_name_the_condition_cells_they_cover():
    from cortex.models import Analysis, BranchSpec, EpisodeResult, SimulationBranch
    from cortex.services.analysis_service import build_clusters, failure_ref

    ep = Episode(project_id="prj_1", index=0, source=EpisodeSource.sim_agent, seed=1, num_frames=10, fps=30,
                 success=False, condition=Condition(cube_region="right", lighting="dim"))
    shifted = SimulationBranch(evaluation_id="ev_1", project_id="prj_1", seeds=[1000], spec=BranchSpec(
        name="shifted_object", condition=Condition(cube_region="center", lighting="normal"), cube_offset=(0.04, 0.06)))
    fail = {"eval_id": "ev_1", "branch": shifted, "ep": EpisodeResult(seed=1000, success=False, reward=0.0, steps=1)}
    an = Analysis(project_id="prj_1", diagnoses=[_vlm_diag(ep.id), _vlm_diag(failure_ref(fail))])

    vlm = [c for c in build_clusters("prj_1", an, [ep], [fail]) if c.source == "vlm"]
    assert len(vlm) == 1 and vlm[0].category == "vlm:grasp_miss"
    assert vlm[0].conditions == ["left/normal", "right/dim"]  # the shifted branch lands nearest 'left'


async def test_vlm_clusters_reach_the_recommendation_ranking(tmp_path):
    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import FailureCluster
    from cortex.services import agent_orchestrator as orch
    from cortex.services import recommendation_service

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    await ctx.repo.save("failure_clusters", FailureCluster(
        project_id=p.id, analysis_id="an_1", category="vlm:grasp_miss", source="vlm", episode_ids=["ep_a", "ep_b"],
        conditions=["center/normal"], evidence=[], uncertainty=0.3), status="current")
    recs = await recommendation_service.recommend(ctx, p)
    cell = next(r for r in recs if r.condition and r.condition.key == "center/normal")
    assert cell.components["vlm_signal"] > 0
    assert any(e.kind == "vlm" for e in cell.evidence)
