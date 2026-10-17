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


def test_split_is_stratified_deterministic_and_complete():
    eps = [_ep(i, r) for i, r in enumerate(["left"] * 5 + ["center"] * 5 + ["right"] * 2)]
    a = split_episodes(eps, 0.2, seed=0)
    b = split_episodes(eps, 0.2, seed=0)
    assert a == b and set(a) == {e.id for e in eps}
    by_region = {r: [a[e.id] for e in eps if e.condition.cube_region == r] for r in ("left", "center", "right")}
    assert by_region["left"].count("eval") == 1 and by_region["center"].count("eval") == 1
    assert by_region["right"].count("eval") == 0  # too few to hold out


def test_stats():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == (0.0, 1.0)
    assert two_proportion_p(2, 10, 9, 10) < 0.05
    assert two_proportion_p(5, 10, 5, 10) == pytest.approx(1.0)


def test_shifted_branch_maps_to_left_region():
    assert region_of("center", (0.0, 0.0)) == "center"
    assert region_of("center", (0.04, 0.06)) == "left"
