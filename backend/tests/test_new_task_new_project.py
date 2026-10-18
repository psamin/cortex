"""A new task typed into a project that already has demos gets its own project and simulation, so the old
task's demos never show up in, or get trained into, the new one ("pick up a water bottle" in the cube project
trained on its 27 cube demos). Rules engine: no Claude Code, no model key."""

import asyncio
import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Condition, CreateProjectRequest, Episode, EpisodeSource, EpisodeStatus
from cortex.services import agent
from cortex.services import agent_orchestrator as orch
from cortex.sim.so101 import configure_gl

configure_gl()

CUBE = "Pick up the cube and place it in the bowl"
BOTTLE = "pick up a water bottle and put it in the bin"


async def _ctx(tmp_path):
    return await build_ctx(Settings(data_dir=tmp_path, _env_file=None, agent_engine="rules"))


async def _turn(ctx, pid, message, tid=None):
    events = []
    async for chunk in agent.chat(ctx, pid, message, tid):
        ev, _, data = chunk.partition("\ndata: ")
        events.append((ev.removeprefix("event: "), json.loads(data)))
    return events


async def _demos(ctx, pid, n, source=EpisodeSource.sim_scripted):
    for i in range(n):
        await ctx.repo.save("episodes", Episode(project_id=pid, index=i, source=source, condition=Condition(),
                                                seed=i, num_frames=120, fps=30, success=True,
                                                status=EpisodeStatus.accepted))


def test_a_new_task_in_a_project_with_demos_starts_its_own_project(tmp_path):
    async def go():
        ctx = await _ctx(tmp_path)
        cube = await orch.create_project(ctx, CreateProjectRequest(task_prompt=CUBE))
        await _demos(ctx, cube.id, 3)
        events = await _turn(ctx, cube.id, BOTTLE)
        moved = next(d for ev, d in events if ev == "project")
        new = await orch.get_project(ctx, moved["id"])
        new_eps = await ctx.repo.list("episodes", Episode, project_id=new.id)
        # the new project's own first turn (what the studio sends after it opens it) builds its scene there
        events2 = await _turn(ctx, new.id, moved["prompt"])
        return events, moved, await orch.get_project(ctx, cube.id), new, new_eps, events2, \
            await orch.get_project(ctx, new.id)

    events, moved, cube, new, new_eps, events2, built = asyncio.run(go())
    assert moved["id"] != cube.id and moved["prompt"] == BOTTLE and moved["name"] == new.name
    assert new.task_prompt == BOTTLE and new_eps == []
    # the cube project is untouched: no intake, no scene, same goal
    assert cube.intake is None and cube.scene is None and "cube" in cube.spec.goal.lower()
    assert "start_intake" not in [d["tool"] for ev, d in events if ev == "tool_call"]
    note = next(d for ev, d in events if ev == "notice")
    assert "3 demos" in note["text"] and moved["id"] in note["ref_ids"]
    assert events[-1][0] == "done"
    assert built.scene is not None and set(built.scene["objects"]) == {"water_bottle"}
    assert "project" not in [ev for ev, _ in events2]  # an empty project builds in place


def test_the_same_task_or_a_question_stays_in_the_project(tmp_path):
    async def go():
        ctx = await _ctx(tmp_path)
        cube = await orch.create_project(ctx, CreateProjectRequest(task_prompt=CUBE))
        await _demos(ctx, cube.id, 2)
        a = await _turn(ctx, cube.id, "pick up the cube again")
        b = await _turn(ctx, cube.id, "what should I do next?")
        return a + b, await ctx.repo.list("projects", type(cube))

    events, projects = asyncio.run(go())
    assert "project" not in [ev for ev, _ in events] and len(projects) == 1


def test_start_intake_will_not_turn_a_project_with_demos_into_another_task(tmp_path):
    """Agents (Claude Code, MCP) reach the same rule through the tools."""
    async def go():
        ctx = await _ctx(tmp_path)
        cube = await orch.create_project(ctx, CreateProjectRequest(task_prompt=CUBE))
        await _demos(ctx, cube.id, 2)
        with pytest.raises(ValueError, match="2 demos"):
            await orch.call_tool(ctx, "start_intake", {"prompt": BOTTLE, "project_id": cube.id}, actor="agent")
        return await orch.get_project(ctx, cube.id)

    assert asyncio.run(go()).intake is None


def test_build_simulation_will_not_swap_the_objects_under_existing_demos(tmp_path):
    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=BOTTLE))
        await _turn(ctx, p.id, BOTTLE)  # empty project: the scene is built in place
        await _demos(ctx, p.id, 2)
        await orch.call_tool(ctx, "update_intake", {"project_id": p.id, "patch": {
            "objects": [{"name": "water_bottle", "_delete": True},
                        {"name": "tennis_ball", "shape": "sphere", "size": [0.065], "mass_kg": 0.06}],
            "task": {"pick": "tennis_ball"}}}, actor="agent")
        with pytest.raises(ValueError, match="2 demos"):
            await orch.call_tool(ctx, "build_simulation", {"project_id": p.id}, actor="agent")
        return await orch.get_project(ctx, p.id)

    p = asyncio.run(go())
    assert set(p.scene["objects"]) == {"water_bottle"}

