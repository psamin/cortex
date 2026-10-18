"""#192: a task prompt typed in the studio builds that task's own sim scene on the studio's project,
whichever engine answers (here the rules engine: no Claude Code, no model key)."""

import asyncio
import json

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest
from cortex.services import agent
from cortex.services import agent_orchestrator as orch
from cortex.sim.so101 import configure_gl

configure_gl()


async def _ctx(tmp_path):
    return await build_ctx(Settings(data_dir=tmp_path, _env_file=None, agent_engine="rules"))


async def _turn(ctx, pid, message):
    events = []
    async for chunk in agent.chat(ctx, pid, message):
        ev, _, data = chunk.partition("\ndata: ")
        events.append((ev.removeprefix("event: "), json.loads(data)))
    return events


def test_a_non_cube_prompt_builds_its_own_scene_not_the_cube(tmp_path):
    prompt = "pick up a water bottle and put it in the bin"

    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=prompt))  # what Home does first
        events = await _turn(ctx, p.id, prompt)
        return events, await orch.get_project(ctx, p.id), await ctx.repo.list("projects", type(p))

    events, p, projects = asyncio.run(go())
    assert [p.id] == [x.id for x in projects]  # the scene is on the studio's project, not a new one
    assert p.intake is not None and [o.name for o in p.intake.objects] == ["water_bottle"]
    assert p.intake.task.place == ["bin"]  # the zone is named after the prompt's target
    assert p.scene is not None and set(p.scene["objects"]) == {"water_bottle"}
    assert "cube" not in p.spec.goal.lower() and "water bottle" in p.spec.goal.lower()
    tools = [d["tool"] for ev, d in events if ev == "tool_call"]
    assert tools[0] == "start_intake" and "build_simulation" in tools
    built = next(d for ev, d in events if ev == "tool_result" and d["tool"] == "build_simulation")
    assert built.get("ok", True) and "built" in built["summary"]
    reply = "".join(d["delta"] for ev, d in events if ev == "text")
    assert "water bottle" in reply.lower() and "record_demonstrations" not in reply
    assert events[-1][0] == "done"


def test_a_new_prompt_in_the_same_chat_rebuilds_the_scene_for_it(tmp_path):
    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="put the cup on the plate"))
        await _turn(ctx, p.id, "put the cup on the plate")
        events = await _turn(ctx, p.id, "now use a tennis ball and put it in the bowl")  # a sphere: size [diameter]
        return events, await orch.get_project(ctx, p.id)

    events, p = asyncio.run(go())
    assert [o.name for o in p.intake.objects] == ["tennis_ball"] and p.intake.task.place == ["bowl"]
    assert set(p.scene["objects"]) == {"tennis_ball"} and "tennis ball" in p.spec.goal.lower()
    assert all(d.get("ok", True) for ev, d in events if ev == "tool_result")
    assert events[-1][0] == "done"


def test_the_cube_prompt_keeps_the_built_in_tabletop_task(tmp_path):
    prompt = "Pick up the cube and place it in the bowl"

    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=prompt))
        events = await _turn(ctx, p.id, prompt)
        return events, await orch.get_project(ctx, p.id)

    events, p = asyncio.run(go())
    assert p.intake is None and p.scene is None
    assert "start_intake" not in [d["tool"] for ev, d in events if ev == "tool_call"]
    assert events[-1][0] == "done"


def test_start_intake_with_a_project_id_seeds_that_project(tmp_path):
    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="teach the robot something"))
        out = await orch.call_tool(ctx, "start_intake", {"prompt": "put the cup on the plate", "project_id": p.id},
                                   actor="agent")
        return p.id, out, await ctx.repo.list("projects", type(p)), await orch.get_project(ctx, p.id)

    pid, out, projects, p = asyncio.run(go())
    assert out["project_id"] == pid and [x.id for x in projects] == [pid]
    assert [o.name for o in p.intake.objects] == ["cup"]


def test_the_same_prompt_after_a_stopped_turn_still_builds_the_scene(tmp_path):
    """Stop pressed right after start_intake left an intake naming the ball and no scene; sending the prompt again
    compared with the intake, saw nothing new and never built it."""
    prompt = "put the tennis ball in the bowl"

    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=prompt))
        await orch.call_tool(ctx, "start_intake", {"prompt": prompt, "project_id": p.id}, actor="agent")  # then Stop
        events = await _turn(ctx, p.id, prompt)
        again = await _turn(ctx, p.id, prompt)  # built now: the same prompt doesn't rebuild it
        return events, again, await orch.get_project(ctx, p.id)

    events, again, p = asyncio.run(go())
    assert "build_simulation" in [d["tool"] for ev, d in events if ev == "tool_call"]
    assert p.scene is not None and set(p.scene["objects"]) == {"tennis_ball"}
    assert "start_intake" not in [d["tool"] for ev, d in again if ev == "tool_call"]


def test_a_prompt_longer_than_the_intake_keeps_still_builds_its_scene(tmp_path):
    """A 7000-character prompt failed start_intake ("prompt: at most 500 characters") and stayed on the cube."""
    prompt = "pick up a water bottle and put it in the bin. " + "It stands on the left, near the edge. " * 180

    async def go():
        ctx = await _ctx(tmp_path)
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=prompt))
        events = await _turn(ctx, p.id, prompt)
        return events, await orch.get_project(ctx, p.id)

    events, p = asyncio.run(go())
    assert all(d.get("ok", True) for ev, d in events if ev == "tool_result")
    assert p.scene is not None and set(p.scene["objects"]) == {"water_bottle"} and len(p.intake.prompt) == 500
