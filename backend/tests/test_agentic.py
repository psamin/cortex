"""Agent-demonstration loop tests. The 'model' here is a scripted test double
behind httpx.MockTransport: it exercises the real AstraPlanner + tool API code
path without calling GPT-6 Astra. It is never used to produce demonstrations."""

import json

import httpx
import numpy as np
import pytest

from cortex.agentic.astra import AstraPlanner, PlannerConfig
from cortex.sim.manipulation import ManipulationSession, MoveArgs, tool_schemas
from cortex.sim.so101 import PickPlaceEnv, Variation


@pytest.fixture(scope="module")
def env():
    e = PickPlaceEnv(episode_steps=10_000)
    yield e
    e.close()


def session(env, seed=3, region="center", mode="camera"):
    s = ManipulationSession(env, mode=mode)
    s.reset_environment(seed, Variation(cube_region=region))
    return s


def test_tool_schemas_are_valid_function_definitions():
    names = {t["function"]["name"] for t in tool_schemas()}
    assert names == {"observe_scene", "get_robot_state", "propose_grasp", "move_end_effector", "set_gripper",
                     "verify_grasp", "place_object", "go_home", "check_task_success", "finish_episode"}
    for t in tool_schemas():
        assert t["function"]["parameters"]["type"] == "object"
    MoveArgs.model_validate({"target": {"x": 0.2, "y": 0.0, "z": 0.05}})


@pytest.mark.parametrize("args,code", [
    ({"target": {"x": 0.6, "y": 0.0, "z": 0.05}}, "outside_workspace"),
    ({"target": {"x": 0.2, "y": 0.0, "z": 0.007}}, "table_collision"),
    ({"target": {"x": 0.12, "y": 0.2, "z": 0.19}}, "unreachable"),
    ({"target": "left please"}, "invalid_arguments"),
    ({}, "invalid_arguments"),
])
def test_invalid_motor_goals_are_rejected_without_moving(env, args, code):
    s = session(env)
    out, _ = s.call("move_end_effector", args)
    assert out["ok"] is False and out["error"]["code"] == code
    assert out["sim_steps"] == 0 and len(s.rec) == 0


def test_gripper_range_and_preconditions(env):
    s = session(env)
    assert s.call("set_gripper", {"state": "half"})[0]["error"]["code"] == "invalid_arguments"
    assert s.call("place_object", {"target": {"x": 0.2, "y": 0.1, "z": 0.05}})[0]["error"]["code"] == "not_holding"
    assert s.call("propose_grasp", {"object_id": "red_cube"})[0]["error"]["code"] == "image_point_required"
    assert s.call("propose_grasp", {"object_id": "blue_mug"})[0]["error"]["code"] == "unknown_object"
    # the tabletop box has no task-spec geometry: in camera mode it must be pointed at
    s.holding = True
    assert s.call("place_object", {"zone": "box"})[0]["error"]["code"] == "image_point_required"
    assert s.call("place_object", {"zone": "bin"})[0]["error"]["code"] == "unknown_zone"
    s.holding = False


def test_cube_prompt_and_tools_keep_naming_the_cube(env):
    from cortex.agentic.astra import scene_prompt

    s = session(env)
    ctx = s.task_context()
    assert ctx["object"] == "red_cube" and ctx["grasp_mode"] == "top" and ctx["scene"] == "tabletop"
    scene, plan = scene_prompt(ctx)
    assert "red cube" in scene and "go_home" in plan
    desc = {t["function"]["name"]: t["function"]["description"] for t in tool_schemas("pixels", ctx)}
    assert "red cube" in desc["check_task_success"] and "the box" in desc["check_task_success"]


def test_go_home_is_a_recorded_validated_motion(env):
    s = session(env)
    s.call("observe_scene", {"camera": "overhead"})
    n = len(s.rec)
    out, _ = s.call("go_home", {})
    assert out["ok"] and out["reached_home"] and out["sim_steps"] > 0 and len(s.rec) > n


def fake_model(sess: ManipulationSession, env: PickPlaceEnv, bad_first_move: bool = False):
    """Returns a MockTransport handler that plays a fixed tool-call script."""
    state = {"n": 0}

    def call(name, args):
        state["n"] += 1
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{state['n']}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

    script = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-6-astra" and body["tools"]
        if not script:
            script.append(("observe_scene", lambda: {"camera": "overhead"}))
            if bad_first_move:
                script.append(("move_end_effector", lambda: {"target": {"x": 0.7, "y": 0, "z": 0.05}}))
            script.append(("propose_grasp", lambda: {"object_id": "red_cube", "image_point": dict(
                zip(("u", "v"), sess.table_to_pixel(env.cube_pos())))}))
            script.append(("set_gripper", lambda: {"state": "open"}))
            script.append(("move_end_effector", lambda: {"target": last["grasp_pre"]}))
            script.append(("move_end_effector", lambda: {"target": last["grasp"]}))
            script.append(("set_gripper", lambda: {"state": "closed"}))
            script.append(("verify_grasp", lambda: {}))
            script.append(("move_end_effector", lambda: {"target": {**last["grasp"], "z": 0.075}}))
            script.append(("place_object", lambda: {"image_point": dict(zip(("u", "v"), sess.table_to_pixel(
                env.box_pos())))}))
            script.append(("go_home", lambda: {}))
            script.append(("check_task_success", lambda: {}))
            script.append(("finish_episode", lambda: {"believes_success": True, "summary": "done"}))
        for m in reversed(body["messages"]):  # read back the latest propose_grasp result
            if m["role"] == "tool" and '"pregrasp"' in m["content"]:
                r = json.loads(m["content"])
                last["grasp_pre"], last["grasp"] = r["pregrasp"], r["grasp"]
                break
        name, mk = script.pop(0) if len(script) > 1 or state["n"] else script.pop(0)
        msg = call(name, mk())
        return httpx.Response(200, json={"model": "openai/gpt-6-astra", "choices": [{"message": msg}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "cost": 0.0125}})
    last: dict = {}
    return handler


def test_planner_loop_records_a_real_successful_trajectory(env):
    s = session(env, seed=5, region="left")
    planner = AstraPlanner(PlannerConfig(provider="insforge", api_key="test"),
                           transport=httpx.MockTransport(fake_model(s, env)))
    events = []
    run = planner.run_episode(s, "pick the cube and place it in the bowl", events.append)
    assert run["finished"] and env.in_box()
    assert run["usage"]["calls"] == 12 and run["usage"]["cost_usd"] == pytest.approx(12 * 0.0125)
    assert run["usage"]["cost_source"].startswith("provider-reported")
    phases = [t["phase"] for t in run["trace"]]
    assert phases[:2] == ["Observing scene", "Selecting grasp"] and "Closing gripper" in phases
    assert "Returning home" in phases
    n = len(s.rec)
    assert n > 60 and len(s.rec.front) == len(s.rec.state) == len(s.rec.wrist) == n
    assert np.stack(s.rec.front).shape[1:] == (96, 128, 3)
    # recorded actions are joint targets in so101_follower units, within limits
    acts = np.stack(s.rec.action)
    assert acts.shape == (n, 6) and acts[:, 5].min() >= 0 and acts[:, 5].max() <= 100


def test_planner_recovers_from_a_rejected_command(env):
    s = session(env, seed=6, region="right")
    planner = AstraPlanner(PlannerConfig(provider="insforge", api_key="test"),
                           transport=httpx.MockTransport(fake_model(s, env, bad_first_move=True)))
    run = planner.run_episode(s, "task", lambda e: None)
    assert run["trace"][1]["ok"] is False and run["trace"][1]["result"]["error"]["code"] == "outside_workspace"
    assert run["recoveries"] >= 1 and env.in_box()


def test_call_budget_stops_the_episode(env):
    s = session(env, seed=7)
    planner = AstraPlanner(PlannerConfig(provider="insforge", api_key="test", max_calls=3),
                           transport=httpx.MockTransport(fake_model(s, env)))
    run = planner.run_episode(s, "task", lambda e: None)
    assert not run["finished"] and "budget" in run["stop_reason"] and run["usage"]["calls"] == 3
    assert not env.in_box()
