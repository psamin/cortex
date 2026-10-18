import json

from cortex.services.claude_code_agent import _result_events

HEADLINE = "HUMAN APPROVAL REQUIRED - cortex.{tool} was not executed. Do not retry until the human confirms."


def approval_block(tool: str, detail: str, tj: str | None = None, as_json: bool = True) -> dict:
    """A tool_result block as Claude Code relays mcp/src/backend.ts approvalRequired()."""
    if as_json:  # structuredContent handed to the model as JSON text
        text = json.dumps({"summary": HEADLINE.format(tool=tool), "ref_ids": [tj] if tj else [],
                           "approval_required": True, "executed": False, "detail": detail})
    else:  # plain text content
        text = "\n".join([HEADLINE.format(tool=tool), f"Backend: {detail}", "Ask the human to do this in the studio."])
    return {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": text}]}


def test_training_refusal_becomes_an_approval_card_for_that_job():
    block = approval_block("launch_training_job", "Training must be approved by a human", tj="tj_abc")
    events = _result_events("launch_training_job", block)
    assert [e for e, _ in events] == ["tool_result", "approval_required"]
    card = events[1][1]
    assert card["kind"] == "training" and card["ref_id"] == "tj_abc" and card["training_job_id"] == "tj_abc"


def test_budget_refusal_is_a_failed_step_with_the_reason_not_a_training_card():
    detail = "Planner budget: $10.00 of the project's $10.00 standing approval is spent"
    for as_json in (True, False):
        events = _result_events("start_agent_collection", approval_block("start_agent_collection", detail,
                                                                         as_json=as_json))
        assert [e for e, _ in events] == ["tool_result"]
        row = events[0][1]
        assert row["ok"] is False and "Planner budget" in row["summary"]


def test_ordinary_results_pass_through():
    block = {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "Created project prj_1\nref_ids: prj_1"}]}
    events = _result_events("create_project", block)
    assert events == [("tool_result", {"tool": "create_project", "summary": "Created project prj_1",
                                       "ref_ids": ["prj_1"], "ok": True})]


# ------------------------------------------------------------------ live status line (#168)

def test_status_lines_narrate_tool_calls_plans_and_thinking():
    from cortex.services.narration import thought_line, todo_line, tool_line

    assert tool_line("start_agent_collection", {"num_demos": 12}) == "Collecting 12 demos in sim…"
    assert tool_line("record_demonstrations", {"conditions": [{"count": 2}, {"count": 3}]}) == "Recording 5 demos in sim…"
    assert tool_line("evaluate_learned_policy", {}) == "Testing the policy in 3 conditions, 10 runs each…"
    assert tool_line("train_from_agent_demonstrations", {}) == "Building the dataset and preparing ACT training…"
    assert tool_line("some_new_tool") == "Some new tool…"
    todos = [{"content": "Collect demos", "status": "completed"},
             {"content": "Train", "status": "in_progress", "activeForm": "Training ACT"}]
    assert todo_line(todos) == "Training ACT…"
    assert thought_line("The user wants a policy. I'll collect demos first.") == "The user wants a policy…"
    assert len(thought_line("word " * 100)) <= 111


def test_a_claude_code_turn_streams_a_live_status_line(tmp_path, monkeypatch):
    import asyncio

    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import CreateProjectRequest
    from cortex.services import claude_code_agent as cc
    from cortex.services import project_service
    from cortex.services.agent import _sse

    stream = [
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "claude"},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u0", "name": "ToolSearch", "input": {}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "Start by collecting demonstrations. Then train."}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u1", "name": "TodoWrite", "input": {
            "todos": [{"content": "Collect demos", "status": "in_progress", "activeForm": "Collecting demos"}]}}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u2",
                                                       "name": "mcp__cortex__start_agent_collection",
                                                       "input": {"num_demos": 20}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "u2",
                                                  "content": [{"type": "text", "text": "ac_1 started\nref_ids: ac_1"}]}]}},
        {"type": "result", "session_id": "s1", "num_turns": 2},
    ]

    class Proc:
        pid = 0
        returncode = None
        stderr = None

        def __init__(self):
            async def lines():
                for ev in stream:
                    yield (json.dumps(ev) + "\n").encode()
            self.stdout = lines()

        async def wait(self):
            self.returncode = 0

    async def fake_exec(*a, **k):
        return Proc()

    monkeypatch.setattr(cc.asyncio, "create_subprocess_exec", fake_exec)

    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        p = project_service.create_project(CreateProjectRequest(task_prompt="pick up the cube"))
        await ctx.repo.save("projects", p, status="active")
        out = [chunk async for chunk in cc.chat(ctx, p.id, "teach it", _sse)]
        return [json.loads(c.split("data: ", 1)[1]).get("text") for c in out if c.startswith("event: status")]

    assert asyncio.run(go()) == ["Starting Claude Code…", "Planning…", "Loading the Cortex tools…",
                                 "Start by collecting demonstrations…", "Collecting demos…",
                                 "Collecting 20 demos in sim…", "Deciding the next step…"]
