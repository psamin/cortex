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
