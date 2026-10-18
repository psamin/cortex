import { describe, expect, it } from "vitest";
import { plainText } from "../../components/ui";
import { applyEvent, buildBlocks, type ChatEvent, type ChatItem } from "../model";

const fold = (evs: ChatEvent[]) => evs.reduce(applyEvent, [] as ChatItem[]);
const PYDANTIC =
  "2 validation errors for Project\nintake.environment.conveyors\n  Extra inputs are not permitted [type=extra_forbidden, input_value=[], input_type=list]\n    For further information visit https://errors.pydantic.dev/2.9/v/extra_forbidden";

// #328: the studio shows what went right; errors stay in the backend logs
describe("chat without errors", () => {
  it("leaves failed tool calls out, and counts only the calls it shows", () => {
    const blocks = buildBlocks(
      fold([
        { type: "user", text: "teach a robot to pick and place a cube" },
        { type: "tool_call", tool: "get_project_state", args: {} },
        { type: "tool_result", tool: "get_project_state", summary: "Project prj_1", ref_ids: [] },
        { type: "tool_call", tool: "update_intake", args: {} },
        { type: "tool_result", tool: "update_intake", summary: PYDANTIC, ref_ids: [], ok: false },
        { type: "tool_call", tool: "start_intake", args: {} },
        { type: "tool_result", tool: "start_intake", summary: "intake started, 3 questions open", ref_ids: [] },
      ]),
    );
    const tools = blocks.filter((b) => b.type === "tools");
    expect(tools).toHaveLength(1);
    expect(tools[0].type === "tools" && tools[0].tools.map((t) => t.tool)).toEqual(["get_project_state", "start_intake"]);
    expect(JSON.stringify(blocks)).not.toContain("validation error");
  });

  it("leaves turn errors and failed jobs out", () => {
    const blocks = buildBlocks(
      fold([
        { type: "user", text: "go" },
        { type: "error", message: "The agent request failed." },
        { type: "job_done", job_id: "job_1", job_kind: "dataset", status: "failed", text: "Dataset failed: ValueError: x", ref_ids: [] },
        { type: "job_done", job_id: "job_2", job_kind: "agent_collect", status: "succeeded", text: "Demos finished", ref_ids: [] },
      ]),
    );
    expect(blocks.map((b) => (b.type === "item" ? b.item.kind : b.type))).toEqual(["user", "job_done"]);
  });

  it("leaves the backend's turn-failure lines out of the agent's reply", () => {
    const blocks = buildBlocks(
      fold([
        { type: "user", text: "go" },
        { type: "text", delta: "Started the demo collection." },
        { type: "text", delta: `\n\nThis turn failed (ValidationError: ${PYDANTIC.slice(0, 120)}). The backend log has the details.` },
        { type: "user", text: "again" },
        { type: "text", delta: "Claude Code exited with an error: idle timeout" },
      ]),
    );
    expect(blocks.map((b) => (b.type === "item" ? b.item.kind : b.type))).toEqual(["user", "assistant", "user"]);
    expect(blocks[1].type === "item" && blocks[1].item.kind === "assistant" && blocks[1].item.text).toBe("Started the demo collection.");
  });

  it("still asks to raise a spent planner budget, without the failed call's row", () => {
    const blocks = buildBlocks(
      fold([
        { type: "tool_call", tool: "start_agent_collection", args: {} },
        { type: "tool_result", tool: "start_agent_collection", summary: "Planner budget: $10.00 of the project's $10.00 used", ref_ids: [], ok: false },
      ]),
    );
    expect(blocks).toMatchObject([{ type: "tools", tools: [], budgetError: "Planner budget: $10.00 of the project's $10.00 used" }]);
  });
});

describe("plainText", () => {
  it("keeps plain sentences as they are", () => {
    const s = "Paused after 8 automatic steps in a row. Send a message to keep going.";
    expect(plainText(s)).toBe(s);
    expect(plainText("Claude Code is not logged in (run `claude` in a terminal and log in): the built-in planner answered instead.")).toBe(
      "Claude Code is not logged in (run `claude` in a terminal and log in): the built-in planner answered instead.",
    );
  });

  it("drops a raw error in brackets and keeps the rest", () => {
    expect(plainText("Stopped the automatic steps: the dataset step failed (ValueError: no episodes). Nothing was trained. Tell me which demos to use, or collect more.")).toBe(
      "Stopped the automatic steps: the dataset step failed. Nothing was trained. Tell me which demos to use, or collect more.",
    );
    expect(plainText("Can't reach Futurama (ssh: connect to host x port 22: Connection refused). Is the GT VPN on?")).toBe("Can't reach Futurama. Is the GT VPN on?");
  });

  it("cuts a raw tail, keeping a plain last clause", () => {
    expect(plainText(`Training tj_1 was approved but didn't start: ${PYDANTIC}`)).toBe("Training tj_1 was approved but didn't start.");
    expect(plainText("Claude Code failed: Traceback (most recent call last): boom: the built-in planner answered instead.")).toBe(
      "Claude Code failed: the built-in planner answered instead.",
    );
    expect(plainText(PYDANTIC)).toBe("");
    expect(plainText("ValueError: no episodes")).toBe("");
  });

  it("drops Claude Code, ssh and tunnel error text", () => {
    const tail = ": the built-in planner answered instead.";
    expect(plainText(`Claude Code failed: API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}${tail}`)).toBe(`Claude Code failed${tail}`);
    expect(plainText(`Claude Code failed: Error: spawn claude ENOENT${tail}`)).toBe(`Claude Code failed${tail}`);
    for (const raw of [
      "ssh: connect to host planetexpress-login.cc.gatech.edu port 22: Operation timed out",
      "ssh: Could not resolve hostname planetexpress-login.cc.gatech.edu: nodename nor servname provided",
      "bind [127.0.0.1]:8765: Address already in use",
      "Host key verification failed.",
    ]) {
      expect(plainText(`Can't reach Futurama (${raw}). Is the GT VPN on?`)).toBe("Can't reach Futurama. Is the GT VPN on?");
    }
  });
});
