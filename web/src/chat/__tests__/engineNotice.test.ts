import { describe, expect, it } from "vitest";
import { applyEvent, historyToEvents, type ChatItem } from "../model";

// #192: the chat says which agent answers and what is missing, live and from history
describe("engine notices", () => {
  const row = { kind: "notice", tone: "engine", text: "Claude Code not found and no model key is set: using the built-in planner", ref_ids: [] };

  it("keep the engine tone, so the row reads 'Agent', not 'Autonomous'", () => {
    const items = historyToEvents([row]).reduce(applyEvent, [] as ChatItem[]);
    expect(items).toMatchObject([{ kind: "notice", tone: "engine", text: row.text }]);
  });

  it("leave the autonomous-mode tones as they were", () => {
    const items = historyToEvents([{ ...row, tone: "needs_you" }, { ...row, tone: "something_else" }]).reduce(applyEvent, [] as ChatItem[]);
    expect(items.map((it) => (it.kind === "notice" ? it.tone : null))).toEqual(["needs_you", "auto"]);
  });
});
