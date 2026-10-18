import { describe, expect, it } from "vitest";
import { groupThreads } from "../threadGroups";

const now = new Date(2026, 9, 18, 14, 0); // local time, Oct 18 2026 14:00
const at = (d: number, h = 12) => new Date(2026, 9, d, h).toISOString();

describe("groupThreads", () => {
  it("buckets chats like ChatGPT, newest first, without empty groups", () => {
    const threads = [
      { id: "old", updated_at: at(1) },
      { id: "today-early", updated_at: at(18, 1) },
      { id: "yesterday", updated_at: at(17, 23) },
      { id: "today-late", updated_at: at(18, 13) },
      { id: "week", updated_at: at(12) },
    ];
    const groups = groupThreads(threads, now).map(([label, xs]) => [label, xs.map((t) => t.id)]);
    expect(groups).toEqual([
      ["Today", ["today-late", "today-early"]],
      ["Yesterday", ["yesterday"]],
      ["Previous 7 days", ["week"]],
      ["Older", ["old"]],
    ]);
    expect(groupThreads([{ id: "x", updated_at: at(18) }], now).map(([l]) => l)).toEqual(["Today"]);
  });
});
