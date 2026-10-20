import { describe, expect, it } from "vitest";
import { VERDICT, episodesLine, findingTitle } from "../readiness";

describe("readiness wording", () => {
  it("names every verdict and finding plainly", () => {
    expect(VERDICT.blocked).toEqual({ label: "Blocked", tone: "fault" });
    expect(findingTitle({ code: "unit_mismatch" })).toBe("Mixed units");
    expect(findingTitle({ code: "something_new" })).toBe("something new");
  });

  it("keeps the episode list to one line", () => {
    expect(episodesLine([])).toBe("");
    expect(episodesLine(["ep_1", "ep_2"])).toBe("ep_1, ep_2");
    expect(episodesLine(["ep_1", "ep_2", "ep_3", "ep_4"])).toBe("ep_1, ep_2 +2");
  });
});
