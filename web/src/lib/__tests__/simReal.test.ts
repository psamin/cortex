import { describe, expect, it } from "vitest";
import { agreementWord, gapText, rateText } from "../simReal";

describe("sim vs real wording", () => {
  it("shows successes, rate and interval in one line", () => {
    expect(rateText({ n: 10, successes: 6, rate: 0.6, ci95: [0.313, 0.832] })).toBe("6/10 · 60% (31–83)");
    expect(rateText({ n: 0, successes: 0, rate: null, ci95: [0, 1] })).toBe("—");
  });

  it("signs the gap in points", () => {
    expect(gapText(-0.2)).toBe("−20 pts");
    expect(gapText(0.05)).toBe("+5 pts");
    expect(gapText(null)).toBe("—");
  });

  it("puts rank agreement in words", () => {
    expect(agreementWord(0.9)).toBe("yes (ρ 0.90)");
    expect(agreementWord(0.4)).toBe("partly (ρ 0.40)");
    expect(agreementWord(-0.2)).toBe("no (ρ −0.20)");
  });
});
