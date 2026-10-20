import { describe, expect, it } from "vitest";
import { canRetry, failureCause } from "../trainingFailure";

describe("training failure wording (#348)", () => {
  it("names the cause of a failed job and never shows it for other states", () => {
    expect(failureCause({ status: "failed", failure_kind: "cluster_unreachable" })?.short).toBe("couldn't reach the cluster");
    expect(failureCause({ status: "failed", failure_kind: null })?.long).toMatch(/Training failed/);
    expect(failureCause({ status: "failed", failure_kind: "something_new" })?.short).toBe("failed");
    expect(failureCause({ status: "cancelled", failure_kind: null })).toBeNull();
    expect(failureCause({ status: "running" })).toBeNull();
  });

  it("offers Retry only on an approved job that failed or was stopped", () => {
    expect(canRetry({ status: "failed", approved_by: "ana" })).toBe(true);
    expect(canRetry({ status: "cancelled", approved_by: "ana" })).toBe(true);
    expect(canRetry({ status: "failed", approved_by: null })).toBe(false);
    expect(canRetry({ status: "succeeded", approved_by: "ana" })).toBe(false);
  });
});
