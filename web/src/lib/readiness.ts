import type { ReadinessFinding, ReadinessVerdict } from "../api/types";

/** Studio words for the readiness check (docs/STACK_SPEC.md 2.1). */
export const VERDICT: Record<ReadinessVerdict, { label: string; tone: "pass" | "caution" | "fault" }> = {
  ready: { label: "Ready", tone: "pass" },
  ready_with_warnings: { label: "Warnings", tone: "caution" },
  blocked: { label: "Blocked", tone: "fault" },
};

const TITLE: Record<string, string> = {
  idle_head: "Idle start",
  idle_tail: "Idle end",
  gripper_unused: "Gripper never moves",
  action_jump: "Glitch step",
  action_state_lag: "Actions out of sync",
  fps_mismatch: "Mixed frame rates",
  shape_mismatch: "Mixed action sizes",
  unit_mismatch: "Mixed units",
  length_outlier: "Unusual length",
  few_demos: "Thin conditions",
  mostly_idle_data: "Mostly idle",
  unreadable: "Unreadable file",
  no_episodes: "No episodes",
};

export const findingTitle = (f: Pick<ReadinessFinding, "code">) => TITLE[f.code] ?? f.code.replace(/_/g, " ");

/** "ep_1, ep_2 +3": the episodes a finding is about, kept to one line. */
export function episodesLine(ids: string[], max = 2): string {
  if (!ids.length) return "";
  return ids.slice(0, max).join(", ") + (ids.length > max ? ` +${ids.length - max}` : "");
}
