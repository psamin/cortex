import type { LivePolicyStatus, PhysicalSkill, SkillScore } from "../api/types";
import type { Tone } from "../components/ui";
import { ago } from "../lib/format";

/** The card's score badge: the simulator's verdict, or plainly pending. */
export function scoreBadge(s: SkillScore): { text: string; tone: Tone } {
  if (s.kind === "sim_success" && s.value != null) {
    return { text: `${Math.round(s.value * 100)}% success`, tone: s.value >= 0.8 ? "pass" : s.value >= 0.5 ? "caution" : "fault" };
  }
  if (s.kind === "joint_error" && s.value != null) return { text: `${s.value.toFixed(2)}° joint error`, tone: "neutral" };
  if (s.kind === "pending") return { text: "Score pending", tone: "servo" };
  return { text: "Not measured yet", tone: "neutral" };
}

export type RunAction = { kind: "play" } | { kind: "request" } | { kind: "starting" } | { kind: "checking" } | { kind: "unavailable"; note: string };

/** What "Run live" can do now. Uses the GPU server's own status (polled), so it is right the moment the server
 *  comes up, even before the skills list is read again. A GPU is only ever requested by an explicit click. */
export function runAction(s: PhysicalSkill, gpu: LivePolicyStatus | undefined): RunAction {
  const task = s.run.task;
  if (s.run.kind !== "sim" || !task) return { kind: "unavailable", note: s.run.note ?? "This policy can't run live yet." };
  if (!gpu) return { kind: "checking" };
  if (!gpu.available) return { kind: gpu.phase === "starting" ? "starting" : "request" };
  const served = gpu.sim?.[task];
  if (served?.variant === s.variant) return { kind: "play" };
  return {
    kind: "unavailable",
    note: served ? `The GPU server is running the ${served.variant.toUpperCase()} policy for this task.` : "The GPU server is up but doesn't have this policy loaded.",
  };
}

/** What the score was measured on, from the backend's label: "on 20 unseen sim layouts", "on 4 runs over 3 practice
 *  layouts it trained on", "on 6 held-out real demos". Null when nothing was measured yet. */
export function scoreWhere(s: SkillScore): string | null {
  if (s.kind !== "sim_success" && s.kind !== "joint_error") return null;
  return s.label.replace(/^\S+ (success|mean joint error) /, "") || null;
}

/** Where a sim run plays: a new layout, or (practice-layout policies) one of the layouts it trained on. */
export function simWhere(s: PhysicalSkill): string {
  return s.layouts ? `one of the ${s.layouts} practice layouts it trained on` : "a new layout it has never seen";
}

/** "Trained on Futurama H200 · 2h ago" */
export function trainedLine(s: PhysicalSkill): string {
  if (!s.trained_where && !s.trained_at) return "";
  return [s.trained_where ? `Trained on ${s.trained_where}` : "Trained", s.trained_at ? ago(s.trained_at) : null].filter(Boolean).join(" · ");
}

/** A layout seed for a live run. The GPU server maps it onto a practice layout when the policy was served on those. */
export const newSeed = () => 100_000 + Math.floor(Math.random() * 900_000);

export type SkillVideo = { url: string; label: string; success?: boolean | null };

/** Every recorded video of a skill (#308): its replays (newest first), then past live runs, then the episodes behind
 *  its score. */
export function skillVideos(s: PhysicalSkill): SkillVideo[] {
  const name = (u: string) => decodeURIComponent(u.split("/").pop() ?? "").replace(/\.mp4$/i, "");
  return [
    ...(s.replays ?? []).map((u) => ({ url: u, label: name(u) })),
    ...s.runs.flatMap((r) => (r.video_url ? [{ url: r.video_url, label: `${r.label} · ${ago(r.when)}`, success: r.success }] : [])),
    ...s.evidence.flatMap((e) => (e.video_url ? [{ url: e.video_url, label: e.label, success: e.success }] : [])),
  ];
}
