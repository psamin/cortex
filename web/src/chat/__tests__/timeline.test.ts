import { describe, expect, it } from "vitest";
import legacyRaw from "./fixtures/th_32e8be713381.jsonl?raw";
import v2Raw from "./fixtures/th_32e8be713381.v2.jsonl?raw";
import { applyEvent, historyToEvents, type ChatEvent, type ChatItem } from "../model";
import {
  buildTurns,
  endsOnFix,
  FALLBACK,
  FIX_BUSY_TIP,
  footerText,
  planText,
  STAGES,
  STATUS_WORD,
  statusLabel,
  stripIds,
  wakeText,
  type Turn,
} from "../timeline";

// The pasted chat (#236): stored thread th_32e8be713381, as stored (legacy) and with the backend's new fields (v2).
const rows = (raw: string) =>
  raw
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as Record<string, unknown>);
const itemsOf = (raw: string) => historyToEvents(rows(raw)).reduce(applyEvent, [] as ChatItem[]);
const turnsOf = (raw: string) => buildTurns(itemsOf(raw));

/** W1: words Cortex writes never carry ids, digests, reprs, backend keys, tool names or internal terms. */
const JARGON =
  /\b(prj|job|tj|ds|ev|ac|ck|ep|an|rec|rs|rr|run|th)_[0-9a-f]{6,}\b|\b[0-9a-f]{12}\b|\['|\{'|\w+\.\w+_\w+|\bcall [a-z]+_[a-z_]+|\b(sim_agent|sim_scripted|futurama_slurm|snapshot|segments|intake|episodes?|standing approval|planner budget|coverage cells?|gpu:)\b/i;
const ID = /\b(prj|job|tj|ds|ev|ac|ck|ep|an|rec|rs|rr|run|th)_[0-9a-f]{6,}\b/;

const HEADLINE_SCENE = "Built the scene: a water bottle to pick up and put on spot A. The arm can reach and grip it.";
const ASSUMED =
  "You didn't give a size, weight or drop-off spot, so Cortex used typical values: the water bottle is 4.5 cm wide, 14 cm tall and 240 g, and spot A is 22 cm in front of the arm and 12 cm to its left. Reply with the real values to rebuild the scene.";
const HEADLINE_TRAIN_FAILED =
  "Couldn't build the training set. No demos have been recorded in this scene yet. All 27 usable demos in this project were recorded before this scene was built, so they show a different setup.";
const NEXT = {
  text: "Collect demos in this scene first.",
  actions: [{ label: "Collect demos in this scene", message: "Collect demonstrations in the current scene, then build a training set from them and train." }],
};
const CORRECTION = "Cortex: this step had already failed when the agent wrote this. See Train above.";

/** Every string the timeline shows outside Details that Cortex (not the agent) wrote. */
function cortexStrings(turns: Turn[]): string[] {
  const out: string[] = [];
  for (const t of turns) {
    if (t.trigger?.kind === "wake") out.push(wakeText(t.trigger));
    for (const m of t.milestones) {
      out.push(STAGES.find((s) => s.id === m.stage)!.name, statusLabel(m), m.headline);
      if (m.assumed) out.push(m.assumed);
      if (m.next) out.push(m.next.text, ...m.next.actions.flatMap((a) => [a.label, a.message]));
      out.push(...m.details.steps.map((s) => s.label));
    }
    if (t.note?.correction) out.push(t.note.correction);
    if (t.footer) out.push(footerText(t.footer));
  }
  return out;
}

describe("timeline: the pasted chat with the backend's new fields (v2)", () => {
  const turns = turnsOf(v2Raw);
  const [t1, t2] = turns;
  const train1 = t1.milestones.find((m) => m.stage === "train")!;

  it("splits at your message and at the wake line", () => {
    expect(turns).toHaveLength(2);
    expect(t1.trigger).toEqual({ kind: "user", text: "make the robot arm pick up a water bottle" });
  });

  it("turn 1 is one scene line by Cortex and one failed train line, nothing for stages that did not run", () => {
    expect(t1.milestones).toMatchObject([
      { stage: "scene", status: "done", by: "cortex" },
      { stage: "train", status: "failed", by: null },
    ]);
    expect(t1.milestones.filter((m) => m.stage === "train")).toHaveLength(1);
    const scene = t1.milestones[0];
    expect(scene.headline).toBe(HEADLINE_SCENE);
    expect(scene.evidence).toBe("checked");
    expect(scene.assumed).toBe(ASSUMED);
    expect(scene.details.tools.map((t) => t.tool)).toEqual(["start_intake", "update_intake", "update_intake", "build_simulation"]);
  });

  it("the failed job, its tool and the read before it are one train line with Cortex's reason and fix", () => {
    expect(train1.details.jobIds).toContain("job_eee50b96409a");
    expect(train1.headline).toBe(HEADLINE_TRAIN_FAILED);
    expect(train1.next).toEqual(NEXT);
    expect(train1.next!.actions[0].label).toBe("Collect demos in this scene");
    expect(train1.laterSuccess).toBe(true);
    expect(t1.extras.some((x) => x.kind === "job_done")).toBe(false);
    expect(train1.details.tools.map((t) => t.tool)).toEqual(["get_project_state", "train_from_agent_demonstrations"]);
    expect(train1.details.steps).toEqual([
      { label: "Checked the project", headline: "Checked the project: 28 demos, 27 usable, none recorded in this scene yet. No training sets or tests yet.", status: "done" },
      {
        label: "Building a training set, then setting up training",
        headline: "Started building a training set from the demos the agent collected. Training gets set up once it's built.",
        status: "done",
      },
      { label: "Building the training set, then setting up training", headline: HEADLINE_TRAIN_FAILED, status: "failed" },
    ]);
    expect(train1.details.agentTexts).toEqual([
      "I'll start by loading the tools I need and checking the current project state.",
      expect.stringContaining("27 accepted agent demos across all five coverage cells"),
    ]);
  });

  it("the note is the agent's last message, without ids, corrected by Cortex, with the plan and the footer", () => {
    expect(t1.note!.raw).toContain("job_eee50b96409a");
    expect(t1.note!.text).not.toMatch(ID);
    expect(t1.note!.text).toContain("running as this step");
    expect(t1.note!.correction).toBe(CORRECTION);
    expect(t1.note!.plan!.filter((p) => p.status === "completed")).toHaveLength(1);
    expect(t1.note!.plan).toHaveLength(4);
    expect(planText(t1.note!.plan!, true)).toBe("1 of 4 done · stopped at: Build dataset and train ACT policy");
    expect(t1.footer).toMatchObject({ durationMs: 49007, failed: 1, isError: false });
    expect(footerText(t1.footer!)).toBe("Done in 49 s · 1 step failed");
  });

  it("turn 2 is a wake after the failed train line; child job, parent job, approval notice and cancel are one train line", () => {
    expect(t2.trigger).toEqual({ kind: "wake", after: { stage: "train", status: "failed" } });
    expect(wakeText(t2.trigger as Extract<Turn["trigger"], { kind: "wake" }>)).toBe("↻ Continuing on its own after Train failed");
    expect(t2.milestones).toHaveLength(1);
    const train2 = t2.milestones[0];
    expect(train2.stage).toBe("train");
    expect(train2.details.jobIds).toEqual(expect.arrayContaining(["job_a5642ec4d3f8", "job_08d8ddf6cfbe", "job_3d5a102671db"]));
    expect(train2.details.notices.map((n) => n.refIds)).toEqual([["tj_149f6e68b29a"]]);
    expect(train2.status).toBe("stopped");
    expect(train2.headline).toBe("Stopped.");
    expect(train2.details.tools.map((t) => t.tool)).toEqual(["list_episodes", "get_demonstration_results", "train_from_agent_demonstrations"]);
    expect(t2.note!.correction).toBeNull(); // nothing in this turn had failed when the agent wrote it
    expect(t2.footer).toMatchObject({ durationMs: 45837, failed: 0 });
  });

  it("the fix replaces the next-step chips only while no later train attempt worked", () => {
    expect(endsOnFix(turns)).toBe(false);
    // the moment the wake turn starts (rows 0-19): its fix is the next step
    const atWake = turnsOf(v2Raw.trim().split("\n").slice(0, 20).join("\n"));
    expect(atWake[0].milestones[1]).toMatchObject({ status: "failed", laterSuccess: false });
    expect(endsOnFix(atWake)).toBe(true);
  });

  it("no Cortex-written string carries jargon", () => {
    for (const s of cortexStrings(turns)) expect(s).not.toMatch(JARGON);
  });
});

describe("timeline: the same chat as stored today (legacy rows)", () => {
  const turns = turnsOf(legacyRaw);
  const [t1, t2] = turns;

  it("falls back to generic headlines, no by-tag, no fix, and still corrects the note", () => {
    expect(t1.milestones.map((m) => [m.stage, m.status, m.by])).toEqual([
      ["scene", "done", null],
      ["train", "failed", null],
    ]);
    expect(t1.milestones.map((m) => m.headline)).toEqual(["Scene built.", "This step failed. Open Details for the reason."]);
    expect(t1.milestones[0].assumed).toBeNull();
    expect(t1.milestones[1].next).toBeNull();
    expect(t1.milestones[1].details.jobDone[0].text).toContain("no accepted episodes to snapshot");
    expect(t1.note!.correction).toBe(CORRECTION);
    expect(footerText(t1.footer!)).toBe("Done in 49 s · 1 step failed");
  });

  it("the old wake line still starts a turn, without a reason", () => {
    expect(t2.trigger).toEqual({ kind: "wake" });
    expect(wakeText(t2.trigger as Extract<Turn["trigger"], { kind: "wake" }>)).toBe("↻ Continuing on its own");
    expect(t2.milestones.map((m) => [m.stage, m.status])).toEqual([["train", "stopped"]]);
  });

  it("no headline, fallback, footer or wake header carries jargon", () => {
    for (const s of cortexStrings(turns)) expect(s).not.toMatch(JARGON);
    for (const byStatus of Object.values(FALLBACK)) for (const s of Object.values(byStatus)) expect(s).not.toMatch(JARGON);
    for (const s of [...Object.values(STATUS_WORD), ...STAGES.map((x) => x.name), FIX_BUSY_TIP]) expect(s).not.toMatch(JARGON);
  });
});

describe("timeline rules", () => {
  const run = (...evs: ChatEvent[]) => buildTurns(evs.reduce(applyEvent, [] as ChatItem[]));
  const call = (tool: string): ChatEvent => ({ type: "tool_call", tool, args: {} });
  const result = (tool: string, ok = true, ref_ids: string[] = []): ChatEvent => ({ type: "tool_result", tool, summary: `${tool} raw`, ref_ids, ok });

  it("a failure then a success in the same line reads Done, after 1 failed attempt", () => {
    const [t] = run({ type: "user", text: "go" }, call("launch_training_job"), result("launch_training_job", false), call("launch_training_job"), result("launch_training_job"));
    expect(t.milestones).toHaveLength(1);
    expect(t.milestones[0]).toMatchObject({ status: "done", failedAttempts: 1 });
    expect(statusLabel(t.milestones[0])).toBe("Done, after 1 failed attempt");
  });

  it("reads and unknown tools never get a line; with no line they sit under the note", () => {
    const [t] = run({ type: "user", text: "status?" }, call("get_project_state"), result("get_project_state"), call("some_new_tool"), result("some_new_tool"), { type: "text", delta: "All good." });
    expect(t.milestones).toEqual([]);
    expect(t.reads.map((r) => r.tool)).toEqual(["get_project_state", "some_new_tool"]);
    expect(t.note).toMatchObject({ text: "All good.", correction: null });
  });

  it("a job that has not reported back keeps its line working; one that finishes after its turn updates it", () => {
    const evs: ChatEvent[] = [{ type: "user", text: "collect" }, call("start_agent_collection"), result("start_agent_collection", true, ["job_aaaaaaaaaaaa", "ac_bbbbbbbbbbbb"]), { type: "done", duration_ms: 1000 }];
    expect(run(...evs)[0].milestones[0]).toMatchObject({ stage: "demos", status: "working", pendingJobs: ["job_aaaaaaaaaaaa"] });
    const later = run(...evs, { type: "user", text: "next" }, { type: "job_done", job_id: "job_aaaaaaaaaaaa", job_kind: "agent_collect", status: "succeeded", text: "raw", ref_ids: ["ac_bbbbbbbbbbbb"] });
    expect(later[0].milestones[0]).toMatchObject({ status: "done", headline: "Demos collected.", details: { cardRefs: ["ac_bbbbbbbbbbbb"] } });
    expect(later[1].milestones).toEqual([]);
  });

  it("a pending approval makes the line need you", () => {
    const [t] = run({ type: "user", text: "train" }, call("propose_training_job"), result("propose_training_job", true, ["tj_cccccccccccc"]), {
      type: "approval_required",
      kind: "training",
      ref_id: "tj_cccccccccccc",
      summary: "raw",
      args: {},
    });
    expect(t.milestones).toHaveLength(1);
    expect(t.milestones[0]).toMatchObject({ stage: "train", status: "needs_you", headline: "Waiting for your OK." });
    expect(t.milestones[0].details.approvals).toHaveLength(1);
  });

  it("stripIds turns ids and scene digests into nouns", () => {
    expect(stripIds("running as job_eee50b96409a on ds_b44451be0bd9, scene 08639b5d15cf, 27 demos")).toBe(
      "running as this step on the training set, scene the scene, 27 demos",
    );
  });
});

describe("model: the end of a turn", () => {
  it("a tool still running when `done` arrives is stopped, not ok", () => {
    const items = [{ type: "user", text: "go" } as ChatEvent, { type: "tool_call", tool: "start_agent_collection", args: {} } as ChatEvent, { type: "done" } as ChatEvent].reduce(
      applyEvent,
      [] as ChatItem[],
    );
    expect(items.find((it) => it.kind === "tool")).toMatchObject({ status: "stopped" });
    expect(buildTurns(items)[0].milestones[0]).toMatchObject({ status: "stopped", headline: "Stopped." });
  });

  it("carries the contract fields from history rows", () => {
    const items = itemsOf(v2Raw);
    expect(items.find((it) => it.kind === "tool" && it.tool === "build_simulation")).toMatchObject({ by: "cortex", headline: HEADLINE_SCENE, evidence: "checked", assumed: ASSUMED });
    expect(items.find((it) => it.kind === "job_done" && it.jobId === "job_eee50b96409a")).toMatchObject({ headline: HEADLINE_TRAIN_FAILED, next: NEXT });
    expect(items.find((it) => it.kind === "notice" && it.wake)).toMatchObject({ refIds: ["job_eee50b96409a"] });
  });
});
