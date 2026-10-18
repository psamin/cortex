import { describe, expect, it } from "vitest";
import type { LivePolicyStatus, PhysicalSkill, SkillScore } from "../../api/types";
import { runAction, scoreBadge, scoreWhere, simWhere, skillVideos, trainedLine } from "../skillView";

const score = (s: Partial<SkillScore>): SkillScore => ({ kind: "none", value: null, n: null, label: "", detail: "", ...s });

const skill = (s: Partial<PhysicalSkill> = {}): PhysicalSkill => ({
  id: "sk_tj_1", name: "Unscrew a bolt with the gripper", task: "unscrew", data: "sim", instruction: "", robot: "SO-101",
  project_id: "prj_1", project_name: "Unscrew bolt (sim)", policy_type: "act", policy: "ACT", variant: "act",
  steps: 30000, layouts: null, trained_where: "Futurama H200", trained_at: null, weights: "on the cluster", checkpoint_id: "ck_1",
  checkpoint: "030000", training_job_id: "tj_1", dataset_version_id: null, slurm_job: null,
  score: score({}), run: { kind: "sim", task: "unscrew", episodes: [], live_now: false, note: null },
  runs: [], evidence: [], thumbnail_url: null, replays: [], live: false, ...s,
});

const gpu = (s: Partial<LivePolicyStatus>): LivePolicyStatus => ({ available: false, phase: "stopped", ...s });
const up = (variant: string) =>
  gpu({ available: true, phase: "ready", sim: { unscrew: { variant, step: 30000, success_rate: 0.8, eval_episodes: 50, eval_seeds: "", instruction: "" } } });

describe("scoreBadge", () => {
  it("names a sim success rate and colours it by how good it is", () => {
    expect(scoreBadge(score({ kind: "sim_success", value: 0.9, n: 50 }))).toEqual({ text: "90% success", tone: "pass" });
    expect(scoreBadge(score({ kind: "sim_success", value: 0.6, n: 50 })).tone).toBe("caution");
    expect(scoreBadge(score({ kind: "sim_success", value: 0.2, n: 50 })).tone).toBe("fault");
  });
  it("says pending and not measured plainly", () => {
    expect(scoreBadge(score({ kind: "pending" })).text).toBe("Score pending");
    expect(scoreBadge(score({ kind: "none" })).text).toBe("Not measured yet");
  });
});

describe("runAction", () => {
  it("plays when the GPU server runs this policy, even if the list was read before it came up", () => {
    expect(runAction(skill(), up("act"))).toEqual({ kind: "play" });
  });
  it("offers one explicit GPU request when the server is down, never more", () => {
    expect(runAction(skill(), gpu({})).kind).toBe("request");
    expect(runAction(skill(), gpu({ phase: "starting" })).kind).toBe("starting");
    expect(runAction(skill(), undefined).kind).toBe("checking");
  });
  it("explains why it can't run when the server runs something else", () => {
    expect(runAction(skill(), up("smolvla"))).toEqual({ kind: "unavailable", note: "The GPU server is running the SMOLVLA policy for this task." });
    expect(runAction(skill(), gpu({ available: true, phase: "ready", sim: {} }))).toEqual({
      kind: "unavailable",
      note: "The GPU server is up but doesn't have this policy loaded.",
    });
  });
  it("can't run a policy no GPU server serves", () => {
    const s = skill({ run: { kind: null, task: null, episodes: [], live_now: false, note: "not on the server" } });
    expect(runAction(s, gpu({}))).toEqual({ kind: "unavailable", note: "not on the server" });
  });
});

describe("where it was measured and where it runs", () => {
  it("keeps the backend's wording, practice layouts included", () => {
    expect(scoreWhere(score({ kind: "sim_success", value: 0.85, label: "85% success on 20 unseen sim layouts" }))).toBe("on 20 unseen sim layouts");
    expect(scoreWhere(score({ kind: "sim_success", value: 0.75, label: "75% success on 4 runs over 3 practice layouts it trained on" }))).toBe(
      "on 4 runs over 3 practice layouts it trained on",
    );
    expect(scoreWhere(score({ kind: "joint_error", value: 0.12, label: "0.12° mean joint error on 6 held-out real demos" }))).toBe("on 6 held-out real demos");
    expect(scoreWhere(score({ kind: "pending", label: "Score pending" }))).toBeNull();
  });
  it("never promises a new layout to a practice-layout policy", () => {
    expect(simWhere(skill({ layouts: 3 }))).toBe("one of the 3 practice layouts it trained on");
    expect(simWhere(skill({ layouts: null }))).toBe("a new layout it has never seen");
  });
});

describe("trainedLine", () => {
  it("says where and when, in plain words", () => {
    expect(trainedLine(skill({ trained_at: null }))).toBe("Trained on Futurama H200");
    expect(trainedLine(skill({ trained_where: null, trained_at: null }))).toBe("");
    expect(trainedLine(skill({ trained_at: new Date(Date.now() - 2 * 3600_000).toISOString() }))).toBe("Trained on Futurama H200 · 2h ago");
  });
});

describe("skillVideos", () => {
  it("lists replays first (named for their file), then past live runs and score episodes that kept a video", () => {
    const v = skillVideos(
      skill({
        replays: ["/api/skills/unscrew/replays/run%202.mp4", "/api/skills/unscrew/replays/a.mp4"],
        runs: [
          { id: "r1", kind: "live_sim", label: "Live sim run", when: new Date().toISOString(), partial: false, success: true, joint_mae_deg: null, steps: 9, inference_ms: null, gpu: null, video_url: "/files/r1.mp4", note: "" },
          { id: "r2", kind: "live_sim", label: "Live sim run", when: new Date().toISOString(), partial: true, success: null, joint_mae_deg: null, steps: 3, inference_ms: null, gpu: null, video_url: null, note: "" },
        ],
        evidence: [{ label: "Unseen layout 100000", success: false, video_url: "/files/e.mp4" }],
      }),
    );
    expect(v.map((x) => x.url)).toEqual(["/api/skills/unscrew/replays/run%202.mp4", "/api/skills/unscrew/replays/a.mp4", "/files/r1.mp4", "/files/e.mp4"]);
    expect(v[0].label).toBe("run 2");
    expect(v[2].success).toBe(true);
    expect(v[3]).toMatchObject({ label: "Unseen layout 100000", success: false });
  });
  it("copes with a backend older than #308, which sends no replays", () => {
    const old = skill({ evidence: [{ label: "Unseen layout 1", success: true, video_url: "/files/e.mp4" }] });
    delete old.replays;
    delete old.live;
    expect(skillVideos(old).map((x) => x.url)).toEqual(["/files/e.mp4"]);
  });
});
