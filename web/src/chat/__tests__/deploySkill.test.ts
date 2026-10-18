import { describe, expect, it } from "vitest";
import type { AgentCall } from "../../agents/api";
import { deploymentPath, deployments, skillEvents, skillStages } from "../deploySkill";

const call = (tool: string, result: unknown, status: AgentCall["status"] = "ok", at = "2026-10-18T15:00:00Z"): AgentCall => ({
  id: `${tool}-${at}`,
  tool,
  args: {},
  status,
  started_at: at,
  ended_at: at,
  result_preview: result === undefined ? undefined : JSON.stringify(result),
});

const saved = call("save_physical_skill", { skill_id: "sk_tj1", name: "Jenga v1", saved: true, score: { label: "62% success on 50 unseen sim layouts" } });
const starting = call("deploy_physical_skill", { deployment_url: "http://127.0.0.1:5190/studio/skills/sk_tj1", run_id: null, skill_id: "sk_tj1", skill_name: "Jenga v1", deployed: false }, "ok", "2026-10-18T15:01:00Z");
const live = call("deploy_physical_skill", { deployment_url: "http://127.0.0.1:5190/studio/skills/sk_tj1?seed=123456", run_id: "live_sim/jenga/123456", seed: 123456, skill_id: "sk_tj1", skill_name: "Jenga v1", deployed: true }, "ok", "2026-10-18T15:05:00Z");

describe("the Save skill and Deploy stages (#310)", () => {
  it("wait until the agent saves and deploys", () => {
    expect(skillStages([]).map((s) => [s.name, s.state])).toEqual([["Save skill", "waiting"], ["Deploy", "waiting"]]);
  });

  it("follow the save, the GPU coming up, then the live run", () => {
    expect(skillStages([saved]).map((s) => s.state)).toEqual(["done", "waiting"]);
    expect(skillStages([saved])[0].note).toBe("Jenga v1");
    expect(skillStages([saved, starting])[1]).toMatchObject({ state: "running", note: "GPU starting" });
    expect(skillStages([saved, starting, live])[1]).toMatchObject({ state: "done", note: "live run ready" });
  });

  it("ignore failed calls and calls without a result", () => {
    expect(deployments([call("deploy_physical_skill", { error: "x" }, "error"), call("deploy_physical_skill", undefined)])).toEqual([]);
    expect(skillStages([call("save_physical_skill", {}, "error")])[0].state).toBe("waiting");
  });
});

describe("milestones and the View deployment link", () => {
  it("say what was saved, with its measured score, and what is deploying", () => {
    expect(skillEvents([saved, live]).map((e) => e.text)).toEqual([
      "Saved physical skill 'Jenga v1' (62% success on 50 unseen sim layouts)",
      "Deploying physical skill 'Jenga v1' on unseen layout 123456…",
    ]);
  });

  it("open the deployment simulator page in the studio", () => {
    const d = deployments([saved, starting, live]).pop()!;
    expect(d).toMatchObject({ deployed: true, runId: "live_sim/jenga/123456", seed: 123456 });
    expect(deploymentPath(d.url)).toBe("/studio/skills/sk_tj1?seed=123456");
  });
});
