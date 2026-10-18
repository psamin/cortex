import { describe, expect, it } from "vitest";
import type { LivePolicyStatus } from "../../api/types";
import { initialPick } from "../livePick";

const sim = { variant: "act", step: 30000, success_rate: 0.9, eval_episodes: 50, eval_seeds: "100000..100049", instruction: "x" };
const server: LivePolicyStatus = {
  available: true,
  phase: "ready",
  policies: {
    jenga: { checkpoint: "jenga_010000", stage: "jenga", held_out: [58, 59] },
    conveyor: { checkpoint: "conveyor_008000", stage: "conveyor", held_out: [97] },
  },
  sim: { jenga: sim, screwdriver: sim },
  projects: { jenga: "prj_real_jenga", conveyor: "prj_conveyor", sim_jenga: "prj_sim_jenga" },
};

describe("initialPick", () => {
  it("opens a real-data project on its own held-out replay, not on a sim run", () => {
    expect(initialPick(server, "prj_conveyor", "Conveyor sorting").pick).toEqual({ kind: "replay", task: "conveyor", ep: 97 });
    expect(initialPick(server, "prj_real_jenga", "Pull a block out of a Jenga tower").pick).toEqual({ kind: "replay", task: "jenga", ep: 58 });
  });

  it("opens a sim project on its sim policy, from the projects map or its name", () => {
    expect(initialPick(server, "prj_sim_jenga", "").pick).toEqual({ kind: "sim", task: "jenga", seed: 100_000 });
    // imported projects' goal used to be the cube task; the name still says what it is
    const r = initialPick(server, "prj_x", "Screwdriver (sim) Pick up a screwdriver and unscrew a screw with it (MuJoCo sim)");
    expect(r).toEqual({ pick: { kind: "sim", task: "screwdriver", seed: 100_000 }, note: null });
  });

  it("says so when the project's own policy is not on the server", () => {
    const r = initialPick(server, "prj_pallet", "Palletizing (sim) Move boxes off a forklift onto a pallet");
    expect(r.pick).toEqual({ kind: "sim", task: "jenga", seed: 100_000 });
    expect(r.note).toBe("No Palletizing (sim) policy on the GPU server yet. Showing Jenga (sim).");
  });

  it("does not take a cube-in-a-box project for palletizing", () => {
    const r = initialPick(server, "prj_cube", "Pick up red cube and put it in the box");
    expect(r.note).toMatch(/^This project has no policy on the GPU server/);
  });

  it("explains an empty server", () => {
    const r = initialPick({ available: true, phase: "ready", policies: {}, sim: {} }, "prj_x", "");
    expect(r).toEqual({ pick: null, note: "The GPU server is up, but it has no trained policy loaded yet." });
  });
});
