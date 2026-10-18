import type { LivePolicyStatus } from "../api/types";

/** What the Inference tab plays: a sim-trained policy on a new layout, or a policy on a held-out real episode. */
export type Pick = { kind: "sim"; task: string; seed: number } | { kind: "replay"; task: string; ep: number };

export const SIM_TITLE: Record<string, string> = {
  jenga: "Jenga", screwdriver: "Screwdriver", unscrew: "Unscrew (grip)", palletize: "Palletizing", suture: "Suture",
};
// which sim task a project is about, from its name and prompt; earlier entries win ("Screwdriver" before "Unscrew").
// No "box": cube-in-a-box projects are not palletizing.
const SIM_MATCH: [string, RegExp][] = [
  ["jenga", /jenga/i], ["screwdriver", /screwdriver/i], ["palletize", /pallet|forklift/i],
  ["unscrew", /unscrew|bolt/i], ["suture", /sutur/i],
];

const title = (p: Pick) =>
  p.kind === "sim" ? `${SIM_TITLE[p.task] ?? p.task} (sim)` : `${p.task} (real held-out episodes)`;

/** What the tab opens on in project `pid`, and a note when that isn't the project's own policy.
 *  The project's own task comes from the server's `projects` map (infra/futurama/projects.json and the sim
 *  importer's project names), else from words in `hint` (the project's name and prompt). */
export function initialPick(s: LivePolicyStatus, pid: string, hint: string): { pick: Pick | null; note: string | null } {
  const key = Object.entries(s.projects ?? {}).find(([, p]) => p === pid)?.[0];
  const ownSim = key ? (key.startsWith("sim_") ? key.slice(4) : undefined) : SIM_MATCH.find(([, re]) => re.test(hint))?.[0];
  const ownReal = key && !key.startsWith("sim_") ? key : undefined;
  const sim = (task: string): Pick => ({ kind: "sim", task, seed: 100_000 });
  const replay = (task: string): Pick => ({ kind: "replay", task, ep: s.policies![task].held_out[0] });
  const hasReplay = (task?: string) => !!task && !!s.policies?.[task]?.held_out.length;
  if (hasReplay(ownReal)) return { pick: replay(ownReal!), note: null };
  if (ownSim && s.sim?.[ownSim]) return { pick: sim(ownSim), note: null };
  const firstSim = Object.keys(s.sim ?? {})[0];
  const firstReal = Object.keys(s.policies ?? {}).find(hasReplay);
  const pick = firstSim ? sim(firstSim) : firstReal ? replay(firstReal) : null;
  if (!pick) return { pick, note: "The GPU server is up, but it has no trained policy loaded yet." };
  const own = ownReal ?? ownSim;
  const ownTitle = own ? (ownSim ? `${SIM_TITLE[own] ?? own} (sim)` : own) : null;
  return {
    pick,
    note: ownTitle
      ? `No ${ownTitle} policy on the GPU server yet. Showing ${title(pick)}.`
      : `This project has no policy on the GPU server. Showing ${title(pick)}; pick another above.`,
  };
}
