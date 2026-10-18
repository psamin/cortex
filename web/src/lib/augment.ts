import { agentLabel, isImported, type Augmentation, type DatasetVersion, type Episode } from "../api/types";

export const AUG_NOTE = "Augmented copies change only pixels and are used for training only.";

/** "brightness ×1.3", "color temperature +0.4", "shadow 0.45" */
export function augWords(a: Augmentation): string {
  const p = a.params ?? {};
  switch (a.type) {
    case "brightness":
      return `brightness ×${p.factor ?? "?"}`;
    case "contrast":
      return `contrast ×${p.factor ?? "?"}`;
    case "color_temperature": {
      const s = p.shift ?? 0;
      return `color temperature ${s >= 0 ? "+" : ""}${s}`;
    }
    case "shadow":
      return `shadow ${p.strength ?? ""}`.trim();
    default:
      return String(a.type).replace(/_/g, " ");
  }
}

/** Who recorded a source demo, short form. */
export function whoShort(e?: Episode): string {
  if (!e) return "unknown";
  if (isImported(e)) return "human (imported)";
  if (e.source === "real_teleop") return "human (teleop)";
  if (e.source === "sim_scripted") return "oracle";
  return agentLabel(e.provenance?.planner).replace(/ agent$/, "");
}

const tally = (xs: string[]) =>
  [...xs.reduce((m, x) => m.set(x, (m.get(x) ?? 0) + 1), new Map<string, number>())].sort((a, b) => b[1] - a[1]);
const fmt = (t: [string, number][]) => t.map(([k, n]) => `${k} ${n}`).join(" · ");

/** "12 source demos (Claude Code 4 · oracle 8) + 24 augmented copies (brightness 12 · shadow 12) · 10 train / 2 held out" */
export function datasetBreakdown(d: DatasetVersion, eps: Map<string, Episode>): { text: string; hasAug: boolean } {
  const src = d.links.filter((l) => !l.augmentation);
  const aug = d.links.filter((l) => l.augmentation);
  const ids = [...new Set(src.map((l) => l.episode_id))];
  const who = tally(ids.map((id) => whoShort(eps.get(id))));
  const types = tally(aug.map((l) => l.augmentation!.type.replace(/_/g, " ")));
  const train = src.filter((l) => l.split === "train").length;
  const sources = `${ids.length} source demo${ids.length === 1 ? "" : "s"}${who.length ? ` (${fmt(who)})` : ""}`;
  const copies = aug.length ? ` + ${aug.length} augmented ${aug.length === 1 ? "copy" : "copies"} (${fmt(types)})` : "";
  return { text: `${sources}${copies} · ${train} train / ${src.length - train} held out`, hasAug: aug.length > 0 };
}

/** episode_id → its augmented copies across dataset versions, with a readable tooltip line per version. */
export function augmentedByEpisode(ds: DatasetVersion[]): Map<string, { n: number; tip: string }> {
  const out = new Map<string, { n: number; lines: string[] }>();
  for (const d of ds) {
    const byEp = new Map<string, Augmentation[]>();
    for (const l of d.links) if (l.augmentation) byEp.set(l.episode_id, [...(byEp.get(l.episode_id) ?? []), l.augmentation]);
    for (const [id, augs] of byEp) {
      const cur = out.get(id) ?? { n: 0, lines: [] };
      cur.n += augs.length;
      cur.lines.push(`${augs.map(augWords).join(", ")} · dataset ${d.name} · training only`);
      out.set(id, cur);
    }
  }
  return new Map([...out].map(([k, v]) => [k, { n: v.n, tip: v.lines.join("\n") }]));
}
