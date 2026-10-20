import type { TrialRate } from "../api/types";

/** "6/10 · 60% (31–83)": successes, rate and the Wilson 95% interval, in one line. */
export function rateText(r: TrialRate): string {
  if (!r.n || r.rate == null) return "—";
  const [lo, hi] = r.ci95.map((x) => Math.round(x * 100));
  return `${r.successes}/${r.n} · ${Math.round(r.rate * 100)}% (${lo}–${hi})`;
}

/** Real minus sim, in points: negative means the real arm does worse than sim predicted. */
export function gapText(gap: number | null): string {
  if (gap == null) return "—";
  const pts = Math.round(gap * 100);
  return pts === 0 ? "0 pts" : `${pts > 0 ? "+" : "−"}${Math.abs(pts)} pts`;
}

/** Spearman of sim vs real success across checkpoints, in words. */
export function agreementWord(rho: number): string {
  const s = rho.toFixed(2).replace("-", "−");
  return rho >= 0.7 ? `yes (ρ ${s})` : rho >= 0.3 ? `partly (ρ ${s})` : `no (ρ ${s})`;
}
