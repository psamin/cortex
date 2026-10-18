const GROUPS = ["Today", "Yesterday", "Previous 7 days", "Older"] as const;
const DAY = 86_400_000;

/** ChatGPT-style buckets by last activity (local calendar days), newest first; empty buckets are dropped. */
export function groupThreads<T extends { updated_at: string }>(threads: T[], now = new Date()): [string, T[]][] {
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const at = (t: T) => new Date(t.updated_at).getTime();
  const out: [string, T[]][] = GROUPS.map((g) => [g, []]);
  for (const t of [...threads].sort((a, b) => at(b) - at(a))) {
    const ms = at(t);
    out[ms >= today ? 0 : ms >= today - DAY ? 1 : ms >= today - 7 * DAY ? 2 : 3][1].push(t);
  }
  return out.filter(([, xs]) => xs.length > 0);
}
