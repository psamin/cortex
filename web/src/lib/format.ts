export const cn = (...xs: (string | false | null | undefined)[]) => xs.filter(Boolean).join(" ");

export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);

export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 60) return `${Math.max(s, 0)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export const usd = (x: number) => (x === 0 ? "$0" : `$${x.toFixed(x < 10 ? 2 : 0)}`);

/** Render an evidence value compactly: numbers to 3 sig figs, others JSON. */
export function evidenceValue(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : Number(v.toPrecision(3)).toString();
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

export const humanize = (s: string) => s.replace(/_/g, " ");

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
