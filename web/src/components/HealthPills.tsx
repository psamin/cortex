import type { ProviderInfo } from "../api/types";

/** Provider name + whether it is a local fallback (from /health's explicit `fallback` flag). */
export function describeProvider(p: ProviderInfo | string | null | undefined): { name: string; fallback: boolean; model?: string } {
  if (!p) return { name: "unknown", fallback: true };
  if (typeof p === "string") return { name: p, fallback: /fallback|local|disabled/i.test(p) };
  return { name: p.provider, fallback: !!p.fallback, model: p.model ?? undefined };
}
