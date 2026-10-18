import type { ReactNode } from "react";
import { CUBE_REGIONS, LIGHTINGS, conditionKey, type Condition } from "../api/types";
import { cn } from "../lib/format";

/**
 * The workcell map: cube_region (columns, as seen from the camera) × lighting (rows).
 * Used for plan coverage, condition picking and failure-cluster footprints.
 */
export function WorkcellGrid({
  renderCell,
  selected,
  onSelect,
  className,
}: {
  renderCell: (c: Condition) => ReactNode;
  selected?: Condition | null;
  onSelect?: (c: Condition) => void;
  className?: string;
}) {
  return (
    <div className={cn("grid grid-cols-[58px_repeat(3,minmax(0,1fr))] gap-1", className)}>
      <div />
      {CUBE_REGIONS.map((r) => (
        <div key={r} className="pb-0.5 text-center text-2xs font-medium capitalize text-ink-3">
          {r}
        </div>
      ))}
      {LIGHTINGS.map((l) => (
        <Row key={l} lighting={l}>
          {CUBE_REGIONS.map((r) => {
            const c: Condition = { cube_region: r, lighting: l };
            const isSel = selected && conditionKey(selected) === conditionKey(c);
            const content = renderCell(c);
            return onSelect ? (
              <button
                key={r}
                type="button"
                onClick={() => onSelect(c)}
                aria-pressed={!!isSel}
                aria-label={`${r}, ${l} lighting`}
                className={cn(
                  "min-w-0 rounded-sm text-left transition-[border-color,background-color] duration-100",
                  isSel
                    ? "bg-servo/[.07] border border-servo"
                    : "bg-plate border border-line hover:border-line-strong",
                )}
              >
                {content}
              </button>
            ) : (
              <div key={r} className="min-w-0 rounded bg-well border border-line">
                {content}
              </div>
            );
          })}
        </Row>
      ))}
    </div>
  );
}

function Row({ lighting, children }: { lighting: string; children: ReactNode }) {
  return (
    <>
      <div className="flex items-center gap-1.5 text-2xs font-medium capitalize text-ink-3">
        <LightGlyph lighting={lighting} />
        {lighting}
      </div>
      {children}
    </>
  );
}

/** Three-step lamp glyph: dim / normal / bright. */
export function LightGlyph({ lighting }: { lighting: string }) {
  const level = lighting === "dim" ? 1 : lighting === "normal" ? 2 : 3;
  return (
    <span aria-hidden className="inline-flex items-end gap-px">
      {[1, 2, 3].map((i) => (
        <span key={i} className={cn("w-[3px] rounded-[1px]", i <= level ? "bg-servo/80" : "bg-ink-4/40")} style={{ height: 3 + i * 2 }} />
      ))}
    </span>
  );
}

/** Tiny 3×3 footprint showing which conditions are touched (e.g. by a failure cluster). */
export function Footprint({ keys, tone = "servo" }: { keys: string[]; tone?: "servo" | "fault" }) {
  const set = new Set(keys);
  return (
    <span className="inline-grid grid-cols-3 gap-[2px]" title={keys.join(", ") || "no conditions"} aria-label={`conditions: ${keys.join(", ")}`}>
      {LIGHTINGS.map((l) =>
        CUBE_REGIONS.map((r) => (
          <span
            key={`${r}/${l}`}
            className={cn(
              "h-[6px] w-[6px] rounded-[1px]",
              set.has(`${r}/${l}`) ? (tone === "fault" ? "bg-fault" : "bg-servo") : "bg-ink-4/30",
            )}
          />
        )),
      )}
    </span>
  );
}

/** "cube left, dim light": the condition in plain words. */
export const conditionWords = (c: Condition, sep = ", ") => `cube ${c.cube_region}${sep}${c.lighting} light`;

export function ConditionLabel({ c, className }: { c: Condition; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap text-xs text-ink-2", className)}>
      <LightGlyph lighting={c.lighting} />
      {conditionWords(c, " · ")}
    </span>
  );
}
