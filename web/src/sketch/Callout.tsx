import type { CSSProperties, ReactNode } from "react";

/**
 * A technical-drawing callout: an anchor node, a leader line (diagonal then horizontal run),
 * and a mono label. Place it inside a `relative` parent; (x, y) is the anchor in that parent,
 * as px numbers or CSS lengths ("50%"). dx/dy/run are px, as in the hero's overlaySpec.
 */
export function Callout({
  x,
  y,
  dx = 40,
  dy = -32,
  run = 28,
  label,
  value,
  accent,
  className,
  style,
}: {
  x: number | string;
  y: number | string;
  dx?: number;
  dy?: number;
  run?: number;
  label: ReactNode;
  value?: ReactNode;
  accent?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  const dir = dx >= 0 ? 1 : -1;
  const ex = dx + run * dir;
  return (
    <div
      className={["sk-callout pointer-events-none absolute", accent ? "sk-accent" : "", className].filter(Boolean).join(" ")}
      style={{ left: x, top: y, ...style }}
      aria-hidden
    >
      <svg className="sk-svg absolute left-0 top-0 overflow-visible" width="1" height="1">
        <path d={`M0 0 L${dx} ${dy} L${ex} ${dy}`} fill="none" className="sk-leader" />
        <circle r="2.6" cx="0" cy="0" className="sk-node" />
      </svg>
      <div
        className="sk-callout-label absolute whitespace-nowrap"
        style={{ left: ex + 4 * dir, top: dy, transform: `translate(${dir < 0 ? "-100%" : "0"}, -50%)` }}
      >
        <span>{label}</span>
        {value != null && <span className="sk-callout-value">{value}</span>}
      </div>
    </div>
  );
}
