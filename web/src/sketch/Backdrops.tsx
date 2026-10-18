import { ISO_ELLIPSE } from "./iso";

/**
 * Faint isometric grid (lines at 30°, 150° and vertical), the engineering floor of the hero.
 * Fills its nearest positioned ancestor; `fade` softens it towards the edges.
 */
export function IsoGrid({ cell = 28, fade = true, className }: { cell?: number; fade?: boolean; className?: string }) {
  return (
    <div
      aria-hidden
      className={["sk-isogrid pointer-events-none absolute inset-0", fade ? "sk-fade" : "", className].filter(Boolean).join(" ")}
      style={{ ["--sk-cell" as string]: `${cell}px` }}
    />
  );
}

/**
 * Dashed construction orbit: an isometric floor circle (or a flat circle) with yaw ticks,
 * like the hero's reach circle. Centred in its positioned ancestor unless you pass a className.
 */
export function DashedOrbit({
  r = 120,
  flat = false,
  ticks = 24,
  axes = true,
  className,
}: {
  r?: number;
  flat?: boolean;
  ticks?: number;
  axes?: boolean;
  className?: string;
}) {
  const rx = flat ? r : r * ISO_ELLIPSE.rx;
  const ry = flat ? r : r * ISO_ELLIPSE.ry;
  const pad = 14;
  const W = rx * 2 + pad * 2;
  const H = ry * 2 + pad * 2;
  const cx = W / 2;
  const cy = H / 2;
  const tickLines = Array.from({ length: ticks }, (_, i) => {
    const a = (i / ticks) * Math.PI * 2;
    const k = i % 6 === 0 ? 0.9 : 0.95;
    return [cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k, cx + Math.cos(a) * rx, cy + Math.sin(a) * ry];
  });
  return (
    <svg
      aria-hidden
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      className={["sk-svg sk-orbit pointer-events-none", className ?? "absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"].join(" ")}
    >
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} className="sk-construction" />
      {tickLines.map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className="sk-construction-solid" />
      ))}
      {axes && (
        <>
          <line x1={cx - rx - pad} y1={cy} x2={cx + rx + pad} y2={cy} className="sk-construction sk-faint" />
          <line x1={cx} y1={cy - ry - pad} x2={cx} y2={cy + ry + pad} className="sk-construction sk-faint" />
        </>
      )}
    </svg>
  );
}
