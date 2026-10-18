import type { ReactNode } from "react";
import { Box } from "./IsoIcon";
import { iso, type Pt, type Vec3 } from "./iso";

function bounds(size: Vec3, s: number, lift: number) {
  const [w, d, h] = size;
  const pts = [0, w].flatMap((x) => [0, d].flatMap((y) => [0, h + lift].map((z) => iso([x, y, z], s))));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

/**
 * An isometric extruded box drawn as flat toon faces with 1 px ink outlines.
 * `size` is [w, d, h] in world units; `scale` is px per unit. The SVG fits its own bounds.
 */
export function IsoBox({
  size,
  scale = 24,
  lift = 0,
  accent,
  className,
  children,
  title,
}: {
  size: Vec3;
  scale?: number;
  /** Raise the box this many world units (exploded-layer look); the footprint stays drawn dashed. */
  lift?: number;
  accent?: boolean;
  className?: string;
  /** Extra SVG drawn on top, in the same coordinates (use `isoPoint`). */
  children?: (pt: (p: Vec3) => Pt) => ReactNode;
  title?: string;
}) {
  const b = bounds(size, scale, lift);
  const pad = 2;
  const o: Pt = [-b.minX + pad, -b.minY + pad];
  const W = b.maxX - b.minX + pad * 2;
  const H = b.maxY - b.minY + pad * 2;
  const [w, d] = size;
  return (
    <svg
      viewBox={`0 0 ${W.toFixed(1)} ${H.toFixed(1)}`}
      width={W}
      height={H}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      className={["sk-svg", "sk-isobox", accent ? "sk-accent" : "", className].filter(Boolean).join(" ")}
    >
      {title && <title>{title}</title>}
      {lift > 0 && (
        <polygon
          className="sk-construction"
          points={[[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0]].map((p) => iso(p as unknown as Vec3, scale, o).join(",")).join(" ")}
        />
      )}
      <g className="sk-lift" style={{ transform: `translateY(${-lift * scale}px)` }}>
        <Box at={[0, 0, 0]} size={size} s={scale} o={o} accent={accent} />
        {children?.((p) => iso(p, scale, o))}
      </g>
    </svg>
  );
}

/** A thin slab: the plate the hero arm stands on, and the unit of the exploded-layers graphic. */
export function IsoPlate({
  w = 2,
  d = 2,
  h = 0.18,
  ...rest
}: { w?: number; d?: number; h?: number } & Omit<Parameters<typeof IsoBox>[0], "size">) {
  return <IsoBox size={[w, d, h]} {...rest} />;
}
