import type { ReactNode, SVGProps } from "react";
import { boxFaces, circle, iso, line, poly, type Pt, type Vec3 } from "./iso";

/** One axis-aligned box: top face lit (f1), left face f2, right face f3, ink outlines. */
export function Box({ at, size, s, o, accent }: { at: Vec3; size: Vec3; s: number; o: Pt; accent?: boolean }) {
  const f = boxFaces(at, size, s, o);
  return (
    <g className={accent ? "sk-accent" : undefined}>
      <polygon points={f.left} className="sk-f2" />
      <polygon points={f.right} className="sk-f3" />
      <polygon points={f.top} className="sk-f1" />
    </g>
  );
}

/** Upright cylinder: front half of the side wall (f2) and the top ellipse (f1). */
export function Cyl({ c, r, h, s, o }: { c: Vec3; r: number; h: number; s: number; o: Pt }) {
  const a = -Math.PI / 4;
  const b = (3 * Math.PI) / 4;
  const bottom = circle(c, r, "z", 24, a, b);
  const top = circle([c[0], c[1], c[2] + h], r, "z", 24, a, b).reverse();
  return (
    <g>
      <polygon points={poly([...bottom, ...top], s, o)} className="sk-f2" />
      <polygon points={poly(circle([c[0], c[1], c[2] + h], r, "z", 32), s, o)} className="sk-f1" />
    </g>
  );
}

/** A flat shape in the x–z plane (y = 0) extruded back by `t`. Points are [x, z]. */
function Plaque({ shape, t, s, o, children }: { shape: [number, number][]; t: number; s: number; o: Pt; children?: ReactNode }) {
  const front: Vec3[] = shape.map(([x, z]) => [x, 0, z]);
  const back: Vec3[] = shape.map(([x, z]) => [x, -t, z]);
  return (
    <g>
      <polygon points={poly(back, s, o)} className="sk-f3" />
      {front.map((p, i) => {
        const q = front[(i + 1) % front.length];
        return <polygon key={i} points={poly([p, q, back[(i + 1) % back.length], back[i]], s, o)} className="sk-f3" />;
      })}
      <polygon points={poly(front, s, o)} className="sk-f1" />
      {children}
    </g>
  );
}

/** World point on a horizontal plane at height z, given a screen offset (u right, v down) from `c`. */
function onFloor(c: Vec3, u: number, v: number): Vec3 {
  const xmy = u / 0.866;
  const xpy = 2 * v;
  return [c[0] + (xmy + xpy) / 2, c[1] + (xpy - xmy) / 2, c[2]];
}

type Draw = () => ReactNode;

const GLYPHS = {
  cube: () => <Box at={[0, 0, 0]} size={[1, 1, 1]} s={8} o={[12, 12]} />,

  arm: () => {
    const s = 8;
    const o: Pt = [11.5, 16];
    return (
      <>
        <Box at={[-0.6, -0.6, 0]} size={[1.2, 1.2, 0.18]} s={s} o={o} />
        <Box at={[-0.18, -0.18, 0.18]} size={[0.36, 0.36, 0.95]} s={s} o={o} />
        <Box at={[0.88, -0.15, 0.62]} size={[0.3, 0.3, 0.52]} s={s} o={o} />
        <path d={line([[0.94, 0.0, 0.62], [0.94, 0.0, 0.36]], s, o)} fill="none" />
        <path d={line([[1.12, 0.0, 0.62], [1.12, 0.0, 0.36]], s, o)} fill="none" />
        <Box at={[-0.18, -0.18, 1.13]} size={[1.36, 0.36, 0.3]} s={s} o={o} />
      </>
    );
  },

  bottle: () => {
    const s = 8;
    const o: Pt = [12, 17.6];
    return (
      <>
        <Cyl c={[0, 0, 0]} r={0.55} h={1.0} s={s} o={o} />
        <Cyl c={[0, 0, 1.0]} r={0.22} h={0.42} s={s} o={o} />
        <Cyl c={[0, 0, 1.42]} r={0.27} h={0.2} s={s} o={o} />
      </>
    );
  },

  camera: () => {
    const s = 8;
    const o: Pt = [13.2, 10.8];
    const lc: Vec3 = [0.9, 0.65, 0.4];
    return (
      <>
        <Box at={[0, 0, 0]} size={[0.9, 1.3, 0.8]} s={s} o={o} />
        <Box at={[0.2, 0.2, 0.8]} size={[0.4, 0.4, 0.18]} s={s} o={o} />
        <polygon points={poly(circle(lc, 0.3, "x", 28), s, o)} className="sk-f3" />
        <polygon points={poly(circle([1.18, lc[1], lc[2]], 0.3, "x", 28), s, o)} className="sk-f1" />
        <polygon points={poly(circle([1.18, lc[1], lc[2]], 0.13, "x", 20), s, o)} className="sk-f4" />
      </>
    );
  },

  dataset: () => {
    const s = 7;
    const o: Pt = [12, 12.3];
    return (
      <>
        {[0, 0.5, 1.0].map((z) => (
          <Box key={z} at={[0, 0, z]} size={[1.2, 1.2, 0.28]} s={s} o={o} />
        ))}
      </>
    );
  },

  gpu: () => {
    const s = 6;
    const o: Pt = [12, 8];
    const pins = [0.3, 0.6, 0.9, 1.2];
    return (
      <>
        {pins.map((p) => (
          <path key={`l${p}`} d={line([[p, 1.5, 0.11], [p, 1.78, 0.11]], s, o)} fill="none" />
        ))}
        {pins.map((p) => (
          <path key={`r${p}`} d={line([[1.5, p, 0.11], [1.78, p, 0.11]], s, o)} fill="none" />
        ))}
        <Box at={[0, 0, 0]} size={[1.5, 1.5, 0.22]} s={s} o={o} />
        <Box at={[0.42, 0.42, 0.22]} size={[0.66, 0.66, 0.2]} s={s} o={o} />
      </>
    );
  },

  chart: () => {
    const s = 7;
    const o: Pt = [10.2, 10.3];
    return (
      <>
        <Box at={[0, 0, 0]} size={[1.6, 1, 0.15]} s={s} o={o} />
        <Box at={[0.15, 0.3, 0.15]} size={[0.35, 0.4, 0.45]} s={s} o={o} />
        <Box at={[0.62, 0.3, 0.15]} size={[0.35, 0.4, 0.85]} s={s} o={o} />
        <Box at={[1.09, 0.3, 0.15]} size={[0.35, 0.4, 1.3]} s={s} o={o} />
      </>
    );
  },

  loop: () => {
    const s = 8;
    const o: Pt = [12, 13.5];
    const arc = circle([0, 0, 0], 0.95, "z", 40, 0.35, Math.PI * 2 - 0.45);
    const end = iso(arc[arc.length - 1], s, o);
    const prev = iso(arc[arc.length - 3], s, o);
    const dx = end[0] - prev[0];
    const dy = end[1] - prev[1];
    const L = Math.hypot(dx, dy) || 1;
    const ux = dx / L;
    const uy = dy / L;
    const tip: Pt = [end[0] + ux * 2.6, end[1] + uy * 2.6];
    const wing = (k: number): Pt => [end[0] - uy * 2 * k, end[1] + ux * 2 * k];
    return (
      <>
        <path d={line(arc, s, o)} fill="none" />
        <polygon points={[tip, wing(1), wing(-1)].map((p) => p.join(",")).join(" ")} className="sk-ink-fill" />
        <Box at={[-0.22, -0.22, 0]} size={[0.44, 0.44, 0.44]} s={s} o={o} />
      </>
    );
  },

  shield: () => {
    const s = 11;
    const o: Pt = [6.3, 17.6];
    return (
      <Plaque
        shape={[[0, 1.25], [0.5, 1.42], [1, 1.25], [1, 0.62], [0.5, 0], [0, 0.62]]}
        t={0.2}
        s={s}
        o={o}
      >
        <path d={line([[0.27, 0, 0.74], [0.45, 0, 0.52], [0.76, 0, 0.98]], s, o)} fill="none" />
      </Plaque>
    );
  },

  warning: () => {
    const s = 9.5;
    const o: Pt = [6.2, 15.6];
    return (
      <Plaque shape={[[0, 0], [1.2, 0], [0.6, 1.12]]} t={0.2} s={s} o={o}>
        <path d={line([[0.6, 0, 0.78], [0.6, 0, 0.4]], s, o)} fill="none" />
        <path d={line([[0.6, 0, 0.24], [0.6, 0, 0.2]], s, o)} fill="none" />
      </Plaque>
    );
  },

  check: () => {
    const s = 7.5;
    const o: Pt = [12, 7.6];
    const c: Vec3 = [0.7, 0.7, 0.22];
    return (
      <>
        <Box at={[0, 0, 0]} size={[1.4, 1.4, 0.22]} s={s} o={o} />
        <path d={line([onFloor(c, -0.42, -0.02), onFloor(c, -0.12, 0.2), onFloor(c, 0.42, -0.24)], s, o)} fill="none" />
      </>
    );
  },
} satisfies Record<string, Draw>;

/** Aliases by tool kind, so callers can name what the row is about. */
const ALIASES = {
  robot: "arm",
  train: "gpu",
  evaluate: "chart",
  approve: "shield",
  improve: "loop",
  demonstrate: "camera",
  object: "cube",
} as const;

export type IsoGlyph = keyof typeof GLYPHS;
export type IsoIconName = IsoGlyph | keyof typeof ALIASES;
export const ISO_ICON_NAMES = [...Object.keys(GLYPHS), ...Object.keys(ALIASES)] as IsoIconName[];

/**
 * Isometric line-art glyph (24×24 viewBox). Stroke stays 1 px at any size.
 * `accent` draws the outline in the accent colour (live / active states).
 */
export function IsoIcon({
  name,
  size = 20,
  accent,
  className,
  title,
  ...rest
}: { name: IsoIconName; size?: number; accent?: boolean; title?: string } & Omit<SVGProps<SVGSVGElement>, "name">) {
  const key: IsoGlyph = (name in ALIASES ? ALIASES[name as keyof typeof ALIASES] : name) as IsoGlyph;
  const draw = GLYPHS[key] ?? GLYPHS.cube;
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      className={["sk-svg", accent ? "sk-accent" : "", className].filter(Boolean).join(" ")}
      {...rest}
    >
      {title && <title>{title}</title>}
      {draw()}
    </svg>
  );
}
