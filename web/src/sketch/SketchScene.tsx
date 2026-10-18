import { Box, Cyl } from "./IsoIcon";
import { circle, iso, line, poly, type Pt, type Vec3 } from "./iso";

type XZ = readonly [number, number];

/** Convex hull of screen points (monotone chain), for the silhouette of an extruded link. */
function hull(pts: Pt[]): Pt[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: Pt[] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** Capsule outline from a to b (radius r) in the arm's x–z plane. */
function capsule(a: XZ, b: XZ, r: number, n = 10): XZ[] {
  const t = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const arc = (c: XZ, from: number): XZ[] =>
    Array.from({ length: n + 1 }, (_, i) => {
      const k = from - (Math.PI * i) / n;
      return [c[0] + r * Math.cos(k), c[1] + r * Math.sin(k)] as const;
    });
  return [...arc(b, t + Math.PI / 2), ...arc(a, t - Math.PI / 2)];
}

/** One arm link: a capsule plate of thickness `t` (in y), flat toon faces, 1 px ink. */
function Link({ a, b, r, t, s, o }: { a: XZ; b: XZ; r: number; t: number; s: number; o: Pt }) {
  const shape = capsule(a, b, r);
  const front = shape.map(([x, z]) => iso([x, t / 2, z], s, o));
  const back = shape.map(([x, z]) => iso([x, -t / 2, z], s, o));
  const pts = (ps: Pt[]) => ps.map((q) => q.map((v) => +v.toFixed(2)).join(",")).join(" ");
  return (
    <g>
      <polygon points={pts(hull([...front, ...back]))} className="sk-f3" />
      <polygon points={pts(front)} className="sk-f1" />
    </g>
  );
}

/** A joint: dark ring with a grey hub, drawn on a link's front face (y = +t/2). */
function Joint({ c, y, r, s, o }: { c: XZ; y: number; r: number; s: number; o: Pt }) {
  const at: Vec3 = [c[0], y, c[1]];
  return (
    <g>
      <polygon points={poly(circle(at, r, "y", 28), s, o)} className="sk-ink-fill" />
      <polygon points={poly(circle(at, r * 0.45, "y", 20), s, o)} className="sk-f2" />
    </g>
  );
}

/**
 * The empty-state illustration: an SO-101-style arm (base plate, rotating shoulder, upper arm,
 * forearm, wrist, two-finger gripper) reaching for a cube, a dashed reach orbit and one callout.
 * Purely decorative (aria-hidden); pair it with a real headline.
 */
export function SketchScene({
  label = "OBJ.01 / CUBE",
  value = "AWAITING DATA",
  width = 280,
  className,
}: {
  label?: string;
  value?: string;
  width?: number;
  className?: string;
}) {
  const s = 36;
  const o: Pt = [-6, 8];
  const top = 0.14; // plate surface
  // arm joints in the x–z plane
  const S: XZ = [0, 0.86];
  const E: XZ = [0.5, 1.95];
  const W: XZ = [1.4, 1.42];
  const G: XZ = [1.48, 1.02];
  // cube on the plate, under the gripper
  const cube: Vec3 = [1.36, -0.14, top];
  const cs = 0.28;
  const anchor = iso([cube[0] + cs, cube[1] + cs / 2, top + cs], s, o);
  const elbow: Pt = [anchor[0] + 18, anchor[1] + 16];
  const end: Pt = [elbow[0] + 14, elbow[1]];
  // Keep the callout text at ~10 px on screen whatever the drawing's width.
  const k = 300 / width;
  const fs = 10 * k;
  return (
    <svg
      aria-hidden
      viewBox="-84 -76 300 146"
      width={width}
      height={(width * 146) / 300}
      className={["sk-svg sk-scene", className].filter(Boolean).join(" ")}
    >
      {/* base plate, reach orbit and axes */}
      <Box at={[-0.85, -0.85, 0]} size={[2.65, 1.7, top]} s={s} o={o} />
      <path d={line(circle([0, 0, top], 1.5, "z", 64), s, o)} className="sk-construction" />
      <path d={line([[-0.8, 0, top], [1.75, 0, top]], s, o)} className="sk-construction sk-faint" />
      <path d={line([[0, -0.8, top], [0, 0.8, top]], s, o)} className="sk-construction sk-faint" />
      {[[-0.72, -0.72], [1.67, -0.72], [1.67, 0.72], [-0.72, 0.72]].map(([x, y]) => (
        <polygon key={`${x}${y}`} points={poly(circle([x, y, top], 0.06, "z", 10), s, o)} className="sk-ink-fill" />
      ))}

      {/* rotating shoulder: turntable, then the motor bracket */}
      <Cyl c={[0, 0, top]} r={0.52} h={0.12} s={s} o={o} />
      <Cyl c={[0, 0, top + 0.12]} r={0.4} h={0.2} s={s} o={o} />
      <Box at={[-0.3, -0.24, top + 0.32]} size={[0.6, 0.48, 0.42]} s={s} o={o} />

      {/* cube */}
      <Box at={cube} size={[cs, cs, cs]} s={s} o={o} />

      {/* upper arm, forearm, wrist */}
      <Link a={S} b={E} r={0.17} t={0.3} s={s} o={o} />
      <Joint c={S} y={0.15} r={0.13} s={s} o={o} />
      <Link a={E} b={W} r={0.15} t={0.26} s={s} o={o} />
      <Joint c={E} y={0.13} r={0.12} s={s} o={o} />
      <Link a={W} b={G} r={0.12} t={0.24} s={s} o={o} />
      <Joint c={W} y={0.12} r={0.1} s={s} o={o} />

      {/* two-finger gripper: palm, fixed jaw and moving jaw */}
      <Box at={[G[0] - 0.2, -0.13, G[1] - 0.16]} size={[0.4, 0.26, 0.14]} s={s} o={o} />
      <Box at={[G[0] - 0.19, -0.05, G[1] - 0.5]} size={[0.07, 0.1, 0.34]} s={s} o={o} />
      <Box at={[G[0] + 0.12, -0.05, G[1] - 0.44]} size={[0.07, 0.1, 0.28]} s={s} o={o} />

      {/* callout to the cube */}
      <path d={`M${anchor[0]} ${anchor[1]} L${elbow[0]} ${elbow[1]} L${end[0]} ${end[1]}`} fill="none" className="sk-leader" />
      <circle cx={anchor[0]} cy={anchor[1]} r={2.6} className="sk-node" />
      <text x={end[0] + 5} y={end[1] + fs * 0.35} className="sk-svg-label" style={{ fontSize: fs }}>
        {label}
      </text>
      <text x={end[0] + 5} y={end[1] + fs * 1.75} className="sk-svg-label sk-svg-label-2" style={{ fontSize: fs }}>
        {value}
      </text>
    </svg>
  );
}
