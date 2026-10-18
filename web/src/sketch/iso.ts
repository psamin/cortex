/**
 * True isometric projection (30°), the camera angle of the Kinetic hero.
 * World axes: x runs down-right, y runs down-left, z is up. One world unit = `s` px.
 */
export type Vec3 = readonly [number, number, number];
export type Pt = readonly [number, number];

const C30 = Math.cos(Math.PI / 6); // 0.866
const S30 = 0.5;

export function iso([x, y, z]: Vec3, s = 1, origin: Pt = [0, 0]): Pt {
  return [origin[0] + (x - y) * C30 * s, origin[1] + (x + y) * S30 * s - z * s];
}

/** SVG `points` string for a polygon of world points. */
export function poly(pts: Vec3[], s = 1, origin: Pt = [0, 0]): string {
  return pts.map((p) => iso(p, s, origin).map((n) => +n.toFixed(2)).join(",")).join(" ");
}

/** SVG path `d` for an open polyline of world points. */
export function line(pts: Vec3[], s = 1, origin: Pt = [0, 0]): string {
  return pts.map((p, i) => (i ? "L" : "M") + iso(p, s, origin).map((n) => +n.toFixed(2)).join(" ")).join(" ");
}

export interface BoxFaces {
  top: string;
  left: string; // the face at y = max (faces down-left on screen)
  right: string; // the face at x = max (faces down-right on screen)
}

/** The three visible faces of an axis-aligned box at (x, y, z) with size (w, d, h). */
export function boxFaces([x, y, z]: Vec3, [w, d, h]: Vec3, s = 1, origin: Pt = [0, 0]): BoxFaces {
  const X = x + w;
  const Y = y + d;
  const Z = z + h;
  return {
    top: poly([[x, y, Z], [X, y, Z], [X, Y, Z], [x, Y, Z]], s, origin),
    left: poly([[x, Y, z], [X, Y, z], [X, Y, Z], [x, Y, Z]], s, origin),
    right: poly([[X, y, z], [X, Y, z], [X, Y, Z], [X, y, Z]], s, origin),
  };
}

/** Points of a circle of radius r around `c`, lying in the plane normal to `axis`. */
export function circle(c: Vec3, r: number, axis: "x" | "y" | "z" = "z", n = 32, from = 0, to = Math.PI * 2): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i <= n; i++) {
    const t = from + ((to - from) * i) / n;
    const a = r * Math.cos(t);
    const b = r * Math.sin(t);
    out.push(axis === "z" ? [c[0] + a, c[1] + b, c[2]] : axis === "x" ? [c[0], c[1] + a, c[2] + b] : [c[0] + a, c[1], c[2] + b]);
  }
  return out;
}

/** Screen-space size of an isometric circle on the floor: rx = 1.2247 r, ry = 0.7071 r. */
export const ISO_ELLIPSE = { rx: Math.SQRT2 * C30, ry: Math.SQRT2 * S30 } as const;
