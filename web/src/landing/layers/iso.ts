/** Isometric projection helpers for the line-art layer stack (SVG user units). */
export const COS30 = Math.cos(Math.PI / 6);
export const SIN30 = 0.5;

export type P3 = [u: number, v: number, z: number];

/** Plate-local (u, v) on the plate plane plus height z, to screen x/y around the plate centre. */
export function proj([u, v, z]: P3): [number, number] {
  return [(u - v) * COS30, (u + v) * SIN30 - z];
}

const f = (n: number) => Math.round(n * 100) / 100;

export function path(points: P3[], close = false): string {
  return (
    points
      .map((p, i) => {
        const [x, y] = proj(p);
        return `${i ? "L" : "M"}${f(x)} ${f(y)}`;
      })
      .join("") + (close ? "Z" : "")
  );
}

/** A circle lying on the plate (an ellipse on screen), sampled as a polyline. */
export function ring(cu: number, cv: number, r: number, from = 0, to = Math.PI * 2, steps = 64, z = 0): P3[] {
  const out: P3[] = [];
  for (let i = 0; i <= steps; i++) {
    const a = from + ((to - from) * i) / steps;
    out.push([cu + Math.cos(a) * r, cv + Math.sin(a) * r, z]);
  }
  return out;
}

/** Visible faces of an extruded box on the plate: top, +v face, +u face. */
export function box(u0: number, u1: number, v0: number, v1: number, h: number) {
  return {
    top: path([[u0, v0, h], [u1, v0, h], [u1, v1, h], [u0, v1, h]], true),
    faceV: path([[u0, v1, 0], [u1, v1, 0], [u1, v1, h], [u0, v1, h]], true),
    faceU: path([[u1, v0, 0], [u1, v1, 0], [u1, v1, h], [u1, v0, h]], true),
  };
}
