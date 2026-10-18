/**
 * Allocation-free 3x3 / vector helpers on Float64Array (row-major matrices).
 * Used by the kinematics core so it stays independent of Three.js.
 */
export type M3 = Float64Array; // length 9
export type V3 = Float64Array; // length 3

export const m3 = (): M3 => {
  const m = new Float64Array(9);
  m[0] = m[4] = m[8] = 1;
  return m;
};
export const v3 = (x = 0, y = 0, z = 0): V3 => Float64Array.of(x, y, z);

export function setIdentity(m: M3): M3 {
  m.fill(0);
  m[0] = m[4] = m[8] = 1;
  return m;
}

/** Rodrigues rotation matrix for a unit axis. */
export function axisAngle(ax: number, ay: number, az: number, angle: number, out: M3): M3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  out[0] = t * ax * ax + c;
  out[1] = t * ax * ay - s * az;
  out[2] = t * ax * az + s * ay;
  out[3] = t * ax * ay + s * az;
  out[4] = t * ay * ay + c;
  out[5] = t * ay * az - s * ax;
  out[6] = t * ax * az - s * ay;
  out[7] = t * ay * az + s * ax;
  out[8] = t * az * az + c;
  return out;
}

/** out = a * b (out may not alias a or b). */
export function mulM3(a: M3, b: M3, out: M3): M3 {
  for (let r = 0; r < 3; r++) {
    const a0 = a[r * 3];
    const a1 = a[r * 3 + 1];
    const a2 = a[r * 3 + 2];
    out[r * 3] = a0 * b[0] + a1 * b[3] + a2 * b[6];
    out[r * 3 + 1] = a0 * b[1] + a1 * b[4] + a2 * b[7];
    out[r * 3 + 2] = a0 * b[2] + a1 * b[5] + a2 * b[8];
  }
  return out;
}

/** out = m * (x, y, z) */
export function mulM3V(m: M3, x: number, y: number, z: number, out: V3): V3 {
  out[0] = m[0] * x + m[1] * y + m[2] * z;
  out[1] = m[3] * x + m[4] * y + m[5] * z;
  out[2] = m[6] * x + m[7] * y + m[8] * z;
  return out;
}

export const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

export const isFiniteArray = (a: ArrayLike<number>) => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false;
  return true;
};

/**
 * Solve A x = b in place for a small dense system (Gaussian elimination, partial pivoting).
 * A is n*n row-major and is destroyed; result is written to b. Returns false if singular.
 */
export function solveInPlace(A: Float64Array, b: Float64Array, n: number): boolean {
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r * n + col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-12) return false;
    if (piv !== col) {
      for (let k = 0; k < n; k++) {
        const t = A[col * n + k];
        A[col * n + k] = A[piv * n + k];
        A[piv * n + k] = t;
      }
      const t = b[col];
      b[col] = b[piv];
      b[piv] = t;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      if (f === 0) continue;
      for (let k = col; k < n; k++) A[r * n + k] -= f * A[col * n + k];
      b[r] -= f * b[col];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    b[r] = s / A[r * n + r];
  }
  return true;
}
