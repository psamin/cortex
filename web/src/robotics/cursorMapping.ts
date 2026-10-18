/** Pointer -> normalized device coordinates for a canvas rect. */
export function pointerToNdc(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
): [number, number] {
  const x = ((clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  const y = -(((clientY - rect.top) / Math.max(rect.height, 1)) * 2 - 1);
  return [x, y];
}

/**
 * Intersect a ray with a plane. Returns false when the ray is (nearly) parallel
 * to the plane, in which case the caller keeps its previous target.
 * For an orthographic camera every ray has the same direction, so negative t is allowed.
 */
export function intersectRayPlane(
  origin: ArrayLike<number>,
  dir: ArrayLike<number>,
  planePoint: ArrayLike<number>,
  planeNormal: ArrayLike<number>,
  out: Float64Array | number[],
  parallelEpsilon = 1e-3,
): boolean {
  const denom = dir[0] * planeNormal[0] + dir[1] * planeNormal[1] + dir[2] * planeNormal[2];
  if (Math.abs(denom) < parallelEpsilon) return false;
  const t =
    ((planePoint[0] - origin[0]) * planeNormal[0] +
      (planePoint[1] - origin[1]) * planeNormal[1] +
      (planePoint[2] - origin[2]) * planeNormal[2]) /
    denom;
  if (!Number.isFinite(t)) return false;
  out[0] = origin[0] + dir[0] * t;
  out[1] = origin[1] + dir[1] * t;
  out[2] = origin[2] + dir[2] * t;
  return true;
}

/**
 * Interaction plane: vertical, facing the camera horizontally, pushed toward the
 * viewer by `forwardOffset` so cursor targets sit in front of the robot.
 */
export function interactionPlane(
  cameraForward: ArrayLike<number>,
  base: ArrayLike<number>,
  forwardOffset: number,
  out: { point: Float64Array | number[]; normal: Float64Array | number[] } = { point: [0, 0, 0], normal: [0, 0, 0] },
): typeof out {
  // Horizontal component of the view direction, pointing from scene to camera.
  let nx = -cameraForward[0];
  let nz = -cameraForward[2];
  const len = Math.hypot(nx, nz) || 1;
  nx /= len;
  nz /= len;
  out.point[0] = base[0] + nx * forwardOffset;
  out.point[1] = base[1];
  out.point[2] = base[2] + nz * forwardOffset;
  out.normal[0] = nx;
  out.normal[1] = 0;
  out.normal[2] = nz;
  return out;
}
