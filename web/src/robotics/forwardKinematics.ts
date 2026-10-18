import { axisAngle, m3, mulM3, mulM3V, setIdentity, v3, type M3, type V3 } from "./math";
import type { RobotConfig } from "./types";

/**
 * Forward kinematics with preallocated buffers (safe to call every frame).
 *
 * Chain: for joint i with parent frame (R_{i-1}, p_{i-1})
 *   p_i   = p_{i-1} + R_{i-1} * offset_i      (joint origin, world)
 *   a_i   = R_{i-1} * axis_i                  (joint axis, world)
 *   R_i   = R_{i-1} * Rot(axis_i, q_i)        (joint frame after rotation)
 * TCP:  p_tcp = p_n + R_n * toolOffset,  R_tcp = R_n
 */
export class ForwardKinematics {
  readonly n: number;
  /** World origin of each joint. */
  readonly positions: V3[];
  /** World rotation axis of each joint. */
  readonly axes: V3[];
  /** World orientation of each joint frame (after its own rotation). */
  readonly rotations: M3[];
  readonly tcpPosition: V3 = v3();
  readonly tcpRotation: M3 = m3();
  /** World tool approach direction (unit). */
  readonly toolDirection: V3 = v3();

  private readonly local: M3 = m3();
  private readonly tmpV: V3 = v3();

  constructor(readonly config: RobotConfig) {
    this.n = config.joints.length;
    this.positions = Array.from({ length: this.n }, () => v3());
    this.axes = Array.from({ length: this.n }, () => v3());
    this.rotations = Array.from({ length: this.n }, () => m3());
  }

  compute(q: ArrayLike<number>): this {
    const { joints, baseOrigin, toolOffset, toolAxis } = this.config;
    let parentR: M3 | null = null;
    let px = baseOrigin[0];
    let py = baseOrigin[1];
    let pz = baseOrigin[2];

    for (let i = 0; i < this.n; i++) {
      const j = joints[i];
      const p = this.positions[i];
      const a = this.axes[i];
      const R = this.rotations[i];
      if (parentR) {
        mulM3V(parentR, j.offset[0], j.offset[1], j.offset[2], this.tmpV);
        p[0] = px + this.tmpV[0];
        p[1] = py + this.tmpV[1];
        p[2] = pz + this.tmpV[2];
        mulM3V(parentR, j.axis[0], j.axis[1], j.axis[2], a);
        axisAngle(j.axis[0], j.axis[1], j.axis[2], q[i], this.local);
        mulM3(parentR, this.local, R);
      } else {
        p[0] = px + j.offset[0];
        p[1] = py + j.offset[1];
        p[2] = pz + j.offset[2];
        a[0] = j.axis[0];
        a[1] = j.axis[1];
        a[2] = j.axis[2];
        axisAngle(j.axis[0], j.axis[1], j.axis[2], q[i], R);
      }
      parentR = R;
      px = p[0];
      py = p[1];
      pz = p[2];
    }

    const Rn = parentR ?? setIdentity(this.local);
    mulM3V(Rn, toolOffset[0], toolOffset[1], toolOffset[2], this.tmpV);
    this.tcpPosition[0] = px + this.tmpV[0];
    this.tcpPosition[1] = py + this.tmpV[1];
    this.tcpPosition[2] = pz + this.tmpV[2];
    this.tcpRotation.set(Rn);
    mulM3V(Rn, toolAxis[0], toolAxis[1], toolAxis[2], this.toolDirection);
    return this;
  }
}

/** Convenience non-pooled FK for tests and one-off queries. */
export function tcpOf(config: RobotConfig, q: ArrayLike<number>): [number, number, number] {
  const fk = new ForwardKinematics(config).compute(q);
  return [fk.tcpPosition[0], fk.tcpPosition[1], fk.tcpPosition[2]];
}
