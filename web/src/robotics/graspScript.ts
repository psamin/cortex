import type { Vec3 } from "./types";

export interface GraspWaypoint {
  /** TCP target (world). */
  target: Vec3;
  /** Gripper opening command (m). */
  gripper: number;
  /** Tool approach direction. */
  direction: Vec3;
  /** Optional world yaw (rad) the finger closing axis should align with (mod 90 deg). */
  roll?: number;
  /** Minimum dwell once reached (s). */
  hold?: number;
  /** Fired when this waypoint is reached and the fingers have settled. */
  action?: "attach" | "release";
  /** Max time before the waypoint is abandoned (s). */
  timeout?: number;
}

export type GraspPhase = "moving" | "holding" | "attached" | "released" | "failed" | "done";

/**
 * Deterministic pick-and-place sequencer. Advances only when the TCP has actually
 * reached each waypoint (or it times out). Attachment is requested through a callback
 * that checks real proximity/orientation conditions; it is never forced.
 */
export class GraspScript {
  private index = 0;
  private dwell = 0;
  private elapsed = 0;
  phase: GraspPhase = "moving";
  attached = false;
  failed = false;

  constructor(
    readonly waypoints: GraspWaypoint[],
    private readonly onAttach?: () => boolean,
    private readonly onRelease?: () => void,
  ) {}

  update(
    dt: number,
    tcp: ArrayLike<number>,
    ikOk: boolean,
    gripperActual: number,
  ): { target: Vec3; gripper: number; direction: Vec3; roll?: number; phase: GraspPhase; done: boolean } {
    const wp = this.waypoints[Math.min(this.index, this.waypoints.length - 1)];
    if (this.index >= this.waypoints.length) {
      return { target: wp.target, gripper: wp.gripper, direction: wp.direction, roll: wp.roll, phase: "done", done: true };
    }
    this.elapsed += dt;
    const d = Math.hypot(tcp[0] - wp.target[0], tcp[1] - wp.target[1], tcp[2] - wp.target[2]);
    const fingersSettled = Math.abs(gripperActual - wp.gripper) < 0.002;
    const reached = d < 0.008 && ikOk;

    if ((reached && fingersSettled) || this.elapsed > (wp.timeout ?? 6)) {
      this.dwell += dt;
      this.phase = "holding";
      if (this.dwell >= (wp.hold ?? 0.15)) {
        if (wp.action === "attach") {
          this.attached = reached && (this.onAttach?.() ?? false);
          if (!this.attached) {
            this.failed = true;
            this.phase = "failed";
          } else this.phase = "attached";
        }
        if (wp.action === "release") {
          if (this.attached) this.onRelease?.();
          this.attached = false;
          this.phase = "released";
        }
        this.index++;
        this.dwell = 0;
        this.elapsed = 0;
        // A failed grasp skips straight to the final (retreat) waypoint.
        if (this.failed) this.index = Math.max(this.index, this.waypoints.length - 1);
      }
    } else {
      this.phase = "moving";
    }
    return {
      target: wp.target,
      gripper: wp.gripper,
      direction: wp.direction,
      roll: wp.roll,
      phase: this.phase,
      done: this.index >= this.waypoints.length,
    };
  }
}
