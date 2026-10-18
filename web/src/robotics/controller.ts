import { isFiniteArray } from "./math";
import { IKSolver } from "./inverseKinematics";
import { ForwardKinematics } from "./forwardKinematics";
import { createMotionState, maxJointError, MAX_FRAME_DT, smoothToward, stepJoints } from "./motionController";
import { RobotStateMachine } from "./stateMachine";
import type { IKResult, RobotConfig, RobotEvent, RobotState, Vec3 } from "./types";
import { clampToWorkspace, workspaceFor, type Workspace } from "./workspace";
import { GraspScript, type GraspWaypoint } from "./graspScript";

export interface ControllerTuning {
  /** Cursor target smoothing time constant (s). */
  targetTau: number;
  /** IK iterations per frame (warm started). */
  ikIterationsPerFrame: number;
  /** Idle end-effector drift amplitude (m). 0 disables idle motion. */
  idleAmplitude: number;
  /** Idle cycle period (s). */
  idlePeriod: number;
  /** Joint-space speed multiplier while returning to rest. */
  returnSpeed: number;
}

export const DEFAULT_TUNING: ControllerTuning = {
  targetTau: 0.14,
  ikIterationsPerFrame: 10,
  idleAmplitude: 0.035,
  idlePeriod: 11,
  returnSpeed: 0.7,
};

/**
 * Framework-agnostic robot runtime. The renderer calls `update(dt)` once per frame
 * and reads `q`, `target`, `ik`, `state`. High-frequency inputs (cursor) go through
 * `setPointerTarget` and never trigger React renders.
 */
export class RobotController {
  readonly sm = new RobotStateMachine();
  readonly solver: IKSolver;
  /** Reporting FK on the *actual* joint angles. */
  readonly fk: ForwardKinematics;
  private readonly rollFk: ForwardKinematics;
  readonly motion;
  /** IK warm-start / desired configuration. */
  readonly qDesired: Float64Array;
  readonly workspace: Workspace;
  tuning: ControllerTuning;

  /** Raw (clamped) pointer target and the smoothed target fed to IK. */
  readonly rawTarget = new Float64Array(3);
  readonly target = new Float64Array(3);
  readonly restTcp = new Float64Array(3);
  readonly restDir = new Float64Array(3);
  hasPointerTarget = false;
  ik: IKResult = { converged: true, positionError: 0, directionError: 0, iterations: 0, limited: false };
  /** True while the cursor target had to be pulled into the reachable region. */
  targetClamped = false;

  /** Gripper opening (m between fingers): commanded and actual. */
  gripperCommand = 0.07;
  gripper = 0.07;
  readonly gripperOpen = 0.07;

  /** Optional joint override (dev panel). */
  manual: Float64Array | null = null;

  time = 0;
  private stateTime = 0;
  private readonly lastGood: Float64Array;
  private readonly tmp = new Float64Array(3);
  private readonly dirBuf: Vec3 = [0, -1, 0];
  grasp: GraspScript | null = null;

  constructor(readonly config: RobotConfig, tuning: Partial<ControllerTuning> = {}) {
    this.tuning = { ...DEFAULT_TUNING, ...tuning };
    this.solver = new IKSolver(config);
    this.fk = new ForwardKinematics(config);
    this.rollFk = new ForwardKinematics(config);
    this.motion = createMotionState(config.rest);
    this.qDesired = Float64Array.from(config.rest);
    this.lastGood = Float64Array.from(config.rest);
    this.workspace = workspaceFor(config);
    this.fk.compute(config.rest);
    this.restTcp.set(this.fk.tcpPosition);
    this.restDir.set(this.fk.toolDirection);
    this.target.set(this.restTcp);
    this.rawTarget.set(this.restTcp);
  }

  get state(): RobotState {
    return this.sm.state;
  }
  get q(): Float64Array {
    return this.motion.q;
  }

  send(event: RobotEvent): boolean {
    const prev = this.sm.state;
    const changed = this.sm.send(event, this.time);
    if (changed) this.onEnter(this.sm.state, prev);
    return changed;
  }

  /** Start with the arm folded so the intro can unfold it to rest. */
  setInitialPose(q: ArrayLike<number>) {
    this.motion.q.set(q);
    this.motion.v.fill(0);
    this.qDesired.set(this.config.rest);
  }

  setPointerTarget(p: ArrayLike<number>) {
    this.targetClamped = clampToWorkspace(p, this.workspace, this.rawTarget);
    this.hasPointerTarget = true;
  }

  startGrasp(waypoints: GraspWaypoint[], onAttach?: () => boolean, onRelease?: () => void): boolean {
    if (this.state !== "IDLE" && this.state !== "RETURNING" && this.state !== "TRACKING") return false;
    this.grasp = new GraspScript(waypoints, onAttach, onRelease);
    return this.send("GRASP");
  }

  private onEnter(state: RobotState, _prev: RobotState) {
    this.stateTime = 0;
    if (state === "TRACKING" || state === "IDLE") {
      // Warm start IK from where the arm actually is, and target from the actual TCP.
      this.qDesired.set(this.motion.q);
      this.fk.compute(this.motion.q);
      this.target.set(this.fk.tcpPosition);
      if (state === "IDLE") this.target.set(this.restTcp);
    }
    if (state === "RETURNING") {
      this.qDesired.set(this.config.rest);
      this.hasPointerTarget = false;
    }
    if (state === "GRASPING") {
      this.qDesired.set(this.motion.q);
      this.fk.compute(this.motion.q);
      this.target.set(this.fk.tcpPosition);
    }
  }

  update(rawDt: number) {
    const dt = Math.min(Math.max(rawDt, 0), MAX_FRAME_DT);
    this.time += dt;
    this.stateTime += dt;
    const t = this.tuning;
    let speedScale = 1;

    switch (this.state) {
      case "INITIALIZING":
      case "INTRO":
        this.qDesired.set(this.config.rest);
        speedScale = 0.55;
        break;

      case "IDLE": {
        const w = (2 * Math.PI) / t.idlePeriod;
        const a = t.idleAmplitude;
        // Slow Lissajous drift of the end effector, solved through real IK.
        const ramp = Math.min(1, this.stateTime / 1.5);
        this.tmp[0] = this.restTcp[0] + a * ramp * Math.sin(w * this.stateTime) * 0.8;
        this.tmp[1] = this.restTcp[1] + a * ramp * Math.sin(2 * w * this.stateTime) * 0.5;
        this.tmp[2] = this.restTcp[2] + a * ramp * Math.sin(w * this.stateTime + 1.1) * 0.6;
        this.target.set(this.tmp);
        this.solveTo(this.target, [this.restDir[0], this.restDir[1], this.restDir[2]], 0.35);
        break;
      }

      case "TRACKING": {
        if (this.hasPointerTarget) smoothToward(this.target, this.rawTarget, dt, t.targetTau);
        this.solveTo(this.target, this.trackingDirection(this.target), 0.3);
        break;
      }

      case "RETURNING": {
        this.qDesired.set(this.config.rest);
        speedScale = t.returnSpeed;
        const err = maxJointError(this.motion.q, this.config.rest);
        let vmax = 0;
        for (let i = 0; i < this.motion.v.length; i++) vmax = Math.max(vmax, Math.abs(this.motion.v[i]));
        if (err < 0.004 && vmax < 0.02) this.send("REACHED_REST");
        break;
      }

      case "GRASPING":
      case "RELEASING": {
        const g = this.grasp;
        if (!g) {
          this.send(this.state === "GRASPING" ? "GRASP_DONE" : "RELEASE_DONE");
          break;
        }
        this.fk.compute(this.motion.q);
        const step = g.update(dt, this.fk.tcpPosition, this.ik.converged || this.ik.positionError < 0.006, this.gripper);
        this.target.set(step.target);
        this.gripperCommand = step.gripper;
        this.solveTo(this.target, step.direction, 0.6);
        if (step.roll !== undefined) this.alignFingerYaw(step.roll);
        speedScale = 0.8;
        if (step.phase === "released" && this.state === "GRASPING") this.send("GRASP_DONE");
        if (step.done) {
          this.grasp = null;
          if (this.state === "GRASPING") this.send("GRASP_DONE");
          this.send("RELEASE_DONE");
        }
        break;
      }

      case "ERROR":
        this.qDesired.set(this.lastGood);
        if (this.stateTime > 0.6) this.send("RECOVER");
        break;
    }

    if (this.manual) this.qDesired.set(this.manual);
    stepJoints(this.motion, this.qDesired, this.config, dt, { speedScale });

    // Gripper fingers: rate-limited.
    const gRate = 0.12 * dt;
    this.gripper += Math.max(-gRate, Math.min(gRate, this.gripperCommand - this.gripper));
    if (this.state !== "GRASPING" && this.state !== "RELEASING") this.gripperCommand = this.gripperOpen;

    // NaN guard: never let invalid numbers reach the transforms.
    if (!isFiniteArray(this.motion.q) || !isFiniteArray(this.qDesired) || !isFiniteArray(this.target)) {
      this.motion.q.set(this.lastGood);
      this.motion.v.fill(0);
      this.qDesired.set(this.lastGood);
      this.target.set(this.restTcp);
      this.send("FAULT");
    } else {
      this.lastGood.set(this.motion.q);
    }

    this.fk.compute(this.motion.q);
  }

  /** World yaw of the finger closing axis (gripper +Y) for a configuration. */
  fingerYaw(q: ArrayLike<number>): number {
    const R = this.rollFk.compute(q).tcpRotation;
    // Gripper +Y in world = second column of R.
    return Math.atan2(-R[7], R[1]);
  }

  /**
   * Rotate the flange (J6) so the fingers close across the requested yaw (mod 90 deg,
   * a cube is symmetric). The TCP lies on the J6 axis, so position and approach
   * direction are unaffected.
   */
  private alignFingerYaw(yaw: number) {
    const q = this.qDesired;
    const j6 = this.config.joints.length - 1;
    const now = this.fingerYaw(q);
    q[j6] += 0.01;
    const sens = (wrapPi(this.fingerYaw(q) - now) / 0.01) || 1;
    q[j6] -= 0.01;
    const quarter = Math.PI / 2;
    let err = wrapPi(yaw - now);
    err -= quarter * Math.round(err / quarter);
    const lim = this.config.joints[j6];
    q[j6] = Math.min(lim.max, Math.max(lim.min, q[j6] + err / sens));
  }

  /** Desired approach: forward along the reach direction, tipped ~35 deg downward. */
  private trackingDirection(p: Float64Array): Vec3 {
    let hx = p[0] - this.config.baseOrigin[0];
    let hz = p[2] - this.config.baseOrigin[2];
    const h = Math.hypot(hx, hz) || 1;
    hx /= h;
    hz /= h;
    // Tip further down as the target gets lower.
    const down = 0.55 + 0.5 * Math.max(0, 0.6 - p[1]);
    const n = Math.hypot(hx, down, hz);
    this.dirBuf[0] = hx / n;
    this.dirBuf[1] = -down / n;
    this.dirBuf[2] = hz / n;
    return this.dirBuf;
  }

  private solveTo(target: ArrayLike<number>, direction: Vec3 | null, directionWeight: number) {
    this.ik = this.solver.solve(this.qDesired, target as Float64Array, {
      maxIterations: this.tuning.ikIterationsPerFrame,
      posture: this.config.rest,
      direction,
      directionWeight,
      // Direction is a soft styling objective; convergence is judged on position.
      directionTolerance: Math.PI,
    });
  }
}

function wrapPi(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}
