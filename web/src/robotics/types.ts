/** World convention: +Y up, XZ ground plane, meters, radians. */
export type Vec3 = [number, number, number];

export interface JointSpec {
  /** Machine name, e.g. "shoulder". */
  name: string;
  /** Short engineering label, e.g. "J02 / SHOULDER". */
  label: string;
  /** Rotation axis expressed in this joint's parent frame (unit vector). */
  axis: Vec3;
  /** Translation from the parent joint origin to this joint origin, in the parent frame. */
  offset: Vec3;
  min: number;
  max: number;
  /** rad/s */
  maxVelocity: number;
  /** rad/s^2 */
  maxAcceleration: number;
}

export interface RobotConfig {
  joints: JointSpec[];
  /** World position of the fixed base origin (floor contact). */
  baseOrigin: Vec3;
  /** Translation from the last joint frame to the tool center point (TCP). */
  toolOffset: Vec3;
  /** Tool approach axis in the last joint frame (direction the gripper points). */
  toolAxis: Vec3;
  /** Resting joint configuration. */
  rest: number[];
}

export interface IKOptions {
  maxIterations: number;
  /** Position tolerance in meters. */
  tolerance: number;
  /** Base damping coefficient (lambda). */
  damping: number;
  /** Max Cartesian error used per iteration (m). */
  maxStep: number;
  /** Max joint change per iteration (rad). */
  maxJointStep: number;
  /** Gain of the null-space posture objective pulling toward `posture`. */
  postureGain: number;
  posture?: ArrayLike<number>;
  /** Damping used for the posture null-space projector (keep small). */
  projectorDamping: number;
  /** Optional desired tool approach direction (world, unit). */
  direction?: Vec3 | null;
  /** Fraction of the direction error corrected per iteration (secondary task gain, 0..1). */
  directionWeight: number;
  /** Direction tolerance (radians of misalignment) when a direction is set. */
  directionTolerance: number;
}

export interface IKResult {
  converged: boolean;
  /** Final position error (m). */
  positionError: number;
  /** Final direction error (rad), 0 when unused. */
  directionError: number;
  iterations: number;
  /** True if any joint ended on a limit. */
  limited: boolean;
}

export type RobotState =
  | "INITIALIZING"
  | "INTRO"
  | "IDLE"
  | "TRACKING"
  | "RETURNING"
  | "GRASPING"
  | "RELEASING"
  | "ERROR";

export type RobotEvent =
  | "READY"
  | "INTRO_DONE"
  | "POINTER_ENTER"
  | "POINTER_LEAVE"
  | "REACHED_REST"
  | "GRASP"
  | "GRASP_DONE"
  | "RELEASE_DONE"
  | "FAULT"
  | "RECOVER";
