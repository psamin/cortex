import type { RobotConfig } from "./types";

/**
 * Design dimensions (meters). Starting values for an illustrative arm,
 * not measurements of a real product. Geometry AND kinematics both read these,
 * so the visual model and the transform chain can never drift apart.
 */
export const DIMS = {
  plinthHeight: 0.06,
  baseHeight: 0.35, // floor -> shoulder axis
  turretHeight: 0.17, // J1 origin -> shoulder axis
  upperArm: 0.65, // shoulder -> elbow
  forearmToRoll: 0.12, // elbow -> forearm roll (J4)
  rollToWrist: 0.43, // J4 -> wrist pitch (J5); elbow -> wrist = 0.55
  wristToFlange: 0.1, // J5 -> flange roll (J6)
  flangeToTool: 0.1, // J6 -> gripper mount (wrist assembly = 0.20)
  gripper: 0.18, // mount -> TCP between fingertips
} as const;

const deg = Math.PI / 180;

/**
 * Six-axis articulated arm.
 * Zero pose: upper arm vertical (+Y), forearm horizontal along the turret's +X.
 * J1 yaw (Y) · J2 shoulder pitch (Z) · J3 elbow pitch (Z) · J4 forearm roll (X)
 * J5 wrist pitch (Z) · J6 flange roll (X)
 */
export const ROBOT: RobotConfig = {
  baseOrigin: [0, 0, 0],
  joints: [
    {
      name: "base",
      label: "J01 / BASE",
      axis: [0, 1, 0],
      offset: [0, DIMS.baseHeight - DIMS.turretHeight, 0],
      min: -165 * deg,
      max: 165 * deg,
      maxVelocity: 1.6,
      maxAcceleration: 5,
    },
    {
      name: "shoulder",
      label: "J02 / SHOULDER",
      axis: [0, 0, 1],
      offset: [0, DIMS.turretHeight, 0],
      min: -95 * deg,
      max: 60 * deg,
      maxVelocity: 1.4,
      maxAcceleration: 4.5,
    },
    {
      name: "elbow",
      label: "J03 / ELBOW",
      axis: [0, 0, 1],
      offset: [0, DIMS.upperArm, 0],
      min: -150 * deg,
      max: 75 * deg,
      maxVelocity: 1.8,
      maxAcceleration: 6,
    },
    {
      name: "forearm",
      label: "J04 / ROLL",
      axis: [1, 0, 0],
      offset: [DIMS.forearmToRoll, 0, 0],
      min: -170 * deg,
      max: 170 * deg,
      maxVelocity: 2.6,
      maxAcceleration: 9,
    },
    {
      name: "wrist",
      label: "J05 / WRIST",
      axis: [0, 0, 1],
      offset: [DIMS.rollToWrist, 0, 0],
      min: -120 * deg,
      max: 120 * deg,
      maxVelocity: 2.8,
      maxAcceleration: 10,
    },
    {
      name: "flange",
      label: "J06 / FLANGE",
      axis: [1, 0, 0],
      offset: [DIMS.wristToFlange, 0, 0],
      min: -175 * deg,
      max: 175 * deg,
      maxVelocity: 3.2,
      maxAcceleration: 12,
    },
  ],
  toolOffset: [DIMS.flangeToTool + DIMS.gripper, 0, 0],
  toolAxis: [1, 0, 0],
  // Turned to reach toward the headline (screen-left in the hero camera), upper arm
  // nearly vertical, forearm slightly down, gripper angled ~45 deg toward the floor.
  rest: [-2.05, -0.08, -0.4, 0, -0.35, 0],
};

export const JOINT_COUNT = ROBOT.joints.length;

/** Distance from the shoulder axis to the TCP with the arm fully stretched. */
export function maxReach(cfg: RobotConfig = ROBOT): number {
  const [, , elbow, roll, wrist, flange] = cfg.joints;
  const len = (v: readonly number[]) => Math.hypot(v[0], v[1], v[2]);
  return (
    len(elbow.offset) + len(roll.offset) + len(wrist.offset) + len(flange.offset) + len(cfg.toolOffset)
  );
}

/** World height of the shoulder axis. */
export function shoulderHeight(cfg: RobotConfig = ROBOT): number {
  return cfg.baseOrigin[1] + cfg.joints[0].offset[1] + cfg.joints[1].offset[1];
}
