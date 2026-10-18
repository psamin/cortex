import type { GraspWaypoint } from "./graspScript";
import type { Vec3 } from "./types";

export const CUBE_SIZE = 0.045;
/** Two floor stations the demo shuttles the cube between. */
export const STATIONS: Vec3[] = [
  [-0.4, CUBE_SIZE / 2, 0.64],
  [-0.78, CUBE_SIZE / 2, 0.22],
];

const DOWN: Vec3 = [0, -1, 0];
const OPEN = 0.07;
const CLOSED = CUBE_SIZE; // pads meet the cube faces

/**
 * Pick-and-place waypoints: hover, descend, close+attach, lift, traverse, lower,
 * open+release, retreat. Fingers align with the cube's yaw for the approach.
 */
export function pickAndPlaceWaypoints(cube: Vec3, cubeYaw: number, place: Vec3): GraspWaypoint[] {
  const hover = 0.26;
  const at = (p: Vec3, y: number): Vec3 => [p[0], y, p[2]];
  return [
    { target: at(cube, hover), gripper: OPEN, direction: DOWN, roll: cubeYaw, timeout: 5 },
    { target: at(cube, cube[1] + 0.004), gripper: OPEN, direction: DOWN, roll: cubeYaw, hold: 0.1, timeout: 4 },
    { target: at(cube, cube[1] + 0.004), gripper: CLOSED, direction: DOWN, roll: cubeYaw, action: "attach", hold: 0.12 },
    { target: at(cube, hover), gripper: CLOSED, direction: DOWN, timeout: 4 },
    { target: at(place, hover), gripper: CLOSED, direction: DOWN, timeout: 6 },
    { target: at(place, CUBE_SIZE / 2 + 0.008), gripper: CLOSED, direction: DOWN, hold: 0.1, timeout: 4 },
    { target: at(place, CUBE_SIZE / 2 + 0.008), gripper: OPEN, direction: DOWN, action: "release", hold: 0.15 },
    { target: at(place, hover), gripper: OPEN, direction: DOWN, timeout: 4 },
  ];
}
