import { describe, expect, it } from "vitest";
import { createMotionState, maxJointError, smoothToward, stepJoints, SUBSTEP } from "../motionController";
import { ROBOT } from "../robotConfig";
import { withinLimits, clampToLimits } from "../jointLimits";

describe("joint motion controller", () => {
  it("respects velocity and acceleration limits every substep", () => {
    const s = createMotionState(ROBOT.rest);
    const goal = ROBOT.joints.map((j) => j.max * 0.9);
    let prevV = Float64Array.from(s.v);
    for (let k = 0; k < 2000; k++) {
      stepJoints(s, goal, ROBOT, SUBSTEP);
      ROBOT.joints.forEach((j, i) => {
        expect(Math.abs(s.v[i])).toBeLessThanOrEqual(j.maxVelocity + 1e-9);
        expect(Math.abs(s.v[i] - prevV[i])).toBeLessThanOrEqual(j.maxAcceleration * SUBSTEP + 1e-9);
      });
      prevV = Float64Array.from(s.v);
    }
  });

  it("converges to the goal without leaving the limits", () => {
    const s = createMotionState(ROBOT.rest);
    const goal = [1, -0.6, -1.2, 0.5, 0.4, -1];
    for (let t = 0; t < 6; t += 1 / 60) stepJoints(s, goal, ROBOT, 1 / 60);
    expect(maxJointError(s.q, goal)).toBeLessThan(1e-3);
    expect(withinLimits(s.q, ROBOT)).toBe(true);
  });

  it("clamps goals beyond the limits and stops at the limit", () => {
    const s = createMotionState(ROBOT.rest);
    const goal = ROBOT.joints.map((j) => j.max + 2);
    for (let t = 0; t < 10; t += 1 / 60) stepJoints(s, goal, ROBOT, 1 / 60);
    const lim = Float64Array.from(goal);
    clampToLimits(lim, ROBOT);
    expect(maxJointError(s.q, lim)).toBeLessThan(1e-6);
  });

  it("is frame-rate independent (30 Hz vs 144 Hz)", () => {
    const a = createMotionState(ROBOT.rest);
    const b = createMotionState(ROBOT.rest);
    const goal = [0.9, -0.5, -1.4, 0.3, -0.2, 0.8];
    for (let k = 0; k < 30; k++) stepJoints(a, goal, ROBOT, 1 / 30);
    for (let k = 0; k < 144; k++) stepJoints(b, goal, ROBOT, 1 / 144);
    expect(maxJointError(a.q, b.q)).toBeLessThan(0.01);
  });

  it("treats huge frame gaps as bounded steps (no teleport)", () => {
    const s = createMotionState(ROBOT.rest);
    const before = Float64Array.from(s.q);
    stepJoints(s, ROBOT.joints.map((j) => j.max), ROBOT, 5);
    // At most 0.1 s of motion at max velocity.
    ROBOT.joints.forEach((j, i) => expect(Math.abs(s.q[i] - before[i])).toBeLessThanOrEqual(j.maxVelocity * 0.1 + 1e-9));
  });

  it("smoothToward is frame-rate independent", () => {
    const a = Float64Array.of(0, 0, 0);
    const b = Float64Array.of(0, 0, 0);
    for (let k = 0; k < 30; k++) smoothToward(a, [1, 2, 3], 1 / 30, 0.2);
    for (let k = 0; k < 120; k++) smoothToward(b, [1, 2, 3], 1 / 120, 0.2);
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i], 6);
  });
});
