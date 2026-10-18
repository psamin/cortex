import { describe, expect, it } from "vitest";
import { tcpOf, ForwardKinematics } from "../forwardKinematics";
import { IKSolver } from "../inverseKinematics";
import { withinLimits } from "../jointLimits";
import { ROBOT } from "../robotConfig";
import { isFiniteArray } from "../math";
import { mulberry } from "./rng";

const solver = new IKSolver(ROBOT);
const rest = ROBOT.rest;

describe("damped least squares IK", () => {
  it("converges for reachable targets sampled around the working pose", () => {
    const rng = mulberry(11);
    let converged = 0;
    const N = 200;
    for (let k = 0; k < N; k++) {
      const qGoal = rest.map((r, i) => {
        const j = ROBOT.joints[i];
        return Math.min(j.max, Math.max(j.min, r + (rng() - 0.5) * 1.2));
      });
      const target = tcpOf(ROBOT, qGoal);
      const q = Float64Array.from(rest);
      const res = solver.solve(q, target, { maxIterations: 200, posture: rest });
      if (res.converged) converged++;
      if (res.converged) {
        const reached = tcpOf(ROBOT, q);
        expect(Math.hypot(reached[0] - target[0], reached[1] - target[1], reached[2] - target[2])).toBeLessThan(1e-3);
      }
      expect(withinLimits(q, ROBOT)).toBe(true);
    }
    expect(converged / N).toBeGreaterThanOrEqual(0.98);
  });

  it("warm starting from a nearby solution converges in very few iterations", () => {
    const q = Float64Array.from(rest);
    const t0 = tcpOf(ROBOT, rest);
    const res = solver.solve(q, [t0[0] + 0.01, t0[1] - 0.01, t0[2] + 0.01], { posture: rest });
    expect(res.converged).toBe(true);
    expect(res.iterations).toBeLessThanOrEqual(6);
  });

  it("handles unreachable targets: no convergence claim, valid pose, moves toward goal, no oscillation", () => {
    const target: [number, number, number] = [6, 4, -5];
    const q = Float64Array.from(rest);
    const start = tcpOf(ROBOT, q);
    const res = solver.solve(q, target, { maxIterations: 100, posture: rest });
    expect(res.converged).toBe(false);
    expect(isFiniteArray(q)).toBe(true);
    expect(withinLimits(q, ROBOT)).toBe(true);
    const d0 = Math.hypot(start[0] - target[0], start[1] - target[1], start[2] - target[2]);
    expect(res.positionError).toBeLessThan(d0);

    // Repeated solving must settle: error never increases and the per-solve motion dies out.
    let prevErr = res.positionError;
    let lastMove = Infinity;
    for (let k = 0; k < 12; k++) {
      const before = Float64Array.from(q);
      const r = solver.solve(q, target, { maxIterations: 100, posture: rest });
      expect(r.positionError).toBeLessThanOrEqual(prevErr + 1e-6);
      prevErr = r.positionError;
      lastMove = Math.max(...Array.from(q, (v, i) => Math.abs(v - before[i])));
    }
    expect(lastMove).toBeLessThan(0.01);
  });

  it("never violates joint limits for arbitrary targets", () => {
    const rng = mulberry(5);
    for (let k = 0; k < 300; k++) {
      const q = Float64Array.from(rest);
      solver.solve(q, [(rng() - 0.5) * 4, (rng() - 0.3) * 3, (rng() - 0.5) * 4], { posture: rest });
      expect(withinLimits(q, ROBOT)).toBe(true);
      expect(isFiniteArray(q)).toBe(true);
    }
  });

  it("satisfies a tool-direction objective (gripper pointing down)", () => {
    const q = Float64Array.from(rest);
    const res = solver.solve(q, [0.7, 0.3, 0.5], {
      maxIterations: 200,
      posture: rest,
      direction: [0, -1, 0],
      directionWeight: 0.5,
    });
    expect(res.converged).toBe(true);
    const fk = new ForwardKinematics(ROBOT).compute(q);
    expect(fk.toolDirection[1]).toBeLessThan(-0.99);
  });

  it("produces no NaN at singular targets (on the base axis, at the shoulder)", () => {
    for (const t of [
      [0, 1.2, 0],
      [0, 0.35, 0],
      [0, 2.5, 0],
    ] as [number, number, number][]) {
      const q = Float64Array.from([0, 0, 0, 0, 0, 0]);
      solver.solve(q, t, { maxIterations: 50 });
      expect(isFiniteArray(q)).toBe(true);
    }
  });
});
