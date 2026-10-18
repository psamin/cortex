import { describe, expect, it } from "vitest";
import { interactionPlane, intersectRayPlane, pointerToNdc } from "../cursorMapping";
import { clampToWorkspace, workspaceFor } from "../workspace";
import { tcpOf } from "../forwardKinematics";
import { ROBOT } from "../robotConfig";

describe("cursor mapping", () => {
  const rect = { left: 100, top: 50, width: 400, height: 200 };
  it("maps canvas corners and center to NDC", () => {
    expect(pointerToNdc(100, 50, rect)).toEqual([-1, 1]);
    expect(pointerToNdc(500, 250, rect)).toEqual([1, -1]);
    expect(pointerToNdc(300, 150, rect)).toEqual([0, -0]);
  });

  it("intersects a ray with a plane", () => {
    const out = [0, 0, 0];
    expect(intersectRayPlane([0, 1, 5], [0, 0, -1], [0, 0, 0], [0, 0, 1], out)).toBe(true);
    expect(out).toEqual([0, 1, 0]);
  });

  it("rejects near-parallel rays so the previous target is kept", () => {
    const out = [9, 9, 9];
    expect(intersectRayPlane([0, 1, 5], [1, 0, 0.0001], [0, 0, 0], [0, 0, 1], out)).toBe(false);
    expect(out).toEqual([9, 9, 9]);
  });

  it("builds a vertical camera-facing plane offset toward the viewer", () => {
    const fwd = [-0.6, -0.5, -0.6];
    const { point, normal } = interactionPlane(fwd, [0, 0, 0], 0.5);
    expect(normal[1]).toBe(0);
    expect(Math.hypot(...normal)).toBeCloseTo(1, 9);
    expect(normal[0]).toBeGreaterThan(0);
    expect(Math.hypot(point[0], point[2])).toBeCloseTo(0.5, 9);
  });
});

describe("workspace clamping", () => {
  const ws = workspaceFor(ROBOT);
  it("leaves comfortably reachable points untouched", () => {
    const p = tcpOf(ROBOT, ROBOT.rest);
    const out = [0, 0, 0];
    expect(clampToWorkspace(p, ws, out)).toBe(false);
    expect(out).toEqual(p);
  });

  it("pulls far, low and on-axis points into the reachable shell", () => {
    for (const p of [
      [10, 10, 10],
      [0.8, -3, 0.2],
      [0, 1.0, 0],
      [0, 0.35, 0],
    ]) {
      const out = [0, 0, 0];
      clampToWorkspace(p, ws, out);
      const r = Math.hypot(out[0] - ws.center[0], out[1] - ws.center[1], out[2] - ws.center[2]);
      expect(r).toBeLessThanOrEqual(ws.rMax + 1e-9);
      expect(out[1]).toBeGreaterThanOrEqual(ws.minY - 1e-9);
      expect(Math.hypot(out[0], out[2])).toBeGreaterThanOrEqual(ws.minHorizontal - 1e-9);
    }
  });
});
