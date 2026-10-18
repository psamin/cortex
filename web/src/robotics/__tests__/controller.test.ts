import { describe, expect, it } from "vitest";
import { RobotController } from "../controller";
import { withinLimits } from "../jointLimits";
import { maxJointError } from "../motionController";
import { ROBOT } from "../robotConfig";
import { pickAndPlaceWaypoints, STATIONS } from "../graspController";

const run = (c: RobotController, seconds: number, hz = 60, each?: () => void) => {
  for (let t = 0; t < seconds; t += 1 / hz) {
    c.update(1 / hz);
    expect(withinLimits(c.q, ROBOT)).toBe(true);
    each?.();
  }
};

function idleController() {
  const c = new RobotController(ROBOT);
  c.send("READY");
  c.send("INTRO_DONE");
  return c;
}

describe("robot controller (integration)", () => {
  it("intro unfolds from a folded pose to rest", () => {
    const c = new RobotController(ROBOT);
    c.setInitialPose([ROBOT.rest[0], 0.3, -1.6, 0, 1.0, 0]);
    c.send("READY");
    expect(c.state).toBe("INTRO");
    run(c, 5);
    expect(maxJointError(c.q, ROBOT.rest)).toBeLessThan(0.01);
  });

  it("cursor tracking moves the end effector to the target", () => {
    const c = idleController();
    c.send("POINTER_ENTER");
    expect(c.state).toBe("TRACKING");
    const goal: [number, number, number] = [-0.2, 0.9, 0.75];
    c.setPointerTarget(goal);
    run(c, 4);
    const tcp = c.fk.tcpPosition;
    expect(Math.hypot(tcp[0] - goal[0], tcp[1] - goal[1], tcp[2] - goal[2])).toBeLessThan(0.01);
    expect(c.ik.converged).toBe(true);
  });

  it("motion is continuous: no frame-to-frame jumps while chasing a jumping cursor", () => {
    const c = idleController();
    c.send("POINTER_ENTER");
    const targets: [number, number, number][] = [
      [-0.8, 0.3, 0.4],
      [0.1, 1.4, 0.8],
      [-0.5, 0.1, 1.0],
      [-1.4, 1.0, -0.2],
    ];
    let prev = Float64Array.from(c.q);
    let k = 0;
    run(c, 6, 60, () => {
      if (k % 90 === 0) c.setPointerTarget(targets[Math.floor(k / 90) % targets.length]);
      k++;
      ROBOT.joints.forEach((j, i) => expect(Math.abs(c.q[i] - prev[i])).toBeLessThanOrEqual(j.maxVelocity / 60 + 1e-9));
      prev = Float64Array.from(c.q);
    });
  });

  it("unreachable cursor targets are clamped and do not destabilize the arm", () => {
    const c = idleController();
    c.send("POINTER_ENTER");
    c.setPointerTarget([8, 6, 8]);
    expect(c.targetClamped).toBe(true);
    run(c, 4);
    const a = Float64Array.from(c.q);
    run(c, 1);
    expect(maxJointError(a, c.q)).toBeLessThan(0.01);
  });

  it("returns to rest after the pointer leaves, then idles", () => {
    const c = idleController();
    c.send("POINTER_ENTER");
    c.setPointerTarget([-0.9, 1.1, 0.2]);
    run(c, 3);
    c.send("POINTER_LEAVE");
    expect(c.state).toBe("RETURNING");
    let reachedIdle = false;
    run(c, 6, 60, () => {
      // At the moment of the RETURNING -> IDLE hand-off the arm must be at rest.
      if (!reachedIdle && c.state === "IDLE") {
        reachedIdle = true;
        expect(maxJointError(c.q, ROBOT.rest)).toBeLessThan(0.01);
      }
    });
    expect(reachedIdle).toBe(true);
    const t = c.fk.tcpPosition;
    expect(Math.hypot(t[0] - c.restTcp[0], t[1] - c.restTcp[1], t[2] - c.restTcp[2])).toBeLessThan(0.06);
  });

  it("idle motion is small and slow", () => {
    const c = idleController();
    let maxDev = 0;
    run(c, 12, 60, () => {
      const t = c.fk.tcpPosition;
      maxDev = Math.max(maxDev, Math.hypot(t[0] - c.restTcp[0], t[1] - c.restTcp[1], t[2] - c.restTcp[2]));
    });
    expect(maxDev).toBeGreaterThan(0.005);
    expect(maxDev).toBeLessThan(0.06);
  });

  it("recovers from injected NaNs without propagating them", () => {
    const c = idleController();
    run(c, 0.5);
    c.qDesired[2] = NaN;
    c.update(1 / 60);
    expect(c.state).toBe("ERROR");
    expect(c.q.every(Number.isFinite)).toBe(true);
    run(c, 1);
    expect(["RETURNING", "IDLE"]).toContain(c.state);
  });

  it("runs a pick-and-place: aligns fingers, attaches only near the cube, releases, returns", () => {
    const c = idleController();
    const cube = STATIONS[0];
    const yaw = 0.3;
    let attached = false;
    let released = false;
    let yawErrAtAttach = Infinity;
    const ok = c.startGrasp(
      pickAndPlaceWaypoints(cube, yaw, STATIONS[1]),
      () => {
        const t = c.fk.tcpPosition;
        attached = Math.hypot(t[0] - cube[0], t[1] - cube[1], t[2] - cube[2]) < 0.025 && c.fk.toolDirection[1] < -0.95;
        const e = c.fingerYaw(c.q) - yaw;
        const q = Math.PI / 2;
        yawErrAtAttach = Math.abs(e - q * Math.round(e / q));
        return attached;
      },
      () => (released = true),
    );
    expect(ok).toBe(true);
    expect(c.state).toBe("GRASPING");
    c.send("POINTER_ENTER"); // ignored during the sequence
    expect(c.state).toBe("GRASPING");
    run(c, 30);
    expect(attached).toBe(true);
    expect(yawErrAtAttach).toBeLessThan(0.03);
    expect(released).toBe(true);
    expect(["RETURNING", "IDLE"]).toContain(c.state);
  });

  it("refuses to attach when the TCP never reaches the cube (no teleporting objects)", () => {
    const c = idleController();
    let attachCalls = 0;
    // Cube far outside the reach: the sequence times out and the attach check fails.
    const far: [number, number, number] = [3, 0.02, 3];
    c.startGrasp(pickAndPlaceWaypoints(far, 0, STATIONS[1]), () => {
      attachCalls++;
      const t = c.fk.tcpPosition;
      return Math.hypot(t[0] - far[0], t[1] - far[1], t[2] - far[2]) < 0.025;
    });
    run(c, 30);
    expect(c.grasp).toBeNull();
    expect(["RETURNING", "IDLE"]).toContain(c.state);
    expect(attachCalls).toBeLessThanOrEqual(1);
  });
});
