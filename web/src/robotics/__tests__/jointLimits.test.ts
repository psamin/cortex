import { describe, expect, it } from "vitest";
import { atLimit, clampToLimits, withinLimits } from "../jointLimits";
import { ROBOT } from "../robotConfig";
import { RobotStateMachine, transition } from "../stateMachine";

describe("joint limits", () => {
  it("rest pose is inside the limits", () => {
    expect(withinLimits(ROBOT.rest, ROBOT)).toBe(true);
  });
  it("clamps out-of-range angles and reports it", () => {
    const q = ROBOT.joints.map((j) => j.max + 1);
    expect(clampToLimits(q, ROBOT)).toBe(true);
    expect(withinLimits(q, ROBOT)).toBe(true);
    q.forEach((_, i) => expect(atLimit(q, ROBOT, i)).toBe(true));
  });
  it("every joint has a valid range and positive dynamics limits", () => {
    for (const j of ROBOT.joints) {
      expect(j.max).toBeGreaterThan(j.min);
      expect(j.maxVelocity).toBeGreaterThan(0);
      expect(j.maxAcceleration).toBeGreaterThan(0);
      expect(Math.hypot(...j.axis)).toBeCloseTo(1, 12);
    }
  });
});

describe("animation state machine", () => {
  it("runs the hero lifecycle", () => {
    const sm = new RobotStateMachine();
    const seq: Array<[Parameters<typeof sm.send>[0], string]> = [
      ["READY", "INTRO"],
      ["INTRO_DONE", "IDLE"],
      ["POINTER_ENTER", "TRACKING"],
      ["POINTER_LEAVE", "RETURNING"],
      ["POINTER_ENTER", "TRACKING"],
      ["POINTER_LEAVE", "RETURNING"],
      ["REACHED_REST", "IDLE"],
      ["GRASP", "GRASPING"],
      ["GRASP_DONE", "RELEASING"],
      ["RELEASE_DONE", "RETURNING"],
      ["REACHED_REST", "IDLE"],
    ];
    for (const [ev, expected] of seq) {
      sm.send(ev);
      expect(sm.state).toBe(expected);
    }
  });
  it("ignores invalid events (e.g. pointer during a pick sequence)", () => {
    expect(transition("GRASPING", "POINTER_ENTER")).toBe("GRASPING");
    expect(transition("INTRO", "POINTER_ENTER")).toBe("INTRO");
  });
  it("recovers from errors through RETURNING", () => {
    expect(transition("TRACKING", "FAULT")).toBe("ERROR");
    expect(transition("ERROR", "POINTER_ENTER")).toBe("ERROR");
    expect(transition("ERROR", "RECOVER")).toBe("RETURNING");
  });
});
