import type { RobotEvent, RobotState } from "./types";

const TABLE: Record<RobotState, Partial<Record<RobotEvent, RobotState>>> = {
  INITIALIZING: { READY: "INTRO", FAULT: "ERROR" },
  INTRO: { INTRO_DONE: "IDLE", FAULT: "ERROR" },
  IDLE: { POINTER_ENTER: "TRACKING", GRASP: "GRASPING", FAULT: "ERROR" },
  TRACKING: { POINTER_LEAVE: "RETURNING", GRASP: "GRASPING", FAULT: "ERROR" },
  RETURNING: {
    REACHED_REST: "IDLE",
    POINTER_ENTER: "TRACKING",
    GRASP: "GRASPING",
    FAULT: "ERROR",
  },
  // A pick sequence runs to completion; pointer events are ignored meanwhile.
  GRASPING: { GRASP_DONE: "RELEASING", FAULT: "ERROR" },
  RELEASING: { RELEASE_DONE: "RETURNING", FAULT: "ERROR" },
  ERROR: { RECOVER: "RETURNING" },
};

/** Pure transition function. Invalid events leave the state unchanged. */
export function transition(state: RobotState, event: RobotEvent): RobotState {
  return TABLE[state][event] ?? state;
}

export function canTransition(state: RobotState, event: RobotEvent): boolean {
  return TABLE[state][event] !== undefined;
}

/** Small stateful wrapper that remembers when the current state was entered. */
export class RobotStateMachine {
  state: RobotState = "INITIALIZING";
  enteredAt = 0;
  private listeners = new Set<(s: RobotState) => void>();

  send(event: RobotEvent, now = 0): boolean {
    const next = transition(this.state, event);
    if (next === this.state) return false;
    this.state = next;
    this.enteredAt = now;
    this.listeners.forEach((l) => l(next));
    return true;
  }

  subscribe(fn: (s: RobotState) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
