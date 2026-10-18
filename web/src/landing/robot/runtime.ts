import { createContext, useContext } from "react";
import { RobotController } from "../../robotics/controller";
import { ROBOT } from "../../robotics/robotConfig";

export interface CalloutRefs {
  path: SVGPathElement | null;
  node: SVGCircleElement | null;
  label: HTMLDivElement | null;
  value: HTMLSpanElement | null;
  /** Cached label width (px), refreshed with the text. */
  width?: number;
}

/**
 * Mutable, non-React runtime shared by the canvas, DOM overlays and pointer handlers.
 * High-frequency data lives here; React only renders structure.
 */
export interface RobotRuntime {
  controller: RobotController;
  pointer: { inside: boolean; ndcX: number; ndcY: number; dirty: boolean };
  canvas: HTMLCanvasElement | null;
  overlay: {
    callouts: Record<string, CalloutRefs>;
    status: HTMLElement | null;
    ik: HTMLElement | null;
    joints: Array<{ value: HTMLElement | null; bar: HTMLElement | null }>;
    graspButton: HTMLButtonElement | null;
    lastTextUpdate: number;
  };
  /** Intro progress values (0..1), driven by GSAP and read in the render loop. */
  reveal: { base: number; links: number[]; annotations: number };
  stats: { fps: number; frames: number; acc: number };
  cube: { attached: boolean; requested: boolean; placeIndex: number; status: string };
  reducedMotion: boolean;
}

export function createRuntime(): RobotRuntime {
  const controller = new RobotController(ROBOT);
  return {
    controller,
    pointer: { inside: false, ndcX: 0, ndcY: 0, dirty: false },
    canvas: null,
    overlay: {
      callouts: {},
      status: null,
      ik: null,
      joints: ROBOT.joints.map(() => ({ value: null, bar: null })),
      graspButton: null,
      lastTextUpdate: 0,
    },
    reveal: { base: 0, links: ROBOT.joints.map(() => 0), annotations: 0 },
    stats: { fps: 0, frames: 0, acc: 0 },
    cube: { attached: false, requested: false, placeIndex: 0, status: "STBY" },
    reducedMotion: false,
  };
}

export const RuntimeContext = createContext<RobotRuntime | null>(null);

export function useRuntime(): RobotRuntime {
  const r = useContext(RuntimeContext);
  if (!r) throw new Error("RuntimeContext missing");
  return r;
}

/* Dev-only debug toggles (tiny external store for useSyncExternalStore). */
export interface DebugFlags {
  grid: boolean;
  edges: boolean;
  outlines: boolean;
  axes: boolean;
  workspace: boolean;
  colliders: boolean;
}
let flags: DebugFlags = { grid: true, edges: true, outlines: true, axes: false, workspace: false, colliders: false };
const subs = new Set<() => void>();
export const debugStore = {
  get: () => flags,
  set: (patch: Partial<DebugFlags>) => {
    flags = { ...flags, ...patch };
    subs.forEach((s) => s());
  },
  subscribe: (fn: () => void) => {
    subs.add(fn);
    return () => subs.delete(fn);
  },
};
