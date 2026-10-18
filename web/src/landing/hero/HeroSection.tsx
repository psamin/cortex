import { Component, lazy, Suspense, useEffect, useRef, useState, type PointerEvent, type ReactNode } from "react";
import gsap from "gsap";
import { pointerToNdc } from "../../robotics/cursorMapping";
import { maxJointError } from "../../robotics/motionController";
import { ROBOT } from "../../robotics/robotConfig";
import { createRuntime, RuntimeContext, type RobotRuntime } from "../robot/runtime";
import { TechnicalOverlay } from "./TechnicalOverlay";
import { HeroFallback } from "./HeroFallback";

const RobotCanvas = lazy(() => import("./RobotCanvas"));
const DevPanel = import.meta.env.DEV ? lazy(() => import("./DevPanel")) : () => null;

/** If the WebGL scene throws (lost context, driver bug), show the still image instead. */
class CanvasBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <HeroFallback /> : this.props.children;
  }
}

/** Stowed pose the intro unfolds from. */
const STOWED = [ROBOT.rest[0], 0.38, -1.95, 0, 1.25, 0];

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Read-only state probe used by the browser tests (no control surface). */
function exposeProbe(rt: RobotRuntime) {
  const c = rt.controller;
  (window as unknown as { __kinetic: unknown }).__kinetic = {
    get state() {
      return c.state;
    },
    get q() {
      return Array.from(c.q);
    },
    get tcp() {
      return Array.from(c.fk.tcpPosition);
    },
    get target() {
      return Array.from(c.rawTarget);
    },
    get ik() {
      return { ...c.ik };
    },
    get gripper() {
      return c.gripper;
    },
    get cube() {
      return { ...rt.cube };
    },
    get fps() {
      return rt.stats.fps;
    },
    limits: ROBOT.joints.map((j) => [j.min, j.max]),
  };
}

export function HeroSection({ copy }: { copy: ReactNode }) {
  const runtimeRef = useRef<RobotRuntime | null>(null);
  runtimeRef.current ??= createRuntime();
  const rt = runtimeRef.current;
  const stageRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [webgl, setWebgl] = useState<boolean | null>(null);

  useEffect(() => {
    const ok = hasWebGL();
    setWebgl(ok);
    exposeProbe(rt);
    if (!ok) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    rt.reducedMotion = reduced;
    const c = rt.controller;
    if (reduced) {
      c.tuning = { ...c.tuning, idleAmplitude: 0 };
      rt.reveal.base = 1;
      rt.reveal.links.fill(1);
      gsap.set([stageRef.current, overlayRef.current], { opacity: 1 });
      c.send("READY");
      c.send("INTRO_DONE");
      return;
    }

    c.setInitialPose(STOWED);
    c.send("READY");
    const links = Object.fromEntries(rt.reveal.links.map((_, i) => [`l${i}`, 0])) as Record<string, number>;
    const tl = gsap.timeline({ defaults: { ease: "power2.inOut" } });
    tl.to(stageRef.current, { opacity: 1, duration: 0.6 }, 0)
      .to(rt.reveal, { base: 1, duration: 0.55 }, 0.2)
      .to(
        links,
        {
          ...Object.fromEntries(Object.keys(links).map((k) => [k, 1])),
          duration: 0.5,
          stagger: 0.08,
          onUpdate: () => rt.reveal.links.forEach((_, i) => (rt.reveal.links[i] = links[`l${i}`])),
        },
        0.4,
      )
      .to(overlayRef.current, { opacity: 1, duration: 0.6 }, 1.2);

    // Hand over to IDLE once the arm has actually unfolded to rest.
    const poll = window.setInterval(() => {
      if (c.state === "INTRO" && tl.progress() > 0.85 && maxJointError(c.q, ROBOT.rest) < 0.03) {
        c.send("INTRO_DONE");
        window.clearInterval(poll);
      }
    }, 100);
    return () => {
      tl.kill();
      window.clearInterval(poll);
    };
  }, [rt]);

  const setPointer = (e: PointerEvent<HTMLDivElement>) => {
    const rect = (rt.canvas ?? e.currentTarget).getBoundingClientRect();
    const [x, y] = pointerToNdc(e.clientX, e.clientY, rect);
    rt.pointer.ndcX = x;
    rt.pointer.ndcY = y;
    rt.pointer.dirty = true;
  };
  const zoneHandlers = {
    onPointerEnter: (e: PointerEvent<HTMLDivElement>) => {
      if (e.pointerType !== "mouse") return;
      setPointer(e);
      rt.pointer.inside = true;
    },
    onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
      if (e.pointerType !== "mouse" && e.buttons === 0) return;
      setPointer(e);
      rt.pointer.inside = true;
    },
    onPointerDown: (e: PointerEvent<HTMLDivElement>) => {
      setPointer(e);
      rt.pointer.inside = true;
    },
    onPointerUp: (e: PointerEvent<HTMLDivElement>) => {
      if (e.pointerType !== "mouse") rt.pointer.inside = false;
    },
    onPointerCancel: () => (rt.pointer.inside = false),
    onPointerLeave: () => (rt.pointer.inside = false),
  };

  return (
    <RuntimeContext.Provider value={rt}>
      <section className="relative isolate flex min-h-[100svh] flex-col overflow-hidden bg-white lg:block lg:h-[100svh] lg:min-h-[720px]">
        {/* Copy layer (left on desktop, top on mobile). Plain DOM; never waits for WebGL. */}
        <div className="relative z-20 lg:absolute lg:inset-y-0 lg:left-0 lg:w-[46%]">{copy}</div>

        {/* Stage: full-bleed canvas on desktop, its own block on mobile. Never captures pointer events. */}
        <div
          className="relative h-[54svh] min-h-[360px] w-full md:h-[60svh] lg:absolute lg:inset-0 lg:h-auto"
          role="img"
          aria-label="Interactive line-art illustration of a six-axis robotic arm. On desktop its gripper follows your cursor over the right half of the hero."
        >
          <div ref={stageRef} className="absolute inset-0 opacity-0" data-testid="robot-stage">
            {webgl && (
              <CanvasBoundary>
                <Suspense fallback={null}>
                  <RobotCanvas runtime={rt} />
                </Suspense>
              </CanvasBoundary>
            )}
          </div>
          {webgl === false && <HeroFallback />}

          {/* Interaction zone: the only element that listens to the pointer. */}
          <div
            className="absolute inset-0 z-10 touch-pan-y lg:left-[46%] lg:top-[72px]"
            data-testid="interaction-zone"
            {...zoneHandlers}
          />

          <div ref={overlayRef} className="pointer-events-none absolute inset-0 z-20 opacity-0">
            {webgl && <TechnicalOverlay />}
          </div>

        </div>
        {webgl && (
          <Suspense fallback={null}>
            <DevPanel runtime={rt} />
          </Suspense>
        )}
      </section>
    </RuntimeContext.Provider>
  );
}
