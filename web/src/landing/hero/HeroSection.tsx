import { Component, lazy, Suspense, useEffect, useState, type ReactNode } from "react";
import { HeroFallback } from "./HeroFallback";

// The only part of the landing page that needs three.js / r3f / rapier / gsap. Everything else renders without them.
const HeroStage = lazy(() => import("./HeroStage"));

type Unavailable = "webgl" | "load";

/** Catches a failed 3D chunk import (deps not installed, offline) or a WebGL crash, and reports why. */
class StageBoundary extends Component<{ onFail: (why: Unavailable) => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    if (import.meta.env.DEV) console.warn("[landing] 3D hero unavailable, showing the still image.", error);
    this.props.onFail("load");
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

const DEV_NOTE: Record<Unavailable, string> = {
  load: "3D hero unavailable · run pnpm install in web/",
  webgl: "3D hero unavailable · no WebGL in this browser",
};

export function HeroSection({ copy }: { copy: ReactNode }) {
  // null until we've checked WebGL on the client; then either the live stage or the still image.
  const [unavailable, setUnavailable] = useState<Unavailable | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!hasWebGL()) setUnavailable("webgl");
    setChecked(true);
  }, []);

  return (
    <section
      aria-labelledby="hero-title"
      className="relative isolate flex min-h-[100svh] flex-col overflow-hidden bg-bench lg:block lg:h-[100svh] lg:min-h-[720px]"
    >
      <div aria-hidden className="kn-splash" />
      {/* Copy layer (left on desktop, top on mobile). Plain DOM; never waits for WebGL. */}
      <div className="relative z-20 lg:absolute lg:inset-y-0 lg:left-0 lg:w-[46%]">{copy}</div>

      {/* Stage: full-bleed canvas on desktop, its own block on mobile. */}
      <div
        className="relative h-[54svh] min-h-[360px] w-full md:h-[60svh] lg:absolute lg:inset-0 lg:h-auto"
        role="img"
        aria-label="Line-art illustration of a six-axis robotic arm. On desktop its gripper follows your cursor over the right half of the hero."
      >
        {unavailable ? (
          <>
            <HeroFallback />
            {import.meta.env.DEV && (
              <p
                className="absolute bottom-3 right-6 z-20 kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3 sm:right-10"
                data-testid="hero-3d-note"
              >
                {DEV_NOTE[unavailable]}
              </p>
            )}
          </>
        ) : (
          checked && (
            <StageBoundary onFail={setUnavailable}>
              <Suspense fallback={null}>
                <HeroStage />
              </Suspense>
            </StageBoundary>
          )
        )}
      </div>
    </section>
  );
}
