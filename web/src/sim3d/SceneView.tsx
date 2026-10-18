import { Component, lazy, Suspense, useMemo, useState, type ReactNode } from "react";
import type { Condition } from "../api/types";
import { useSimModel, useSimState } from "./api";
import "../landing/landing.css";
import "@fontsource/ibm-plex-mono/400.css";

const SimViewport = lazy(() => import("./SimViewport"));

export type RenderMode = "stylized" | "camera";

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

class Boundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="kn-mono flex h-full w-full items-center justify-center bg-white text-[10px] uppercase tracking-[0.14em] text-[#858585]">
      {children}
    </div>
  );
}

/**
 * A scene viewport with a "RENDER: STYLIZED / CAMERA" switch. STYLIZED draws MuJoCo's body poses in
 * the Kinetic line-art style; CAMERA is the MuJoCo render the planner actually sees (`camera`).
 * During a live recording the default is CAMERA, because only camera frames stream from workers.
 */
export function SceneView({
  pid,
  seed = 0,
  condition,
  live = false,
  sceneKey,
  camera,
  className,
}: {
  pid: string;
  seed?: number;
  /** Plan condition for the upstream tabletop (used until the project's scene is built). */
  condition?: Condition;
  live?: boolean;
  /** Changes when the project's scene is rebuilt (e.g. its digest), so poses refetch. */
  sceneKey?: string;
  camera: ReactNode;
  className?: string;
}) {
  const webgl = useMemo(hasWebGL, []);
  const [choice, setChoice] = useState<RenderMode | null>(null);
  const mode: RenderMode = !webgl ? "camera" : (choice ?? (live ? "camera" : "stylized"));
  const stylized = mode === "stylized";
  const state = useSimState(pid, seed, condition, live && stylized, sceneKey);
  const model = useSimModel(pid, stylized ? state.data?.scene : undefined);

  let body: ReactNode = camera;
  if (stylized) {
    if (state.error || model.error) body = <Notice>Sim state unavailable · {String((state.error ?? model.error) as Error)}</Notice>;
    else if (!state.data || !model.data) body = <Notice>Loading MuJoCo state…</Notice>;
    else
      body = (
        <Boundary fallback={<Notice>3D view failed · use camera</Notice>}>
          <Suspense fallback={<Notice>Loading viewer…</Notice>}>
            <SimViewport model={model.data} state={state.data} note={live ? "Live · poses at reset, camera is live" : undefined} />
          </Suspense>
        </Boundary>
      );
  }

  return (
    <div className={`relative h-full w-full ${className ?? ""}`}>
      {body}
      <RenderToggle mode={mode} onChange={setChoice} disabled={!webgl} />
    </div>
  );
}

export function RenderToggle({ mode, onChange, disabled }: { mode: RenderMode; onChange: (m: RenderMode) => void; disabled?: boolean }) {
  const opt = (m: RenderMode, label: string) => (
    <button
      type="button"
      onClick={() => onChange(m)}
      disabled={disabled && m === "stylized"}
      aria-pressed={mode === m}
      className={`px-1.5 py-0.5 transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        mode === m ? "bg-[#171717] text-white" : "text-[#555] hover:text-[#171717]"
      }`}
    >
      {label}
    </button>
  );
  return (
    <div
      className="kn-mono absolute right-2.5 top-2.5 z-10 flex items-center gap-1 border border-[#d4d4d4] bg-white/90 py-0.5 pl-1.5 pr-0.5 text-[9.5px] uppercase tracking-[0.12em] text-[#858585]"
      role="group"
      aria-label="Render mode"
      data-testid="render-toggle"
    >
      <span>Render:</span>
      {opt("stylized", "Stylized")}
      <span aria-hidden>/</span>
      {opt("camera", "Camera")}
    </div>
  );
}
