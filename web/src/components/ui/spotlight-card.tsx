import { useEffect, type CSSProperties, type HTMLAttributes, type ReactNode } from "react";
import { cn } from "../../lib/format";

/*
  Spotlight edge: a thin black-or-white shine that follows the pointer along a box's existing
  1 px outline. Adapted from the owner's GlowCard reference, but monochrome (ink on paper in
  light mode, white on dark) and subtle, so the Kinetic hairlines stay the design.
  Styles live in index.css (.glow-edge / .glow-surface); the pointer position is one
  document-level listener writing --gx / --gy on <html>, shared by every card.
*/

let listeners = 0;
function onPointer(e: PointerEvent) {
  const s = document.documentElement.style;
  s.setProperty("--gx", e.clientX.toFixed(1));
  s.setProperty("--gy", e.clientY.toFixed(1));
}

/** Mount once near the root. Fine-pointer devices only; touch screens keep plain outlines. */
export function useGlowPointer() {
  useEffect(() => {
    if (!window.matchMedia?.("(pointer: fine)").matches) return;
    if (listeners++ === 0) document.addEventListener("pointermove", onPointer, { passive: true });
    return () => {
      if (--listeners === 0) document.removeEventListener("pointermove", onPointer);
    };
  }, []);
}

/** Drop inside any `relative` box with a border to give that border the spotlight. `surface` adds a faint inner wash. */
export function GlowEdge({ surface = false }: { surface?: boolean }) {
  return (
    <>
      {surface && <span aria-hidden className="glow-surface" />}
      <span aria-hidden className="glow-edge" />
    </>
  );
}

export interface GlowCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  /** Spotlight radius in px (default 220). */
  size?: number;
}

/** A bordered info box with the spotlight edge. Size it with className like any div. */
export function GlowCard({ children, className, size, style, ...rest }: GlowCardProps) {
  return (
    <div
      {...rest}
      className={cn("relative rounded border border-line-strong bg-plate", className)}
      style={size ? ({ ...style, "--glow-size": `${size}px` } as CSSProperties) : style}
    >
      <GlowEdge />
      {children}
    </div>
  );
}
