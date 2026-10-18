import { useRuntime } from "../robot/runtime";
import { CALLOUTS } from "./overlaySpec";

/**
 * Engineering annotations around the arm. Structure is rendered once by React;
 * every value and position is written by the render loop from real controller state.
 */
export function TechnicalOverlay() {
  const rt = useRuntime();
  const o = rt.overlay;
  const reg = (id: string) => (o.callouts[id] ??= { path: null, node: null, label: null, value: null });

  return (
    <div className="pointer-events-none absolute inset-0 select-none" data-overlay>
      {/* Leader lines */}
      <svg className="absolute inset-0 hidden h-full w-full overflow-visible md:block" aria-hidden>
        {CALLOUTS.map((c) => (
          <g key={c.id}>
            <path ref={(el) => void (reg(c.id).path = el)} fill="none" stroke="var(--kn-ink)" strokeWidth={0.75} />
            <circle ref={(el) => void (reg(c.id).node = el)} r={2.6} fill="var(--kn-paper)" stroke="var(--kn-ink)" strokeWidth={0.9} />
          </g>
        ))}
      </svg>
      {CALLOUTS.map((c) => (
        <div
          key={c.id}
          ref={(el) => void (reg(c.id).label = el)}
          className="absolute left-0 top-0 hidden whitespace-nowrap bg-bench/85 md:block px-1 py-0.5 kn-mono text-[10px] leading-none tracking-[0.08em] text-ink will-change-transform"
          aria-hidden
        >
          <span>{c.label}</span>
          <span ref={(el) => void (reg(c.id).value = el)} className="ml-2 text-ink-3 tabular-nums" />
        </div>
      ))}

      {/* Footer row: affordance hint + the one optional action. */}
      <div className="absolute inset-x-6 bottom-[4.5%] flex items-center justify-between gap-6 kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3 lg:left-[52%] lg:right-[4.5%]">
        <span className="hidden xl:inline">Move the cursor here to command the end effector</span>
        <button
          ref={(el) => void (o.graspButton = el)}
          type="button"
          onClick={() => (rt.cube.requested = true)}
          className="pointer-events-auto group ml-auto inline-flex items-center gap-2 whitespace-nowrap border-b border-ink pb-1 text-ink transition-opacity hover:opacity-70 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-4 focus-visible:outline-ink disabled:cursor-not-allowed disabled:border-line disabled:text-ink-3"
          data-testid="grasp-button"
        >
          Run pick sequence
          <span aria-hidden className="transition-transform group-hover:translate-x-0.5">→</span>
        </button>
      </div>
    </div>
  );
}
