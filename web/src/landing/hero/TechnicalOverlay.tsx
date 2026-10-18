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
            <path ref={(el) => void (reg(c.id).path = el)} fill="none" stroke="#171717" strokeWidth={0.75} />
            <circle ref={(el) => void (reg(c.id).node = el)} r={2.6} fill="#fff" stroke="#171717" strokeWidth={0.9} />
          </g>
        ))}
      </svg>
      {CALLOUTS.map((c) => (
        <div
          key={c.id}
          ref={(el) => void (reg(c.id).label = el)}
          className="absolute left-0 top-0 hidden whitespace-nowrap bg-white/85 md:block px-1 py-0.5 kn-mono text-[10px] leading-none tracking-[0.08em] text-[#171717] will-change-transform"
          aria-hidden
        >
          <span>{c.label}</span>
          <span ref={(el) => void (reg(c.id).value = el)} className="ml-2 text-[#858585] tabular-nums" />
        </div>
      ))}

      {/* Status block */}
      <div className="absolute right-[4.5%] top-[17%] hidden kn-mono text-[10px] uppercase leading-[1.9] tracking-[0.12em] text-[#858585] lg:block">
        <div className="flex items-center gap-3 text-[#171717]">
          <span className="h-px w-8 bg-[#171717]" />
          <span>FIG.01</span>
          <span className="text-[#858585]">KN-6 Manipulator</span>
        </div>
        <div className="mt-2 pl-11">Kinematic control</div>
        <div className="flex items-center gap-2 pl-11 text-[#171717]">
          <span className="relative inline-flex h-1.5 w-1.5">
            <span className="absolute inset-0 animate-ping rounded-full bg-[#171717]/40 motion-reduce:hidden" />
            <span className="relative h-1.5 w-1.5 rounded-full bg-[#171717]" />
          </span>
          <span ref={(el) => void (o.status = el)} data-testid="robot-state">
            Initializing
          </span>
        </div>
        <div className="pl-11 tabular-nums" ref={(el) => void (o.ik = el)} data-testid="robot-ik" />
      </div>

      {/* Footer row: affordance hint + the one optional action. */}
      <div className="absolute inset-x-6 bottom-[4.5%] flex items-center justify-between gap-6 kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585] lg:left-[52%] lg:right-[4.5%]">
        <span className="hidden xl:inline">Move the cursor here to command the end effector</span>
        <button
          ref={(el) => void (o.graspButton = el)}
          type="button"
          onClick={() => (rt.cube.requested = true)}
          className="pointer-events-auto group ml-auto inline-flex items-center gap-2 whitespace-nowrap border-b border-[#171717] pb-1 text-[#171717] transition-opacity hover:opacity-70 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-4 focus-visible:outline-[#171717] disabled:cursor-not-allowed disabled:border-[#e6e6e6] disabled:text-[#858585]"
          data-testid="grasp-button"
        >
          Run pick sequence
          <span aria-hidden className="transition-transform group-hover:translate-x-0.5">→</span>
        </button>
      </div>
    </div>
  );
}
