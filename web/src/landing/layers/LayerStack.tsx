import { CORNER, GREY, INK, Plate, T } from "./Plate";
import { STEPS } from "./steps";

export const VIEW_W = 760;
export const VIEW_H = 820;
const CX = 290;
const MID = 410;
const GAP_CLOSED = 14;
const GAP_OPEN = 92;
/** While stepping: tight gaps, with a wide one above the focused plate so its glyph is in full view. */
const GAP_TIGHT = 50;
const GAP_FOCUS = 218;
const GAP_BELOW = 36;

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));
const smooth = (t: number) => t * t * (3 - 2 * t);

/** Plate centre lines. `focus` is the continuous step position (step i is centred on i + 0.5); `mix` blends from the uniform explode. */
export function stackYs(explode: number, focus: number, mix: number, uniformGap?: number): number[] {
  const g0 = uniformGap ?? GAP_CLOSED + (GAP_OPEN - GAP_CLOSED) * clamp01(explode);
  const f = Math.min(focus, STEPS.length - 0.5);
  const open = STEPS.map((_, i) => smooth(clamp01(1 - (Math.abs(f - (i + 0.5)) - 0.25) / 0.5)));
  const ys = [0];
  for (let i = 1; i < STEPS.length; i++) {
    const stepGap = GAP_TIGHT + (GAP_FOCUS - GAP_TIGHT) * open[i] + GAP_BELOW * open[i - 1];
    ys.push(ys[i - 1] + g0 * (1 - mix) + stepGap * mix);
  }
  const shift = MID - (ys[0] + ys[ys.length - 1]) / 2;
  return ys.map((y) => y + shift);
}
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

/**
 * The exploded technical drawing: seven plates that separate vertically as `explode` goes 0 → 1,
 * with construction lines between them. `active` (0–6, or -1) highlights one layer and draws its callout.
 */
export function LayerStack({
  explode,
  active,
  focus = -1,
  mix = 0,
  uniformGap,
  className,
  title = `Exploded isometric diagram of the seven stacked layers of the Cortex robot-policy training loop: ${STEPS.map((s) => s.label.toLowerCase()).join(", ")}`,
}: {
  explode: number;
  active: number;
  focus?: number;
  mix?: number;
  uniformGap?: number;
  className?: string;
  title?: string;
}) {
  const e = clamp01(explode);
  const ys = stackYs(e, focus, mix, uniformGap);
  const construct = clamp01((e - 0.25) / 0.5);
  const labels = clamp01((e - 0.55) / 0.4);
  const top = ys[0] + CORNER.back[1] - 22;
  const bottom = ys[6] + CORNER.front[1] + T + 22;
  // Tall static layouts (reduced motion) grow the canvas instead of overlapping it.
  const y0 = Math.min(0, top - 10);
  const h = Math.max(VIEW_H, bottom + 10) - y0;

  return (
    <svg viewBox={`0 ${y0} ${VIEW_W} ${h}`} className={className} role="img" aria-label={title} fontFamily={MONO}>
      {/* Centre line, dash-dot like a drafting axis. */}
      <line x1={CX} x2={CX} y1={top} y2={bottom} stroke={GREY} strokeWidth={0.75} strokeDasharray="10 4 2 4" opacity={0.6} />

      {/* Plates, bottom first so upper plates sit in front. */}
      {[...STEPS.keys()].reverse().map((i) => (
        <g key={STEPS[i].n} transform={`translate(${CX} ${ys[i]})`}>
          <g
            style={{
              transform: `translateY(${active === i ? -10 : 0}px)`,
              transition: "transform 520ms cubic-bezier(0.23,1,0.32,1)",
            }}
          >
            <Plate glyph={STEPS[i].glyph} tone={active === i ? "active" : "idle"} />
          </g>
        </g>
      ))}

      {/* Construction lines in the gaps (left, front, right corners). */}
      <g stroke={GREY} strokeWidth={0.75} strokeDasharray="2 3" opacity={construct}>
        {ys.slice(0, -1).map((y, i) =>
          (["left", "front", "right"] as const).map((k) => {
            const [x, cy] = CORNER[k];
            const y1 = y + cy + T + (active === i ? -10 : 0) + 2;
            const y2 = ys[i + 1] + cy + (active === i + 1 ? -10 : 0) - 2;
            return y2 > y1 ? <line key={`${i}${k}`} x1={CX + x} x2={CX + x} y1={y1} y2={y2} /> : null;
          }),
        )}
      </g>

      {/* Layer index on the left corners. */}
      <g opacity={labels} fontSize={10} letterSpacing="0.14em">
        {ys.map((y, i) => (
          <g key={i}>
            <line x1={CX + CORNER.left[0] - 30} x2={CX + CORNER.left[0] - 8} y1={y} y2={y} stroke={GREY} strokeWidth={0.75} />
            <text
              x={CX + CORNER.left[0] - 36}
              y={y + 3.5}
              textAnchor="end"
              fill={active === i ? INK : GREY}
              style={{ transition: "fill 300ms" }}
            >
              L{STEPS[i].n}
            </text>
          </g>
        ))}
      </g>

      {active >= 0 && <Callout key={active} y={ys[active] - 10} i={active} />}
    </svg>
  );
}

/** Leader line from the active plate's right corner to a mono readout. */
function Callout({ y, i }: { y: number; i: number }) {
  const s = STEPS[i];
  const x0 = CX + CORNER.right[0];
  const x1 = x0 + 28;
  const y1 = y - 34;
  const x2 = VIEW_W - 6;
  return (
    <g className="kn-callout">
      <path d={`M${x0} ${y} L${x1} ${y1} L${x2} ${y1}`} fill="none" stroke={INK} strokeWidth={0.8} pathLength={1} className="kn-draw" />
      <circle cx={x0} cy={y} r={3} fill="var(--kn-paper)" stroke={INK} strokeWidth={0.9} />
      <text x={x1 + 4} y={y1 - 10} fontSize={12} letterSpacing="0.12em" fill={INK} className="kn-fade">
        {s.n} / {s.label.toUpperCase()}
      </text>
      <text x={x1 + 4} y={y1 + 19} fontSize={10.5} letterSpacing="0.06em" fill={GREY} className="kn-fade">
        {s.readout}
      </text>
    </g>
  );
}
