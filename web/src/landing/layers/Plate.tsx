import { memo } from "react";
import { box, path, proj, ring, type P3 } from "./iso";
import type { GlyphKind } from "./steps";

export const INK = "var(--kn-ink)";
export const GREY = "var(--kn-grey)";
export const LINE = "var(--line)";
export const ACCENT = "var(--kn-accent)";

/** Plate half-size and thickness (plate-local units). */
export const H = 110;
export const T = 7;

export const CORNER = {
  left: proj([-H, H, 0]),
  right: proj([H, -H, 0]),
  front: proj([H, H, 0]),
  back: proj([-H, -H, 0]),
};

const TOP = path([[-H, -H, 0], [H, -H, 0], [H, H, 0], [-H, H, 0]], true);
const SIDE_V = path([[-H, H, 0], [H, H, 0], [H, H, -T], [-H, H, -T]], true);
const SIDE_U = path([[H, H, 0], [H, -H, 0], [H, -H, -T], [H, H, -T]], true);
const HALO = path(ring(0, 0, 96, 0, Math.PI * 2, 96));

export type PlateTone = "idle" | "active" | "dim";

const TONES = {
  idle: { top: "var(--kn-paper)", sideV: "var(--kn-face-v)", sideU: "var(--kn-face-u)", edge: INK, glyph: GREY, glyphFill: "var(--kn-face-v)" },
  dim: { top: "var(--kn-paper)", sideV: "var(--kn-dim-v)", sideU: "var(--kn-dim-u)", edge: "var(--kn-faint)", glyph: "var(--kn-faint-2)", glyphFill: "var(--kn-dim-v)" },
  active: { top: "var(--kn-accent-top)", sideV: ACCENT, sideU: "var(--kn-accent-deep)", edge: ACCENT, glyph: ACCENT, glyphFill: "var(--kn-accent-fill)" },
} as const;

const ease = "fill 420ms cubic-bezier(0.23,1,0.32,1), stroke 420ms cubic-bezier(0.23,1,0.32,1), opacity 420ms ease";

/** One line-art plate with its glyph. Drawn around (0, 0); place it with a parent transform. */
export const Plate = memo(function Plate({ glyph, tone }: { glyph: GlyphKind; tone: PlateTone }) {
  const c = TONES[tone];
  const s = { transition: ease };
  return (
    <g strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke">
      <path d={SIDE_V} style={{ ...s, fill: c.sideV, stroke: c.edge }} strokeWidth={1} />
      <path d={SIDE_U} style={{ ...s, fill: c.sideU, stroke: c.edge }} strokeWidth={1} />
      <path d={TOP} style={{ ...s, fill: c.top, stroke: c.edge }} strokeWidth={1} />
      <path
        d={HALO}
        fill="none"
        strokeWidth={0.8}
        strokeDasharray="3 4"
        style={{ ...s, stroke: ACCENT, opacity: tone === "active" ? 0.55 : 0 }}
      />
      <Glyph kind={glyph} stroke={c.glyph} fill={c.glyphFill} top={c.top} />
    </g>
  );
});

function Glyph({ kind, stroke, fill, top }: { kind: GlyphKind; stroke: string; fill: string; top: string }) {
  const st = { transition: ease, stroke };
  const line = (pts: P3[], extra: Record<string, unknown> = {}) => (
    <path d={path(pts)} fill="none" strokeWidth={1} style={st} {...extra} />
  );

  switch (kind) {
    case "card": {
      // A robot description card (arm, cameras, objects, zones, task) and a typed question beside it.
      const rows = [-48, -30, -12, 6, 24];
      const lens = [56, 40, 62, 34, 48];
      return (
        <g>
          <path d={path([[-70, -52, 0], [38, -52, 0], [38, 52, 0], [-70, 52, 0]], true)} strokeWidth={1} style={{ ...st, fill }} />
          {rows.map((u, i) => line([[u, -38, 0], [u, -38 + lens[i], 0]], { key: u }))}
          {line([[-62, -40, 0], [-62, -44, 0]])}
          <path
            d={path([[52, -26, 0], [86, -26, 0], [86, 26, 0], [52, 26, 0]], true)}
            fill="none"
            strokeWidth={1}
            strokeDasharray="3 3"
            style={st}
          />
          {line([[69, -6, 0], [69, 8, 0]])}
        </g>
      );
    }
    case "grid": {
      // Scene grid with a cube and the arm's reach envelope (dashed).
      const ticks = [-88, -44, 0, 44, 88];
      const b = box(4, 34, -30, 0, 26);
      return (
        <g>
          {ticks.map((t) => line([[t, -88, 0], [t, 88, 0]], { key: `u${t}`, strokeOpacity: 0.55 }))}
          {ticks.map((t) => line([[-88, t, 0], [88, t, 0]], { key: `v${t}`, strokeOpacity: 0.55 }))}
          <path d={path(ring(-40, 40, 62, -Math.PI * 0.95, Math.PI * 0.45, 48))} fill="none" strokeWidth={1} strokeDasharray="4 4" style={st} />
          <path d={b.faceV} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.faceU} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.top} strokeWidth={1} style={{ ...st, fill: top }} />
        </g>
      );
    }
    case "loop": {
      // The cycle: home → pick → place → home, with a reset.
      const r = 62;
      const arc = ring(0, 0, r, -Math.PI * 0.5 + 0.25, Math.PI * 1.5 - 0.25, 72);
      const end = arc[arc.length - 1];
      const t = Math.PI * 1.5 - 0.25;
      const [tu, tv] = [-Math.sin(t), Math.cos(t)]; // direction of travel
      const [nu, nv] = [Math.cos(t), Math.sin(t)]; // outward normal
      const head: P3[] = [
        [end[0] - tu * 13 + nu * 7, end[1] - tv * 13 + nv * 7, 0],
        end,
        [end[0] - tu * 13 - nu * 7, end[1] - tv * 13 - nv * 7, 0],
      ];
      const nodes = [0, 0.5, 1, 1.5].map((k) => proj([Math.cos(Math.PI * k) * r, Math.sin(Math.PI * k) * r, 0]));
      return (
        <g>
          <path d={path(arc)} fill="none" strokeWidth={1.1} style={st} />
          <path d={path(head)} fill="none" strokeWidth={1.1} style={st} />
          {nodes.map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r={3.4} strokeWidth={1} style={{ ...st, fill: top }} />
          ))}
          <circle cx={0} cy={0} r={2} style={{ transition: ease, fill: stroke }} />
        </g>
      );
    }
    case "loss": {
      // A loss curve on a standing panel.
      const pts: P3[] = [];
      for (let u = -80; u <= 80; u += 8) pts.push([u, 30, 6 + 62 * Math.exp(-(u + 80) / 42) + 3 * Math.sin(u / 5)]);
      return (
        <g>
          <path d={path([[-80, 30, 0], [80, 30, 0], [80, 30, 76], [-80, 30, 76]], true)} strokeWidth={1} style={{ ...st, fill: top, strokeOpacity: 0.5 }} />
          {[19, 38, 57].map((z) => line([[-80, 30, z], [80, 30, z]], { key: z, strokeOpacity: 0.35, strokeDasharray: "2 4" }))}
          <path d={path(pts)} fill="none" strokeWidth={1.3} style={st} />
          {line([[-80, 30, 0], [-80, 30, 76]])}
          {line([[-80, 30, 0], [80, 30, 0]])}
        </g>
      );
    }
    case "ci": {
      // Interval bars: point estimate and a confidence interval per branch.
      const bars = [
        [-64, 22, 44, 34],
        [-32, 38, 62, 50],
        [0, 14, 36, 25],
        [32, 46, 66, 57],
        [64, 30, 54, 42],
      ];
      return (
        <g>
          {line([[-84, 20, 0], [84, 20, 0]])}
          {bars.map(([u, lo, hi, m]) => (
            <g key={u}>
              {line([[u, 20, 0], [u, 20, lo]], { strokeOpacity: 0.35, strokeDasharray: "2 3" })}
              {line([[u, 20, lo], [u, 20, hi]], { strokeWidth: 1.3 })}
              {line([[u - 6, 20, hi], [u + 6, 20, hi]])}
              {line([[u - 6, 20, lo], [u + 6, 20, lo]])}
              {(() => {
                const [x, y] = proj([u, 20, m]);
                return <circle cx={x} cy={y} r={2.8} strokeWidth={1} style={{ ...st, fill: top }} />;
              })()}
            </g>
          ))}
        </g>
      );
    }
    case "branch": {
      // A branching path: the weakest condition gets more data (dashed return loop).
      const ends: P3[] = [
        [70, -56, 0],
        [80, 0, 0],
        [70, 56, 0],
      ];
      return (
        <g>
          {line([[-84, 0, 0], [-20, 0, 0]], { strokeWidth: 1.2 })}
          {ends.map((e, i) => line([[-20, 0, 0], [20, e[1] * 0.6, 0], e], { key: i, strokeWidth: i === 2 ? 1.3 : 1 }))}
          {ends.map((e, i) => {
            const [x, y] = proj(e);
            return <circle key={i} cx={x} cy={y} r={3.4} strokeWidth={1} style={{ ...st, fill: i === 2 ? stroke : top }} />;
          })}
          <path d={path(ring(10, 56, 60, Math.PI * 0.05, Math.PI * 0.95, 32))} fill="none" strokeWidth={1} strokeDasharray="3 4" style={st} />
          {(() => {
            const [x, y] = proj([-84, 0, 0]);
            return <circle cx={x} cy={y} r={3.4} strokeWidth={1} style={{ ...st, fill: top }} />;
          })()}
        </g>
      );
    }
    case "chip": {
      // The policy as a chip: an extruded die with pins.
      const b = box(-40, 40, -40, 40, 12);
      const inner = path([[-22, -22, 12], [22, -22, 12], [22, 22, 12], [-22, 22, 12]], true);
      const pins = [-28, -10, 10, 28];
      return (
        <g>
          {pins.map((p) => line([[p, -40, 0], [p, -60, 0]], { key: `a${p}` }))}
          {pins.map((p) => line([[-40, p, 0], [-60, p, 0]], { key: `b${p}` }))}
          <path d={b.faceV} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.faceU} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.top} strokeWidth={1} style={{ ...st, fill: top }} />
          <path d={inner} fill="none" strokeWidth={1} style={st} />
          {pins.map((p) => line([[p, 40, 0], [p, 60, 0]], { key: `c${p}` }))}
          {pins.map((p) => line([[40, p, 0], [60, p, 0]], { key: `d${p}` }))}
        </g>
      );
    }
  }
}
