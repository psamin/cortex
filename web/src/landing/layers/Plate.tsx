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

type St = { transition: string; stroke: string };

/** A standing cylinder (the bottle): body silhouette, then the top ellipse. */
function Cyl({ u, v, r, h, st, fill, top }: { u: number; v: number; r: number; h: number; st: St; fill: string; top: string }) {
  const q = Math.PI / 4;
  const body = [...ring(u, v, r, -q, 3 * q, 24, 0), ...ring(u, v, r, 3 * q, 7 * q, 24, h)];
  return (
    <g>
      <path d={path(body, true)} strokeWidth={1} style={{ ...st, fill }} />
      <path d={path(ring(u, v, r, 0, Math.PI * 2, 32, h), true)} strokeWidth={1} style={{ ...st, fill: top }} />
    </g>
  );
}

/** A small cube on the plate. */
function Cube({ u, v, s, st, fill, top }: { u: number; v: number; s: number; st: St; fill: string; top: string }) {
  const b = box(u - s, u + s, v - s, v + s, s * 2);
  return (
    <g>
      <path d={b.faceV} strokeWidth={1} style={{ ...st, fill }} />
      <path d={b.faceU} strokeWidth={1} style={{ ...st, fill }} />
      <path d={b.top} strokeWidth={1} style={{ ...st, fill: top }} />
    </g>
  );
}

/** A small square on the plate: a target zone (dashed) or one evaluation scene. */
const square = (u: number, v: number, w: number): string =>
  path([[u - w, v - w, 0], [u + w, v - w, 0], [u + w, v + w, 0], [u - w, v + w, 0]], true);

function Zone({ u, v, w, st }: { u: number; v: number; w: number; st: St }) {
  return <path d={square(u, v, w)} fill="none" strokeWidth={1} strokeDasharray="3 3" style={st} />;
}

/** A line-drawn arm in profile: base, turret, upper arm, forearm and an open two-jaw gripper pointing down at `to`. */
function Arm({ u, v, to, st, fill, top, scale = 1 }: { u: number; v: number; to: P3; st: St; fill: string; top: string; scale?: number }) {
  const k = scale;
  const S: P3 = [u, v, 17 * k];
  const W: P3 = [to[0], to[1], to[2] + 12 * k];
  const E: P3 = [u + (W[0] - u) * 0.3, v + (W[1] - v) * 0.3, Math.max(S[2], W[2]) + 24 * k];
  const dx = W[0] - u;
  const dy = W[1] - v;
  const d = Math.hypot(dx, dy) || 1;
  const [pu, pv] = [(dx / d) * 5 * k, (dy / d) * 5 * k];
  const jaw = (sgn: number): P3[] => [
    [W[0] + sgn * pu, W[1] + sgn * pv, W[2]],
    [W[0] + sgn * pu, W[1] + sgn * pv, W[2] - 11 * k],
  ];
  const joint = (p: P3, r: number) => {
    const [x, y] = proj(p);
    return <circle cx={x} cy={y} r={r} strokeWidth={1} style={{ ...st, fill: top }} />;
  };
  return (
    <g>
      <Cyl u={u} v={v} r={15 * k} h={7 * k} st={st} fill={fill} top={top} />
      <Cyl u={u} v={v} r={8 * k} h={14 * k} st={st} fill={fill} top={top} />
      <path d={path([S, E])} fill="none" strokeWidth={3.2} style={st} />
      <path d={path([E, W])} fill="none" strokeWidth={2.6} style={st} />
      <path d={path([[W[0] - pu, W[1] - pv, W[2]], [W[0] + pu, W[1] + pv, W[2]]])} fill="none" strokeWidth={1.6} style={st} />
      <path d={path(jaw(1))} fill="none" strokeWidth={1.4} style={st} />
      <path d={path(jaw(-1))} fill="none" strokeWidth={1.4} style={st} />
      {joint(S, 3.6 * k)}
      {joint(E, 3.6 * k)}
      {joint(W, 2.6 * k)}
    </g>
  );
}

/** A path from a to b that rises `lift` in the middle (a carried object's arc). */
const arc = (a: P3, b: P3, lift: number, n = 16): P3[] =>
  Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t + Math.sin(Math.PI * t) * lift] as P3;
  });

/** Each layer's glyph is a small drawing of what that step actually does. */
function Glyph({ kind, stroke, fill, top }: { kind: GlyphKind; stroke: string; fill: string; top: string }) {
  const st: St = { transition: ease, stroke };
  const line = (pts: P3[], extra: Record<string, unknown> = {}) => (
    <path d={path(pts)} fill="none" strokeWidth={1} style={st} {...extra} />
  );
  const dot = (p: P3, r = 2.6, solid = false) => {
    const [x, y] = proj(p);
    return <circle cx={x} cy={y} r={r} strokeWidth={1} style={{ ...st, fill: solid ? stroke : top }} />;
  };

  switch (kind) {
    case "describe": {
      // An agent chatting with Cortex: the agent's message, Cortex's reply with a question, the input bar.
      const v = -40;
      const panel: P3[] = [[-88, v, 0], [72, v, 0], [72, v, 84], [-88, v, 84]];
      const bubble = (u0: number, u1: number, z0: number, z1: number, solid: boolean) => (
        <path
          d={path([[u0, v, z0], [u1, v, z0], [u1, v, z1], [u0, v, z1]], true)}
          strokeWidth={1}
          style={{ ...st, fill: solid ? fill : top }}
        />
      );
      return (
        <g>
          <path d={path(panel, true)} strokeWidth={1} style={{ ...st, fill: top }} />
          {line([[-88, v, 74], [72, v, 74]], { strokeOpacity: 0.5 })}
          {dot([-80, v, 79], 1.6)}
          {dot([-74, v, 79], 1.6)}
          {dot([-68, v, 79], 1.6)}
          {/* agent: avatar + message */}
          {dot([-78, v, 60], 4.2, true)}
          {bubble(-70, 14, 52, 68, false)}
          {line([[-64, v, 63], [6, v, 63]])}
          {line([[-64, v, 57], [-20, v, 57]])}
          {/* Cortex: reply bubble on the right */}
          {bubble(-24, 62, 28, 46, true)}
          {line([[-18, v, 40], [52, v, 40]])}
          {line([[-18, v, 34], [30, v, 34]])}
          {/* input bar with a caret */}
          {bubble(-80, 62, 6, 18, false)}
          {line([[-72, v, 9], [-72, v, 15]], { strokeWidth: 1.4 })}
          {dot([54, v, 12], 3, true)}
        </g>
      );
    }
    case "simulate": {
      // That layout built as a physics scene: grid floor, the arm, the bottle, the zone and the reach envelope.
      const ticks = [-88, -44, 0, 44, 88];
      return (
        <g>
          {ticks.map((t) => (
            <g key={t}>
              {line([[t, -88, 0], [t, 88, 0]], { strokeOpacity: 0.35 })}
              {line([[-88, t, 0], [88, t, 0]], { strokeOpacity: 0.35 })}
            </g>
          ))}
          <path d={path(ring(-50, 50, 104, -Math.PI / 2 + 0.1, 0.05, 40))} fill="none" strokeWidth={1} strokeDasharray="4 4" style={st} />
          <Zone u={46} v={-38} w={18} st={st} />
          <Cyl u={8} v={10} r={10} h={30} st={st} fill={fill} top={top} />
          <Arm u={-50} v={50} to={[-6, 14, 30]} st={st} fill={fill} top={top} />
        </g>
      );
    }
    case "demonstrate": {
      // One demonstration: the arm carries the bottle from where it stands to the zone.
      const pick: P3 = [8, 10, 34];
      const place: P3 = [46, -38, 22];
      return (
        <g>
          <Zone u={46} v={-38} w={18} st={st} />
          <Cyl u={8} v={10} r={10} h={30} st={st} fill={fill} top={top} />
          <Arm u={-50} v={50} to={[pick[0], pick[1], pick[2] - 12]} st={st} fill={fill} top={top} />
          <path d={path(arc(pick, place, 22))} fill="none" strokeWidth={1.2} strokeDasharray="4 3" style={st} />
          {dot(place, 3.2, true)}
        </g>
      );
    }
    case "train": {
      // The training loss falling, on a standing panel.
      const loss: P3[] = [];
      for (let u = -76; u <= 76; u += 6) loss.push([u, 20, 8 + 58 * Math.exp(-(u + 76) / 38) + 2.5 * Math.sin(u / 4)]);
      return (
        <g>
          <path d={path([[-76, 20, 0], [76, 20, 0], [76, 20, 76], [-76, 20, 76]], true)} strokeWidth={1} style={{ ...st, fill: top, strokeOpacity: 0.5 }} />
          {[19, 38, 57].map((z) => (
            <g key={z}>{line([[-76, 20, z], [76, 20, z]], { strokeOpacity: 0.3, strokeDasharray: "2 4" })}</g>
          ))}
          {line([[-76, 20, 0], [-76, 20, 76]])}
          {line([[-76, 20, 0], [76, 20, 0]])}
          <path d={path(loss)} fill="none" strokeWidth={1.4} style={st} />
        </g>
      );
    }
    case "evaluate": {
      // Three seeded test scenes (baseline, shifted object, dim light), each with a measured success and interval.
      const scenes: [number, number, number, number, boolean][] = [
        [-58, 58, 52, 44, false],
        [0, 0, 40, 30, false],
        [58, -58, 22, 12, true],
      ];
      return (
        <g>
          {scenes.map(([u, v, hi, m, dim]) => (
            <g key={u}>
              <path d={square(u, v, 26)} strokeWidth={1} strokeDasharray={dim ? "3 3" : undefined} style={{ ...st, fill: top }} />
              <Cube u={u - 8} v={v + 8} s={6} st={st} fill={fill} top={top} />
              {line([[u + 12, v - 12, Math.max(4, m - 10)], [u + 12, v - 12, hi]], { strokeWidth: 1.3 })}
              {line([[u + 8, v - 12, hi], [u + 16, v - 12, hi]])}
              {line([[u + 8, v - 12, Math.max(4, m - 10)], [u + 16, v - 12, Math.max(4, m - 10)]])}
              {dot([u + 12, v - 12, m])}
            </g>
          ))}
        </g>
      );
    }
    case "improve": {
      // Success per test condition as bars: the weakest one gets new demos and rises (dashed = after).
      const bars: [number, number, number][] = [
        [-56, 56, 58],
        [0, 0, 50],
        [56, -56, 18],
      ];
      const col = (u: number, v: number, h: number, dashed = false) => {
        const b = box(u - 9, u + 9, v - 9, v + 9, h);
        return (
          <g strokeDasharray={dashed ? "3 3" : undefined}>
            <path d={b.faceV} strokeWidth={1} style={{ ...st, fill: dashed ? "none" : fill }} />
            <path d={b.faceU} strokeWidth={1} style={{ ...st, fill: dashed ? "none" : fill }} />
            <path d={b.top} strokeWidth={1} style={{ ...st, fill: dashed ? "none" : top }} />
          </g>
        );
      };
      const [wu, wv] = [56, -56];
      return (
        <g>
          {line([[-84, 84, 0], [84, -84, 0]], { strokeOpacity: 0.4 })}
          {col(wu, wv, 54, true)}
          {bars.map(([u, v, h]) => (
            <g key={u}>{col(u, v, h)}</g>
          ))}
          {line([[wu + 22, wv + 22, 22], [wu + 22, wv + 22, 50]], { strokeWidth: 1.3 })}
          {line([[wu + 18, wv + 18, 44], [wu + 22, wv + 22, 50], [wu + 26, wv + 26, 44]], { strokeWidth: 1.3 })}
          {[0, 1, 2].map((i) => (
            <path
              key={i}
              d={path([[wu - 30 + i * 6, wv + 30 - i * 6, 0], [wu - 14 + i * 6, wv + 30 - i * 6, 0], [wu - 14 + i * 6, wv + 46 - i * 6, 0], [wu - 30 + i * 6, wv + 46 - i * 6, 0]], true)}
              strokeWidth={1}
              style={{ ...st, fill: top }}
            />
          ))}
        </g>
      );
    }
    case "deploy": {
      // Deployment: a line of physical arms working parts on an assembly conveyor.
      const belt = (u0: number, u1: number, v0: number, v1: number, z0: number, z1: number) => ({
        top: path([[u0, v0, z1], [u1, v0, z1], [u1, v1, z1], [u0, v1, z1]], true),
        faceV: path([[u0, v1, z0], [u1, v1, z0], [u1, v1, z1], [u0, v1, z1]], true),
        faceU: path([[u1, v0, z0], [u1, v1, z0], [u1, v1, z1], [u1, v0, z1]], true),
      });
      const b = belt(-96, 96, 14, 34, 0, 8);
      const parts = [-74, -24, 26, 76];
      return (
        <g>
          <path d={b.faceV} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.faceU} strokeWidth={1} style={{ ...st, fill }} />
          <path d={b.top} strokeWidth={1} style={{ ...st, fill: top }} />
          {[-80, -60, -40, -20, 0, 20, 40, 60, 80].map((u) => (
            <g key={u}>{line([[u, 14, 8], [u, 34, 8]], { strokeOpacity: 0.45 })}</g>
          ))}
          {parts.map((u) => {
            const p = belt(u - 6, u + 6, 18, 30, 8, 18);
            return (
              <g key={u}>
                <path d={p.faceV} strokeWidth={1} style={{ ...st, fill }} />
                <path d={p.faceU} strokeWidth={1} style={{ ...st, fill }} />
                <path d={p.top} strokeWidth={1} style={{ ...st, fill: top }} />
              </g>
            );
          })}
          {[-50, 0, 50].map((u) => (
            <g key={u}>
              <Arm u={u} v={-36} to={[u + 2, 22, 20]} st={st} fill={fill} top={top} scale={0.75} />
            </g>
          ))}
          {line([[-96, 50, 0], [-80, 50, 0]], { strokeWidth: 1.2 })}
          {line([[-86, 46, 0], [-80, 50, 0], [-86, 54, 0]], { strokeWidth: 1.2 })}
        </g>
      );
    }
  }
}
