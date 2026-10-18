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
      // The agent's terminal (a standing panel) describing the layout, with leaders to the objects and the zone.
      const v = -62;
      const rows = [48, 36, 24, 12];
      const lens = [70, 104, 58, 86];
      return (
        <g>
          <path d={path([[-84, v, 0], [60, v, 0], [60, v, 72], [-84, v, 72]], true)} strokeWidth={1} style={{ ...st, fill: top }} />
          {line([[-84, v, 60], [60, v, 60]], { strokeOpacity: 0.5 })}
          {[-78, -72, -66].map((u) => (
            <g key={u}>{dot([u, v, 66], 1.6)}</g>
          ))}
          {rows.map((z, i) => (
            <g key={z}>{line([[-74, v, z], [-74 + lens[i], v, z]], { strokeWidth: i === 0 ? 1.3 : 1 })}</g>
          ))}
          {line([[10, v, 24], [-18, 30, 0]], { strokeDasharray: "2 3", strokeOpacity: 0.7 })}
          {line([[40, v, 12], [52, 40, 0]], { strokeDasharray: "2 3", strokeOpacity: 0.7 })}
          <Zone u={52} v={44} w={20} st={st} />
          <Cyl u={-18} v={34} r={11} h={34} st={st} fill={fill} top={top} />
          <Cube u={-58} v={54} s={9} st={st} fill={fill} top={top} />
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
      // One demonstration: home, pick, place, home, traced in the air above the scene.
      const home: P3 = [-24, 24, 44];
      const pick: P3 = [8, 10, 34];
      const place: P3 = [46, -38, 22];
      const trace = [...arc(home, pick, 8), ...arc(pick, place, 14).slice(1), ...arc(place, home, 10).slice(1)];
      return (
        <g>
          <Zone u={46} v={-38} w={18} st={st} />
          <Cyl u={8} v={10} r={10} h={30} st={st} fill={fill} top={top} />
          <Arm u={-50} v={50} to={[home[0], home[1], home[2] - 12]} st={st} fill={fill} top={top} />
          <path d={path(trace)} fill="none" strokeWidth={1.2} strokeDasharray="4 3" style={st} />
          {dot(pick, 3.2, true)}
          {dot(place, 3.2, true)}
          {dot(home, 3.2)}
        </g>
      );
    }
    case "train": {
      // Demonstration frames feed the policy, and the training loss falls.
      const loss: P3[] = [];
      for (let u = -20; u <= 84; u += 6) loss.push([u, 30, 8 + 52 * Math.exp(-(u + 20) / 26) + 2.5 * Math.sin(u / 4)]);
      return (
        <g>
          {[-78, -64, -50].map((u, i) => (
            <path
              key={u}
              d={path([[u, -10 + i * 14, 0], [u + 34, -10 + i * 14, 0], [u + 34, 24 + i * 14, 0], [u, 24 + i * 14, 0]], true)}
              strokeWidth={1}
              style={{ ...st, fill: i === 2 ? fill : top }}
            />
          ))}
          {line([[-28, 30, 0], [-16, 30, 0]], { strokeWidth: 1.2 })}
          {line([[-20, 26, 0], [-16, 30, 0], [-20, 34, 0]], { strokeWidth: 1.2 })}
          <path d={path([[-12, 30, 0], [84, 30, 0], [84, 30, 72], [-12, 30, 72]], true)} strokeWidth={1} style={{ ...st, fill: top, strokeOpacity: 0.5 }} />
          {[18, 36, 54].map((z) => (
            <g key={z}>{line([[-12, 30, z], [84, 30, z]], { strokeOpacity: 0.3, strokeDasharray: "2 4" })}</g>
          ))}
          <path d={path(loss.filter((p) => p[0] >= -12))} fill="none" strokeWidth={1.4} style={st} />
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
      // The weakest scene gets new demonstrations (extra paths), then the loop returns to training.
      const [wu, wv] = [40, -40];
      const back = ring(-6, 6, 70, Math.PI * 0.62, Math.PI * 1.32, 40);
      const e = back[back.length - 1];
      return (
        <g>
          <path d={square(-50, 50, 24)} strokeWidth={1} style={{ ...st, fill: top, strokeOpacity: 0.6 }} />
          <path d={square(wu, wv, 30)} strokeWidth={1.2} style={{ ...st, fill }} />
          {[16, 30, 44].map((lift, i) => (
            <path
              key={lift}
              d={path(arc([wu - 30 + i * 4, wv + 34, 6], [wu + 22 + i * 4, wv - 26, 6], lift))}
              fill="none"
              strokeWidth={1.1}
              strokeDasharray="4 3"
              style={st}
            />
          ))}
          <path d={path(back)} fill="none" strokeWidth={1} style={st} />
          {line([[e[0] - 10, e[1] + 2, 0], e, [e[0] - 2, e[1] + 10, 0]])}
          {dot([wu, wv, 0], 3, true)}
        </g>
      );
    }
    case "deploy": {
      // The physical arm runs the policy (a chip plugged in), with an e-stop in reach.
      const chip = box(34, 70, -70, -34, 10);
      return (
        <g>
          <path d={square(0, 0, 70)} fill="none" strokeWidth={1} strokeOpacity={0.45} style={st} />
          <Arm u={-46} v={46} to={[10, -6, 18]} st={st} fill={fill} top={top} scale={1.15} />
          <path d={chip.faceV} strokeWidth={1} style={{ ...st, fill }} />
          <path d={chip.faceU} strokeWidth={1} style={{ ...st, fill }} />
          <path d={chip.top} strokeWidth={1} style={{ ...st, fill: top }} />
          {[40, 52, 64].map((u) => (
            <g key={u}>{line([[u, -34, 0], [u, -26, 0]])}</g>
          ))}
          {line([[52, -26, 0], [-34, 36, 0]], { strokeDasharray: "2 3" })}
          <Cyl u={48} v={44} r={12} h={8} st={st} fill={fill} top={top} />
          <path d={path(ring(48, 44, 6, 0, Math.PI * 2, 24, 8), true)} strokeWidth={1} style={{ ...st, fill: stroke }} />
        </g>
      );
    }
  }
}
