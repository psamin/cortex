import { Box } from "./IsoIcon";
import { circle, iso, line, poly, type Pt } from "./iso";

/**
 * The empty-state illustration: a small arm and a cube on a plate, a dashed reach orbit,
 * and one callout. Purely decorative (aria-hidden); pair it with a real headline.
 */
export function SketchScene({
  label = "OBJ.01 / CUBE",
  value = "AWAITING DATA",
  width = 280,
  className,
}: {
  label?: string;
  value?: string;
  width?: number;
  className?: string;
}) {
  const s = 34;
  const o: Pt = [0, 0];
  const anchor = iso([1.26, 0.02, 0.36], s, o);
  const elbow: Pt = [anchor[0] + 30, anchor[1] + 26];
  const end: Pt = [elbow[0] + 26, elbow[1]];
  // Keep the callout text at ~10 px on screen whatever the drawing's width.
  const k = 300 / width;
  const fs = 10 * k;
  return (
    <svg
      aria-hidden
      viewBox="-92 -72 300 140"
      width={width}
      height={(width * 140) / 300}
      className={["sk-svg sk-scene", className].filter(Boolean).join(" ")}
    >
      <Box at={[-1.4, -1.4, 0]} size={[2.8, 2.8, 0.14]} s={s} o={o} />
      <path d={line(circle([0, 0, 0.14], 1.18, "z", 64), s, o)} className="sk-construction" />
      <path d={line([[-1.3, 0, 0.14], [1.3, 0, 0.14]], s, o)} className="sk-construction sk-faint" />
      <path d={line([[0, -1.3, 0.14], [0, 1.3, 0.14]], s, o)} className="sk-construction sk-faint" />
      <Box at={[-0.36, -0.36, 0.14]} size={[0.72, 0.72, 0.14]} s={s} o={o} />
      <Box at={[-0.12, -0.12, 0.28]} size={[0.24, 0.24, 1.05]} s={s} o={o} />
      <Box at={[0.95, -0.12, 0.86]} size={[0.24, 0.24, 0.5]} s={s} o={o} />
      <path d={line([[1.0, 0.0, 0.86], [1.0, 0.0, 0.66]], s, o)} fill="none" />
      <path d={line([[1.14, 0.0, 0.86], [1.14, 0.0, 0.66]], s, o)} fill="none" />
      <Box at={[-0.13, -0.13, 1.33]} size={[1.32, 0.26, 0.22]} s={s} o={o} accent />
      <polygon points={poly([[0.62, 0.62, 0.14], [1.08, 0.62, 0.14], [1.08, 1.08, 0.14], [0.62, 1.08, 0.14]], s, o)} className="sk-construction" fill="none" />
      <path d={line([[0.85, 0.85, 0.14], [1.06, 0.0, 0.14]], s, o)} className="sk-construction" />
      <Box at={[0.86, -0.2, 0.14]} size={[0.4, 0.4, 0.4]} s={s} o={o} />
      <path d={`M${anchor[0]} ${anchor[1]} L${elbow[0]} ${elbow[1]} L${end[0]} ${end[1]}`} fill="none" className="sk-leader" />
      <circle cx={anchor[0]} cy={anchor[1]} r={2.6} className="sk-node" />
      <text x={end[0] + 5} y={end[1] + fs * 0.35} className="sk-svg-label" style={{ fontSize: fs }}>
        {label}
      </text>
      <text x={end[0] + 5} y={end[1] + fs * 1.75} className="sk-svg-label sk-svg-label-2" style={{ fontSize: fs }}>
        {value}
      </text>
    </svg>
  );
}
