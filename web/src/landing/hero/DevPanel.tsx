import { useEffect, useState, useSyncExternalStore } from "react";
import { BRAND } from "../brand";
import { ROBOT } from "../../robotics/robotConfig";
import { debugStore, type DebugFlags, type RobotRuntime } from "../robot/runtime";

/** Development-only diagnostics. Tree-shaken from production builds. Toggle with the backtick key. */
export default function DevPanel({ runtime }: { runtime: RobotRuntime }) {
  const [open, setOpen] = useState(false);
  const [, tick] = useState(0);
  const flags = useSyncExternalStore(debugStore.subscribe, debugStore.get, debugStore.get);
  const c = runtime.controller;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "`" && setOpen((o) => !o);
    window.addEventListener("keydown", onKey);
    const id = window.setInterval(() => tick((n) => n + 1), 200);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.clearInterval(id);
    };
  }, []);

  const f3 = (a: ArrayLike<number>) => Array.from(a, (v) => v.toFixed(3)).join(", ");
  if (!open)
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-3 left-3 z-50 rounded border border-[#e6e6e6] bg-white px-2 py-1 kn-mono text-[10px] text-[#858585]"
      >
        DEV `
      </button>
    );

  return (
    <div className="fixed bottom-3 left-3 z-50 max-h-[80vh] w-[300px] overflow-auto rounded border border-[#171717] bg-white p-3 kn-mono text-[10px] leading-relaxed text-[#171717] shadow-sm">
      <div className="mb-2 flex justify-between">
        <b>{BRAND.name.toUpperCase()} · DEV</b>
        <button onClick={() => setOpen(false)}>close ×</button>
      </div>
      <div>state: {c.state}</div>
      <div>fps: {runtime.stats.fps.toFixed(0)}</div>
      <div>tcp: {f3(c.fk.tcpPosition)}</div>
      <div>target: {f3(c.target)}</div>
      <div>
        ik: {c.ik.converged ? "converged" : "not converged"} · err {(c.ik.positionError * 1000).toFixed(2)} mm · dir{" "}
        {c.ik.directionError.toFixed(3)} rad · it {c.ik.iterations}
      </div>
      <div>clamped: {String(c.targetClamped)} · gripper {(c.gripper * 1000).toFixed(0)} mm · cube {runtime.cube.status}</div>
      <hr className="my-2 border-[#e6e6e6]" />
      {ROBOT.joints.map((j, i) => (
        <label key={j.name} className="flex items-center gap-2">
          <span className="w-16">{j.label.split(" / ")[0]}</span>
          <input
            type="range"
            min={j.min}
            max={j.max}
            step={0.001}
            value={(c.manual ?? c.q)[i]}
            onChange={(e) => {
              c.manual ??= Float64Array.from(c.q);
              c.manual[i] = Number(e.target.value);
              tick((n) => n + 1);
            }}
            className="flex-1"
          />
          <span className="w-24 text-right tabular-nums">
            {c.q[i].toFixed(2)} [{j.min.toFixed(1)},{j.max.toFixed(1)}]
          </span>
        </label>
      ))}
      <button className="mt-1 underline" onClick={() => (c.manual = null)}>
        release manual override
      </button>
      <hr className="my-2 border-[#e6e6e6]" />
      {(Object.keys(flags) as (keyof DebugFlags)[]).map((k) => (
        <label key={k} className="mr-3 inline-flex items-center gap-1">
          <input type="checkbox" checked={flags[k]} onChange={(e) => debugStore.set({ [k]: e.target.checked })} />
          {k}
        </label>
      ))}
    </div>
  );
}
