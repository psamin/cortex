import { createContext, Suspense, useContext, useLayoutEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { ContactShadows, Edges, Grid, Line, Outlines } from "@react-three/drei";
import { STLLoader } from "three-stdlib";
import * as THREE from "three";
import { INK, PALETTE, robotMaterials, type MaterialKey } from "../landing/robot/RobotMaterials";
import { meshUrl, type QuatWXYZ, type SimModel, type SimPrimitive, type SimState, type Vec3 } from "./api";

/**
 * Stylized, render-only view of MuJoCo state in the landing page's drawing language: toon fills,
 * crease edges and silhouettes on white, an orthographic three-quarter camera, a floor grid with
 * construction lines, and mono callouts. Poses come from the backend; nothing is simulated here.
 *
 * MuJoCo is z-up; the scene root rotates it into three.js's y-up, so (x, y, z) -> (x, z, -y).
 */

const ELEVATION = THREE.MathUtils.degToRad(30);
const AZIMUTH = THREE.MathUtils.degToRad(45);
/** Workspace the camera frames (MuJoCo frame, metres): base at the origin, table out to ~0.4 m. */
const LOOK_AT_MJ: Vec3 = [0.16, 0.0, 0.1];
const FRAME_W = 0.78;
const FRAME_H = 0.6;
/** Silhouette width in screen pixels (converted to world units for the current zoom). */
const OUTLINE_PX = 1.15;

const toThree = (p: Vec3 | number[], out = new THREE.Vector3()) => out.set(p[0], p[2], -p[1]);
const quat = (q: QuatWXYZ) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);

const OutlineContext = createContext(0.0012);

// -------------------------------------------------------------------- camera
function SimCamera() {
  const camera = useThree((s) => s.camera) as THREE.OrthographicCamera;
  const size = useThree((s) => s.size);
  useLayoutEffect(() => {
    const ppm = Math.min(size.width / FRAME_W, size.height / FRAME_H);
    const W = size.width / ppm;
    const H = size.height / ppm;
    (camera as THREE.OrthographicCamera & { manual?: boolean }).manual = true;
    camera.left = -W / 2;
    camera.right = W / 2;
    camera.top = H * 0.52;
    camera.bottom = -H * 0.48;
    camera.zoom = 1;
    camera.near = 0.01;
    camera.far = 40;
    const look = toThree(LOOK_AT_MJ);
    const d = 10;
    camera.position.set(
      look.x + d * Math.cos(ELEVATION) * Math.sin(AZIMUTH),
      look.y + d * Math.sin(ELEVATION),
      look.z + d * Math.cos(ELEVATION) * Math.cos(AZIMUTH),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(look);
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height]);
  return null;
}

function useOutlineWorld() {
  const size = useThree((s) => s.size);
  return OUTLINE_PX / Math.min(size.width / FRAME_W, size.height / FRAME_H);
}

// -------------------------------------------------------------------- parts
function Part({
  geometry,
  material,
  edges = true,
  outline = true,
  threshold = 30,
  position,
  quaternion,
}: {
  geometry: THREE.BufferGeometry;
  material: MaterialKey;
  edges?: boolean;
  outline?: boolean;
  threshold?: number;
  position?: THREE.Vector3Tuple;
  quaternion?: THREE.Quaternion;
}) {
  const thickness = useContext(OutlineContext);
  const mats = robotMaterials();
  const dark = material === "dark" || material === "black";
  return (
    <mesh geometry={geometry} material={mats[material]} position={position} quaternion={quaternion}>
      {edges && !dark && <Edges threshold={threshold} color={INK} lineWidth={0.8} />}
      {outline && <Outlines thickness={thickness} color={INK} angle={Math.PI} />}
    </mesh>
  );
}

function StlPart({ name, role, pos, q, scale }: { name: string; role: "arm" | "motor"; pos: Vec3; q: QuatWXYZ; scale: Vec3 }) {
  const geometry = useLoader(STLLoader, meshUrl(name));
  return (
    <group position={pos} quaternion={quat(q)} scale={scale}>
      <Part geometry={geometry} material={role === "motor" ? "dark" : "light"} threshold={35} />
    </group>
  );
}

const primitiveGeometry = (p: SimPrimitive): THREE.BufferGeometry | null => {
  const [a, b, c] = p.size;
  switch (p.type) {
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "sphere":
      return new THREE.SphereGeometry(a, 32, 16);
    case "cylinder":
      return new THREE.CylinderGeometry(a, a, 2 * b, 40).rotateX(Math.PI / 2); // MuJoCo cylinders run along z
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 8, 24).rotateX(Math.PI / 2);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 32, 16).scale(a, b, c);
    default:
      return null;
  }
};

const MATERIAL_FOR: Partial<Record<SimPrimitive["role"], MaterialKey>> = {
  table: "light",
  object: "mid",
  obstacle: "deep",
  box: "mid",
  motor: "dark",
  prop: "light",
};

function Primitive({ p, pick }: { p: SimPrimitive; pick: boolean }) {
  const geometry = useMemo(() => primitiveGeometry(p), [p]);
  if (p.role === "zone") return <ZoneMarker p={p} />;
  const material = pick ? "deep" : MATERIAL_FOR[p.role];
  if (!geometry || !material) return null;
  return (
    <Part
      geometry={geometry}
      material={material}
      position={p.pos}
      quaternion={quat(p.quat)}
      outline={p.role !== "table"}
      threshold={p.role === "table" ? 80 : 30}
    />
  );
}

/** Place zones: a dashed boundary and a faint fill, like a marked area on a drawing. */
function ZoneMarker({ p }: { p: SimPrimitive }) {
  const [hx, hy, hz] = p.size;
  const z = hz + 0.0008;
  const pts: Vec3[] = [
    [-hx, -hy, z],
    [hx, -hy, z],
    [hx, hy, z],
    [-hx, hy, z],
    [-hx, -hy, z],
  ];
  return (
    <group position={p.pos} quaternion={quat(p.quat)}>
      <Line points={pts} color={INK} lineWidth={1} dashed dashSize={0.008} gapSize={0.005} />
      <mesh position={[0, 0, z - 0.0003]}>
        <planeGeometry args={[2 * hx, 2 * hy]} />
        <meshBasicMaterial color={INK} transparent opacity={0.06} depthWrite={false} />
      </mesh>
    </group>
  );
}

// -------------------------------------------------------------------- decor
function TableDecor({ model }: { model: SimModel }) {
  const table = model.primitives.find((p) => p.role === "table");
  const top = table ? table.pos[2] + table.size[2] : 0;
  const [cx, cy] = table ? [table.pos[0], table.pos[1]] : [0, 0];
  const [sx, sy] = table ? [table.size[0] * 2, table.size[1] * 2] : [0.8, 0.8];
  const r = 0.3;
  const circle = useMemo(() => {
    const pts: Vec3[] = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push([Math.cos(a) * r, Math.sin(a) * r, top + 0.0006]);
    }
    return pts;
  }, [top]);
  const ticks = useMemo(() => {
    const segs: Vec3[] = [];
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const r0 = i % 6 === 0 ? r - 0.024 : r - 0.012;
      segs.push([Math.cos(a) * r0, Math.sin(a) * r0, top + 0.0006], [Math.cos(a) * r, Math.sin(a) * r, top + 0.0006]);
    }
    return segs;
  }, [top]);
  return (
    <group>
      {/* Grid lies in the table plane: drei's Grid is XZ in its own frame, so undo the root rotation. */}
      <group position={[cx, cy, top + 0.0004]} rotation={[Math.PI / 2, 0, 0]}>
        <Grid
          args={[sx, sy]}
          cellSize={0.02}
          cellThickness={0.5}
          cellColor="#EFEFEF"
          sectionSize={0.1}
          sectionThickness={0.8}
          sectionColor="#E0E0E0"
          fadeDistance={6}
          fadeStrength={0}
          followCamera={false}
        />
      </group>
      <Line points={circle} color="#BDBDBD" lineWidth={0.8} dashed dashSize={0.01} gapSize={0.008} />
      <Line points={ticks} segments color="#BDBDBD" lineWidth={0.8} />
      <Line
        points={[
          [-0.08, 0, top + 0.0006],
          [r + 0.08, 0, top + 0.0006],
        ]}
        color="#D4D4D4"
        lineWidth={0.7}
        dashed
        dashSize={0.016}
        gapSize={0.01}
      />
      <Line
        points={[
          [0, -r - 0.06, top + 0.0006],
          [0, r + 0.06, top + 0.0006],
        ]}
        color="#D4D4D4"
        lineWidth={0.7}
        dashed
        dashSize={0.016}
        gapSize={0.01}
      />
    </group>
  );
}

const reticle = (() => {
  const ring: Vec3[] = [];
  const R = 0.012;
  for (let i = 0; i <= 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    ring.push([Math.cos(a) * R, Math.sin(a) * R, 0]);
  }
  const ticks: Vec3[] = [];
  for (const [x, y] of [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ]) {
    ticks.push([x * 0.016, y * 0.016, 0], [x * 0.024, y * 0.024, 0]);
  }
  return { ring, ticks };
})();

/** Camera-facing TCP reticle (in three.js world space, outside the z-up root). */
function TcpReticle({ tcp }: { tcp: Vec3 }) {
  const g = useRef<THREE.Group>(null);
  const pos = useMemo(() => toThree(tcp), [tcp]);
  useFrame(({ camera }) => g.current?.quaternion.copy(camera.quaternion));
  return (
    <group ref={g} position={pos}>
      <Line points={reticle.ring} color={INK} lineWidth={1} depthTest={false} renderOrder={10} />
      <Line points={reticle.ticks} segments color={INK} lineWidth={1} depthTest={false} renderOrder={10} />
    </group>
  );
}

// -------------------------------------------------------------------- callouts
interface Callout {
  id: string;
  label: string;
  anchor: Vec3;
  value: string;
  dx: number;
  dy: number;
  run: number;
}

const deg = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}°`;
const xyz = (p: Vec3) => `${p[0].toFixed(3)}  ${p[1].toFixed(3)}  ${p[2].toFixed(3)} m`;
/** Leader offsets (px) per joint, alternating sides so labels rarely collide. */
const JOINT_LEADERS: Array<[number, number, number]> = [
  [-58, 46, -26],
  [-74, -18, -26],
  [-52, -64, -26],
  [62, -66, 26],
  [78, -26, 26],
  [70, 30, 26],
];

export function buildCallouts(model: SimModel, state: SimState): Callout[] {
  const out: Callout[] = [];
  model.joints.forEach((j, i) => {
    const b = state.bodies[j.body];
    const v = state.joints_deg[j.name];
    if (!b || v === undefined) return;
    const [dx, dy, run] = JOINT_LEADERS[i] ?? [60, 0, 26];
    out.push({ id: j.name, label: j.label, anchor: b.pos, value: deg(v), dx, dy, run });
  });
  out.push({ id: "tcp", label: "TCP", anchor: state.tcp, value: xyz(state.tcp), dx: 40, dy: 74, run: 24 });
  const obj = state.pick_body ? state.bodies[state.pick_body] : undefined;
  if (obj && state.pick) {
    out.push({
      id: "object",
      label: `OBJ / ${state.pick.toUpperCase()}`,
      anchor: obj.pos,
      value: xyz(obj.pos),
      dx: -46,
      dy: 70,
      run: -24,
    });
  }
  for (const z of model.primitives.filter((p) => p.role === "zone")) {
    out.push({
      id: `zone:${z.name}`,
      label: `ZONE / ${(z.name ?? "").replace(/^zone_/, "").toUpperCase()}`,
      anchor: z.pos,
      value: "",
      dx: 36,
      dy: 44,
      run: 22,
    });
  }
  return out;
}

interface CalloutRefs {
  path: SVGPathElement | null;
  node: SVGCircleElement | null;
  label: HTMLDivElement | null;
}

/** Projects anchors every rendered frame and writes positions straight to the DOM (no React state). */
function CalloutProjector({ callouts, refs }: { callouts: Callout[]; refs: React.MutableRefObject<Record<string, CalloutRefs>> }) {
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    for (const c of callouts) {
      const r = refs.current[c.id];
      if (!r?.path) continue;
      toThree(c.anchor, v).project(camera);
      const x = (v.x * 0.5 + 0.5) * size.width;
      const y = (-v.y * 0.5 + 0.5) * size.height;
      let { dx, run } = c;
      const w = r.label?.offsetWidth ?? 140;
      if (run >= 0 && x + dx + run + 6 + w > size.width - 8) {
        dx = -dx;
        run = -run;
      } else if (run < 0 && x + dx + run - 6 - w < 8) {
        dx = -dx;
        run = -run;
      }
      const ex = x + dx;
      const ey = Math.min(Math.max(y + c.dy, 10), size.height - 10);
      r.path.setAttribute("d", `M${x.toFixed(1)},${y.toFixed(1)} L${ex.toFixed(1)},${ey.toFixed(1)} L${(ex + run).toFixed(1)},${ey.toFixed(1)}`);
      r.node?.setAttribute("cx", x.toFixed(1));
      r.node?.setAttribute("cy", y.toFixed(1));
      if (r.label) {
        const lx = ex + run + (run >= 0 ? 5 : -5);
        r.label.style.transform = `translate(${lx.toFixed(1)}px, ${ey.toFixed(1)}px) translate(${run >= 0 ? "0" : "-100%"}, -50%)`;
        r.label.style.opacity = "1";
      }
    }
  });
  return null;
}

function CalloutLayer({ callouts, refs }: { callouts: Callout[]; refs: React.MutableRefObject<Record<string, CalloutRefs>> }) {
  const reg = (id: string) => (refs.current[id] ??= { path: null, node: null, label: null });
  return (
    <div className="pointer-events-none absolute inset-0 select-none" aria-hidden>
      <svg className="absolute inset-0 h-full w-full overflow-visible">
        {callouts.map((c) => (
          <g key={c.id}>
            <path ref={(el) => void (reg(c.id).path = el)} fill="none" stroke={INK} strokeWidth={0.75} />
            <circle ref={(el) => void (reg(c.id).node = el)} r={2.4} fill="#fff" stroke={INK} strokeWidth={0.9} />
          </g>
        ))}
      </svg>
      {callouts.map((c) => (
        <div
          key={c.id}
          ref={(el) => void (reg(c.id).label = el)}
          className="kn-mono absolute left-0 top-0 whitespace-nowrap bg-white/85 px-1 py-0.5 text-[9.5px] leading-none tracking-[0.08em] text-[#171717] opacity-0"
        >
          <span>{c.label}</span>
          {c.value && <span className="ml-2 tabular-nums text-[#858585]">{c.value}</span>}
        </div>
      ))}
    </div>
  );
}

// -------------------------------------------------------------------- scene
function Scene({ model, state }: { model: SimModel; state: SimState }) {
  const outline = useOutlineWorld();
  const meshesByBody = useMemo(() => {
    const m = new Map<string, SimModel["meshes"]>();
    for (const g of model.meshes) m.set(g.body, [...(m.get(g.body) ?? []), g]);
    return m;
  }, [model]);
  const primsByBody = useMemo(() => {
    const m = new Map<string, SimPrimitive[]>();
    const show = new Set(["table", "object", "zone", "obstacle", "box", "motor", "prop"]);
    for (const p of model.primitives) if (show.has(p.role)) m.set(p.body, [...(m.get(p.body) ?? []), p]);
    return m;
  }, [model]);

  const bodies = useMemo(() => Array.from(new Set([...meshesByBody.keys(), ...primsByBody.keys()])), [meshesByBody, primsByBody]);

  return (
    <OutlineContext.Provider value={outline}>
      <SimCamera />
      <ambientLight intensity={1.15} />
      <directionalLight position={[-1.5, 6, 3.5]} intensity={1.9} />
      <directionalLight position={[4, 2, -2]} intensity={0.35} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <TableDecor model={model} />
        {bodies.map((name) => {
          const pose = name === "world" ? null : state.bodies[name];
          if (name !== "world" && !pose) return null;
          // Bodies parked under the table (unused upstream props) stay out of the drawing.
          if (pose && pose.pos[2] < -0.2) return null;
          return (
            <group key={name} position={pose?.pos} quaternion={pose ? quat(pose.quat) : undefined}>
              {(primsByBody.get(name) ?? []).map((p, i) => (
                <Primitive key={`p${i}`} p={p} pick={name === state.pick_body} />
              ))}
              <Suspense fallback={null}>
                {(meshesByBody.get(name) ?? []).map((g, i) => (
                  <StlPart key={`m${i}`} name={g.mesh} role={g.role} pos={g.pos} q={g.quat} scale={g.scale} />
                ))}
              </Suspense>
            </group>
          );
        })}
      </group>
      <TcpReticle tcp={state.tcp} />
      <ContactShadows position={[0, 0.0003, 0]} scale={1.2} resolution={512} blur={2.4} far={0.35} opacity={0.18} color="#000000" />
    </OutlineContext.Provider>
  );
}

export interface SimViewportProps {
  model: SimModel;
  state: SimState;
  /** Small mono caption under the status block (e.g. "LIVE · CAMERA FEED ONLY"). */
  note?: string;
  className?: string;
}

/** The stylized MuJoCo viewport. Lazy-load it (it pulls in three.js). */
export default function SimViewport({ model, state, note, className }: SimViewportProps) {
  const callouts = useMemo(() => buildCallouts(model, state), [model, state]);
  const refs = useRef<Record<string, CalloutRefs>>({});
  return (
    <div className={`relative h-full w-full overflow-hidden bg-white ${className ?? ""}`} data-testid="sim-viewport">
      <Canvas
        orthographic
        flat
        frameloop="demand"
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: false }}
        onCreated={({ gl }) => gl.setClearColor(PALETTE.paper, 1)}
        aria-hidden
      >
        <Scene model={model} state={state} />
        <CalloutProjector callouts={callouts} refs={refs} />
      </Canvas>
      <CalloutLayer callouts={callouts} refs={refs} />
      <div className="kn-mono pointer-events-none absolute left-3 top-3 text-[9.5px] uppercase leading-[1.8] tracking-[0.12em] text-[#858585]">
        <div className="flex items-center gap-2 text-[#171717]">
          <span className="h-px w-6 bg-[#171717]" />
          <span>FIG.03</span>
          <span className="text-[#858585]">SO-101 · MuJoCo state</span>
        </div>
        <div className="pl-8 tabular-nums">
          Scene {state.scene} · seed {state.seed} · {state.source}
        </div>
        {note && <div className="pl-8 text-[#171717]">{note}</div>}
      </div>
    </div>
  );
}
