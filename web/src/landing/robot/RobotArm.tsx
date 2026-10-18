import { Fragment, useMemo, useRef, type ReactNode } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { ROBOT } from "../../robotics/robotConfig";
import { interactionPlane, intersectRayPlane } from "../../robotics/cursorMapping";
import { CALLOUTS } from "../hero/overlaySpec";
import { useRuntime, type RobotRuntime } from "./runtime";
import {
  BaseVisual,
  ElbowVisual,
  ForearmVisual,
  GripperVisual,
  TurretVisual,
  UpperArmVisual,
  WristVisual,
} from "./RobotGeometry";

const LINK_VISUALS = [TurretVisual, UpperArmVisual, ElbowVisual, ForearmVisual, WristVisual] as const;

/** Distance (m) the interaction plane sits in front of the base, toward the viewer. */
const PLANE_OFFSET = 0.25;

interface Props {
  showAxes?: boolean;
  /** Exposes the TCP frame for the physics cube. */
  tcpRef: React.RefObject<THREE.Object3D>;
}

export function RobotArm({ showAxes = false, tcpRef }: Props) {
  const rt = useRuntime();
  const camera = useThree((s) => s.camera);
  const jointRefs = useRef<(THREE.Group | null)[]>([]);
  const visualRefs = useRef<(THREE.Group | null)[]>([]);
  const baseRef = useRef<THREE.Group>(null);
  const fingerRefs = useRef<(THREE.Group | null)[]>([]);
  const axes = useMemo(() => ROBOT.joints.map((j) => new THREE.Vector3(...j.axis)), []);

  // Scratch objects reused every frame (no per-frame allocation).
  const scratch = useMemo(
    () => ({
      raycaster: new THREE.Raycaster(),
      ndc: new THREE.Vector2(),
      hit: new Float64Array(3),
      fwd: new THREE.Vector3(),
      v: new THREE.Vector3(),
      fwdArr: new Float64Array(3),
      origin: new Float64Array(3),
      dir: new Float64Array(3),
      plane: { point: new Float64Array(3), normal: new Float64Array(3) },
    }),
    [],
  );

  useFrame((state, dt) => {
    const c = rt.controller;

    // Pointer -> plane -> workspace-clamped target (only while inside the zone).
    if (rt.pointer.inside) {
      scratch.ndc.set(rt.pointer.ndcX, rt.pointer.ndcY);
      scratch.raycaster.setFromCamera(scratch.ndc, camera);
      camera.getWorldDirection(scratch.fwd);
      scratch.fwd.toArray(scratch.fwdArr);
      const plane = interactionPlane(scratch.fwdArr, ROBOT.baseOrigin, PLANE_OFFSET, scratch.plane);
      const r = scratch.raycaster.ray;
      r.origin.toArray(scratch.origin);
      r.direction.toArray(scratch.dir);
      if (intersectRayPlane(scratch.origin, scratch.dir, plane.point, plane.normal, scratch.hit)) {
        c.setPointerTarget(scratch.hit);
      }
      if (c.state === "IDLE" || c.state === "RETURNING") c.send("POINTER_ENTER");
    } else if (c.state === "TRACKING") {
      c.send("POINTER_LEAVE");
    }

    c.update(dt);

    // Apply joint angles to the kinematic hierarchy (same chain the FK uses).
    for (let i = 0; i < ROBOT.joints.length; i++) {
      const g = jointRefs.current[i];
      if (g) g.quaternion.setFromAxisAngle(axes[i], c.q[i]);
    }

    // Fingers slide symmetrically; pads meet the commanded opening exactly.
    const half = c.gripper / 2 + 0.013;
    const [fa, fb] = fingerRefs.current;
    if (fa) fa.position.y = half;
    if (fb) fb.position.y = -half;

    applyReveal(rt, baseRef.current, visualRefs.current);
    updateOverlays(rt, state.camera, state.size, scratch.v);

    rt.stats.frames++;
    rt.stats.acc += dt;
    if (rt.stats.acc > 0.5) {
      rt.stats.fps = rt.stats.frames / rt.stats.acc;
      rt.stats.frames = 0;
      rt.stats.acc = 0;
    }
  }, -2);

  // Build the nested joint groups from the config so geometry and kinematics share one chain.
  const chain = ROBOT.joints.reduceRight<ReactNode>((child, j, i) => {
    const Visual = LINK_VISUALS[i];
    return (
      <group
        key={j.name}
        name={j.name}
        ref={(el) => {
          jointRefs.current[i] = el;
        }}
        position={j.offset}
      >
        <group
          ref={(el) => {
            visualRefs.current[i] = el;
          }}
        >
          {Visual ? <Visual /> : <GripperVisual fingerRefs={fingerRefs} />}
        </group>
        {showAxes && <axesHelper args={[0.18]} />}
        {child}
        {i === ROBOT.joints.length - 1 && <object3D ref={tcpRef} position={ROBOT.toolOffset} />}
      </group>
    );
  }, null);

  return (
    <group position={ROBOT.baseOrigin}>
      <group ref={baseRef}>
        <BaseVisual />
      </group>
      <Fragment>{chain}</Fragment>
    </group>
  );
}

/** Intro: parts rise into place in sequence. Visual-only offsets; kinematics untouched. */
function applyReveal(rt: RobotRuntime, base: THREE.Group | null, visuals: (THREE.Group | null)[]) {
  const ease = (t: number) => 1 - Math.pow(1 - t, 3);
  if (base) {
    const t = ease(rt.reveal.base);
    base.visible = t > 0.001;
    base.position.y = (1 - t) * -0.06;
  }
  for (let i = 0; i < visuals.length; i++) {
    const v = visuals[i];
    if (!v) continue;
    const t = ease(rt.reveal.links[i] ?? 1);
    v.visible = t > 0.001;
    v.position.y = (1 - t) * 0.08;
  }
}

/** Project joints to screen and update the DOM callouts (positions every frame, text ~8 Hz). */
function updateOverlays(
  rt: RobotRuntime,
  camera: THREE.Camera,
  size: { width: number; height: number },
  v: THREE.Vector3,
) {
  const c = rt.controller;
  const o = rt.overlay;
  const now = performance.now();
  const textDue = now - o.lastTextUpdate > 120;
  if (textDue) o.lastTextUpdate = now;
  const deg = (r: number) => `${(r * 180) / Math.PI >= 0 ? "+" : "−"}${Math.abs((r * 180) / Math.PI).toFixed(1)}°`;

  for (const spec of CALLOUTS) {
    const refs = o.callouts[spec.id];
    if (!refs?.path) continue;
    if (spec.joint >= 0) {
      const p = c.fk.positions[spec.joint];
      v.set(p[0], p[1], p[2]);
      if (spec.joint === 0) v.y -= 0.07;
    } else {
      const p = c.fk.tcpPosition;
      v.set(p[0], p[1], p[2]);
    }
    v.project(camera);
    const x = (v.x * 0.5 + 0.5) * size.width;
    const y = (-v.y * 0.5 + 0.5) * size.height;
    // Mirror the leader if the label would leave the canvas.
    let dx = spec.dx;
    let run = spec.run;
    const w = refs.width ?? 160;
    if (run >= 0 && x + dx + run + 6 + w > size.width - 12) {
      dx = -dx;
      run = -run;
    } else if (run < 0 && x + dx + run - 6 - w < 12) {
      dx = -dx;
      run = -run;
    }
    const ex = x + dx;
    const ey = y + spec.dy;
    refs.path.setAttribute("d", `M${x.toFixed(1)},${y.toFixed(1)} L${ex.toFixed(1)},${ey.toFixed(1)} L${(ex + run).toFixed(1)},${ey.toFixed(1)}`);
    refs.node?.setAttribute("cx", x.toFixed(1));
    refs.node?.setAttribute("cy", y.toFixed(1));
    if (refs.label) {
      const lx = ex + run + (run >= 0 ? 6 : -6);
      refs.label.style.transform = `translate(${lx.toFixed(1)}px, ${ey.toFixed(1)}px) translate(${run >= 0 ? "0" : "-100%"}, -50%)`;
    }
    if (textDue && refs.value) {
      if (spec.joint >= 0) refs.value.textContent = deg(c.q[spec.joint]);
      else {
        const t = c.fk.tcpPosition;
        refs.value.textContent = `${t[0].toFixed(3)}  ${t[1].toFixed(3)}  ${t[2].toFixed(3)} m`;
      }
      if (refs.label) refs.width = refs.label.offsetWidth;
    }
  }

  if (!textDue) return;
  if (o.status) {
    const label: Record<string, string> = {
      INITIALIZING: "INITIALIZING",
      INTRO: "HOMING",
      IDLE: "IDLE · HOLD",
      TRACKING: c.targetClamped ? "TRACKING · LIMIT" : "TRACKING · LIVE",
      RETURNING: "RETURNING · REST",
      GRASPING: `PICK · ${rt.cube.status}`,
      RELEASING: `PLACE · ${rt.cube.status}`,
      ERROR: "FAULT · RECOVERING",
    };
    o.status.textContent = label[c.state] ?? c.state;
  }
  if (o.ik) {
    if (c.state === "RETURNING" || c.state === "INTRO" || c.state === "INITIALIZING") {
      o.ik.textContent = "JOINT-SPACE MOVE";
    } else {
      const mm = (c.ik.positionError * 1000).toFixed(1);
      o.ik.textContent = c.ik.converged ? `IK converged · residual ${mm} mm` : `IK nearest feasible · residual ${mm} mm`;
    }
  }
  ROBOT.joints.forEach((j, i) => {
    const row = o.joints[i];
    if (row?.value) row.value.textContent = deg(c.q[i]);
    if (row?.bar) row.bar.style.left = `${(((c.q[i] - j.min) / (j.max - j.min)) * 100).toFixed(1)}%`;
  });
  if (o.graspButton) o.graspButton.disabled = !(c.state === "IDLE" || c.state === "RETURNING" || c.state === "TRACKING");
}
