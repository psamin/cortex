import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { ContactShadows, Grid, Line } from "@react-three/drei";
import * as THREE from "three";
import { INK, PALETTE } from "../robot/RobotMaterials";
import { useRuntime } from "../robot/runtime";
import { maxReach } from "../../robotics/robotConfig";

export function SceneLighting() {
  return (
    <>
      <ambientLight intensity={1.15} />
      {/* Key light from above and slightly camera-left: tops read white, sides step down. */}
      <directionalLight position={[-1.5, 6, 3.5]} intensity={1.9} />
      <directionalLight position={[4, 2, -2]} intensity={0.35} />
    </>
  );
}

export function EngineeringGrid({ visible = true }: { visible?: boolean }) {
  return (
    <group visible={visible}>
      <Grid
        position={[0, -0.001, 0]}
        args={[12, 12]}
        cellSize={0.1}
        cellThickness={0.55}
        cellColor="#EFEFEF"
        sectionSize={0.5}
        sectionThickness={0.8}
        sectionColor="#E2E2E2"
        fadeDistance={4.2}
        fadeStrength={2.2}
        fadeFrom={0}
        infiniteGrid
        followCamera={false}
      />
    </group>
  );
}

export function GroundShadow() {
  return (
    <ContactShadows
      position={[0, 0.0005, 0]}
      scale={3.2}
      resolution={512}
      blur={2.6}
      far={1.1}
      opacity={0.22}
      color="#000000"
      frames={Infinity}
    />
  );
}

/** Dashed reach circle and yaw ticks on the floor: construction lines of the drawing. */
export function FloorConstruction() {
  const r = maxReach() * 0.62;
  const circle = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * r, 0.002, Math.sin(a) * r));
    }
    return pts;
  }, [r]);
  const ticks = useMemo(() => {
    const segs: THREE.Vector3[] = [];
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const r0 = i % 6 === 0 ? r - 0.06 : r - 0.03;
      segs.push(new THREE.Vector3(Math.cos(a) * r0, 0.002, Math.sin(a) * r0));
      segs.push(new THREE.Vector3(Math.cos(a) * r, 0.002, Math.sin(a) * r));
    }
    return segs;
  }, [r]);
  return (
    <group>
      <Line points={circle} color="#BDBDBD" lineWidth={0.8} dashed dashSize={0.025} gapSize={0.02} />
      <Line points={ticks} segments color="#BDBDBD" lineWidth={0.8} />
      <Line
        points={[
          [-r - 0.15, 0.002, 0],
          [r + 0.15, 0.002, 0],
        ]}
        color="#D4D4D4"
        lineWidth={0.7}
        dashed
        dashSize={0.04}
        gapSize={0.025}
      />
      <Line
        points={[
          [0, 0.002, -r - 0.15],
          [0, 0.002, r + 0.15],
        ]}
        color="#D4D4D4"
        lineWidth={0.7}
        dashed
        dashSize={0.04}
        gapSize={0.025}
      />
    </group>
  );
}

const reticlePoints = (() => {
  const ring: THREE.Vector3[] = [];
  const R = 0.04;
  for (let i = 0; i <= 64; i++) {
    const a = (i / 64) * Math.PI * 2;
    ring.push(new THREE.Vector3(Math.cos(a) * R, Math.sin(a) * R, 0));
  }
  const ticks: THREE.Vector3[] = [];
  const t0 = 0.052;
  const t1 = 0.075;
  for (const [x, y] of [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ]) {
    ticks.push(new THREE.Vector3(x * t0, y * t0, 0), new THREE.Vector3(x * t1, y * t1, 0));
  }
  return { ring, ticks };
})();

/** Camera-facing target reticle + dashed trajectory from the TCP to the target. */
export function TargetMarkers() {
  const rt = useRuntime();
  const group = useRef<THREE.Group>(null);
  const traj = useRef<THREE.Object3D & { geometry: { setPositions: (a: ArrayLike<number>) => void }; computeLineDistances: () => void }>(null);
  const positions = useMemo(() => new Float32Array(6), []);
  const fade = useRef(0);

  useFrame(({ camera }, dt) => {
    const c = rt.controller;
    const active = c.state === "TRACKING";
    fade.current += ((active ? 1 : 0) - fade.current) * (1 - Math.exp(-dt / 0.12));
    const g = group.current;
    if (g) {
      const src = c.state === "TRACKING" ? c.rawTarget : c.target;
      g.position.set(src[0], src[1], src[2]);
      g.quaternion.copy(camera.quaternion);
      g.scale.setScalar(0.6 + 0.4 * fade.current);
      g.visible = fade.current > 0.02;
    }
    const l = traj.current;
    if (l) {
      const t = c.fk.tcpPosition;
      const src = c.state === "TRACKING" ? c.rawTarget : c.target;
      positions[0] = t[0];
      positions[1] = t[1];
      positions[2] = t[2];
      positions[3] = src[0];
      positions[4] = src[1];
      positions[5] = src[2];
      l.geometry.setPositions(positions);
      l.computeLineDistances();
      l.visible = fade.current > 0.02 && Math.hypot(t[0] - src[0], t[1] - src[1], t[2] - src[2]) > 0.02;
    }
  });

  return (
    <>
      <group ref={group}>
        <Line points={reticlePoints.ring} color={INK} lineWidth={1.1} depthTest={false} renderOrder={10} />
        <Line points={reticlePoints.ticks} segments color={INK} lineWidth={1.1} depthTest={false} renderOrder={10} />
        <mesh renderOrder={10}>
          <circleGeometry args={[0.007, 20]} />
          <meshBasicMaterial color={INK} depthTest={false} />
        </mesh>
      </group>
      <Line
        ref={traj as never}
        points={[
          [0, 0, 0],
          [0, 0, 0],
        ]}
        color={PALETTE.label}
        lineWidth={1}
        dashed
        dashSize={0.018}
        gapSize={0.014}
        depthTest={false}
        renderOrder={9}
      />
    </>
  );
}
