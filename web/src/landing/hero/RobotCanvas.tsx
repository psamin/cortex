import { lazy, Suspense, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { Canvas } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { RobotArm } from "../robot/RobotArm";
import { RenderStyleContext } from "../robot/RobotGeometry";
import { useScenePalette } from "../robot/RobotMaterials";
import { debugStore, RuntimeContext, type RobotRuntime } from "../robot/runtime";
import { IsometricCamera } from "../scene/IsometricCamera";
import {
  EngineeringGrid,
  FloorConstruction,
  GroundShadow,
  SceneLighting,
  TargetMarkers,
} from "../scene/SceneDecor";
const PhysicsLayer = lazy(() => import("../physics/PhysicsLayer"));

/** Reachable-workspace approximation (debug only). */
function WorkspaceShell({ runtime }: { runtime: RobotRuntime }) {
  const c = useScenePalette();
  const ws = runtime.controller.workspace;
  const ring = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * ws.rMax, ws.center[1], Math.sin(a) * ws.rMax));
    }
    return pts;
  }, [ws]);
  return (
    <group>
      <mesh position={ws.center}>
        <sphereGeometry args={[ws.rMax, 32, 16]} />
        <meshBasicMaterial color={c.ghost} wireframe transparent opacity={0.15} />
      </mesh>
      <Line points={ring} color={c.ring} lineWidth={1} />
    </group>
  );
}

export default function RobotCanvas({ runtime }: { runtime: RobotRuntime }) {
  const flags = useSyncExternalStore(debugStore.subscribe, debugStore.get, debugStore.get);
  const tcpRef = useRef<THREE.Object3D | null>(null);
  const style = useMemo(() => ({ edges: flags.edges, outlines: flags.outlines }), [flags.edges, flags.outlines]);
  // Stop rendering while the hero is scrolled out of view, so the sections below scroll smoothly.
  const [onScreen, setOnScreen] = useState(true);
  const ioRef = useRef<IntersectionObserver | null>(null);
  useEffect(() => () => ioRef.current?.disconnect(), []);

  return (
    <Canvas
      orthographic
      flat
      dpr={[1, 2]}
      frameloop={onScreen ? "always" : "never"}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        runtime.canvas = gl.domElement;
        gl.setClearColor(0xffffff, 0);
        ioRef.current?.disconnect();
        ioRef.current = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting));
        ioRef.current.observe(gl.domElement);
      }}
      style={{ pointerEvents: "none" }}
      aria-hidden
      data-testid="robot-canvas"
    >
      <RuntimeContext.Provider value={runtime}>
        <IsometricCamera />
        <SceneLighting />
        <EngineeringGrid visible={flags.grid} />
        <FloorConstruction />
        <GroundShadow />
        <RenderStyleContext.Provider value={style}>
          <RobotArm tcpRef={tcpRef} showAxes={flags.axes} />
          <Suspense fallback={null}>
            <PhysicsLayer debug={flags.colliders} />
          </Suspense>
        </RenderStyleContext.Provider>
        <TargetMarkers />
        {flags.workspace && <WorkspaceShell runtime={runtime} />}
      </RuntimeContext.Provider>
    </Canvas>
  );
}
