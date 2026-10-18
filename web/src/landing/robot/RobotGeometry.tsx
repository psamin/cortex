import { createContext, useContext, useMemo, type ReactNode } from "react";
import type * as THREE from "three";
import { Edges, Outlines } from "@react-three/drei";
import { DIMS } from "../../robotics/robotConfig";
import { robotMaterials, useScenePalette, type MaterialKey } from "./RobotMaterials";
import {
  alongX,
  boneLink,
  boneShape,
  cylinder,
  extrude,
  lathe,
  roundedBox,
  yokeShape,
} from "./geometryKit";

export interface RenderStyle {
  edges: boolean;
  outlines: boolean;
}
export const RenderStyleContext = createContext<RenderStyle>({ edges: true, outlines: true });

type V3 = [number, number, number];

/** Silhouette thickness in meters. */
export const OUTLINE_WORLD = 0.0035;

interface PartProps {
  geometry: THREE.BufferGeometry;
  material?: MaterialKey;
  position?: V3;
  rotation?: V3;
  /** Crease angle (deg) above which an edge line is drawn. */
  threshold?: number;
  edges?: boolean;
  outline?: boolean;
  children?: ReactNode;
}

/** One mesh drawn in the house style: toon fill, crease edges, screen-space silhouette. */
export function Part({
  geometry,
  material = "light",
  position,
  rotation,
  threshold = 28,
  edges = true,
  outline = true,
  children,
}: PartProps) {
  const style = useContext(RenderStyleContext);
  const mats = robotMaterials();
  const { ink } = useScenePalette();
  const dark = material === "dark" || material === "black";
  return (
    <mesh geometry={geometry} material={mats[material]} position={position} rotation={rotation}>
      {style.edges && edges && !dark && <Edges threshold={threshold} color={ink} lineWidth={1} />}
      {/* World-space thickness: under the fixed orthographic scale this is a constant ~1.3 px line. */}
      {style.outlines && outline && <Outlines thickness={OUTLINE_WORLD} color={ink} angle={Math.PI} />}
      {children}
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/* Geometry is built once per page and shared.                          */
/* ------------------------------------------------------------------ */

function buildGeometry() {
  const D = DIMS;
  const turretShoulder = D.turretHeight; // shoulder axis height in the J1 frame
  return {
    // Base (world frame, floor at y = 0)
    plate: roundedBox(0.6, D.plinthHeight * 0.6, 0.6, 0.01),
    plateInset: roundedBox(0.5, 0.008, 0.5, 0.004),
    bolt: cylinder(0.017, 0.014, "y", 6),
    pedestal: lathe(
      [
        [0, 0.036],
        [0.21, 0.036],
        [0.21, 0.056],
        [0.19, 0.068],
        [0.19, 0.112],
        [0.193, 0.112],
        [0.193, 0.124],
        [0.18, 0.124],
        [0.18, 0.165],
        [0.168, 0.176],
        [0, 0.176],
      ],
      "y",
    ),
    pedestalBand: cylinder(0.196, 0.012, "y"),
    cablePort: roundedBox(0.08, 0.05, 0.05, 0.008),

    // J1 turret (origin on the yaw bearing)
    turretRing: cylinder(0.172, 0.016, "y"),
    drum: lathe(
      [
        [0, 0.016],
        [0.165, 0.016],
        [0.165, 0.058],
        [0.15, 0.072],
        [0, 0.072],
      ],
      "y",
    ),
    cheek: extrude(yokeShape(0.13, 0.05, turretShoulder, 0.112), 0.042, 0.009),
    shoulderMotor: cylinder(0.086, 0.07, "z"),
    shoulderMotorBand: cylinder(0.089, 0.01, "z"),
    shoulderCap: cylinder(0.062, 0.01, "z"),
    shoulderBoss: cylinder(0.026, 0.012, "z"),
    shoulderRing: cylinder(0.07, 0.008, "z"),

    // J2 upper arm (origin on the shoulder axis, elbow at +upperArm along Y)
    upperArm: boneLink(D.upperArm, 0.105, 0.08, 0.15, 0.012),
    upperCover: extrude(boneShape(D.upperArm, 0.07, 0.052), 0.01, 0.003),
    screw: cylinder(0.0065, 0.006, "z", 12),
    slot: roundedBox(0.026, 0.2, 0.008, 0.004),

    // J3 elbow + rear forearm (origin on the elbow axis, forearm along +X)
    elbowHousing: cylinder(0.088, 0.2, "z"),
    elbowCap: cylinder(0.064, 0.01, "z"),
    elbowBoss: cylinder(0.028, 0.012, "z"),
    forearmRear: lathe(
      [
        [0, -0.215],
        [0.058, -0.215],
        [0.068, -0.205],
        [0.068, -0.11],
        [0.072, -0.11],
        [0.072, -0.098],
        [0.08, -0.098],
        [0.08, D.forearmToRoll - 0.012],
        [0, D.forearmToRoll - 0.012],
      ],
      "x",
    ),
    rollSeam: cylinder(0.078, 0.012, "x"),

    // J4 forearm roll (origin at roll joint, wrist axis at +rollToWrist along X)
    forearm: lathe(
      [
        [0, 0],
        [0.074, 0],
        [0.074, 0.022],
        [0.068, 0.034],
        [0.061, 0.24],
        [0.056, 0.29],
        [0, 0.29],
      ],
      "x",
    ),
    forkBridge: roundedBox(0.07, 0.092, 0.14, 0.012),
    forkCheek: alongX(extrude(boneShape(D.rollToWrist - 0.3, 0.046, 0.05), 0.022, 0.005), 0.3),

    // J5 wrist (origin on the wrist pitch axis)
    knuckle: cylinder(0.046, 0.092, "z"),
    knuckleCap: cylinder(0.034, 0.008, "z"),
    nose: lathe(
      [
        [0, 0],
        [0.043, 0],
        [0.043, 0.058],
        [0.037, 0.072],
        [0.037, D.wristToFlange - 0.004],
        [0, D.wristToFlange - 0.004],
      ],
      "x",
    ),

    // J6 flange + gripper (origin at flange roll, tool mount at +flangeToTool)
    flange: lathe(
      [
        [0, 0],
        [0.047, 0],
        [0.047, 0.014],
        [0, 0.014],
      ],
      "x",
    ),
    adapter: lathe(
      [
        [0, 0.014],
        [0.04, 0.014],
        [0.04, 0.036],
        [0, 0.036],
      ],
      "x",
    ),
    gripperBody: roundedBox(0.075, 0.12, 0.064, 0.01),
    gripperRail: roundedBox(0.006, 0.1, 0.014, 0.002),
    finger: roundedBox(D.gripper - 0.015, 0.018, 0.04, 0.005),
    fingerPad: roundedBox(0.04, 0.004, 0.034, 0.0015),
  };
}

type Geo = ReturnType<typeof buildGeometry>;
let geoCache: Geo | null = null;
export function useRobotGeometry(): Geo {
  return useMemo(() => (geoCache ??= buildGeometry()), []);
}

/* ------------------------------------------------------------------ */
/* Link visuals. Each lives inside its joint's group (local frame).     */
/* ------------------------------------------------------------------ */

export function BaseVisual() {
  const g = useRobotGeometry();
  const h = DIMS.plinthHeight * 0.6;
  const bolts: V3[] = [
    [0.25, h + 0.007, 0.25],
    [-0.25, h + 0.007, 0.25],
    [0.25, h + 0.007, -0.25],
    [-0.25, h + 0.007, -0.25],
  ];
  return (
    <group>
      <Part geometry={g.plate} position={[0, h / 2, 0]} />
      <Part geometry={g.plateInset} material="mid" position={[0, h + 0.004, 0]} outline={false} />
      {bolts.map((p, i) => (
        <Part key={i} geometry={g.bolt} material="dark" position={p} />
      ))}
      <Part geometry={g.pedestal} position={[0, 0, 0]} />
      <Part geometry={g.pedestalBand} material="dark" position={[0, 0.118, 0]} outline={false} />
      <Part geometry={g.cablePort} material="mid" position={[0.2, 0.06, 0]} />
    </group>
  );
}

export function TurretVisual() {
  const g = useRobotGeometry();
  const sy = DIMS.turretHeight;
  return (
    <group>
      <Part geometry={g.turretRing} material="dark" position={[0, 0.008, 0]} />
      <Part geometry={g.drum} />
      <Part geometry={g.cheek} position={[0, 0, 0.116]} />
      <Part geometry={g.cheek} position={[0, 0, -0.116]} />
      {/* Shoulder drive on +Z */}
      <Part geometry={g.shoulderMotor} position={[0, sy, 0.172]} />
      <Part geometry={g.shoulderMotorBand} material="dark" position={[0, sy, 0.15]} outline={false} />
      <Part geometry={g.shoulderCap} material="dark" position={[0, sy, 0.211]} />
      <Part geometry={g.shoulderBoss} material="mid" position={[0, sy, 0.218]} />
      {/* Bearing cap on -Z */}
      <Part geometry={g.shoulderRing} material="dark" position={[0, sy, -0.141]} />
      <Part geometry={g.shoulderBoss} material="mid" position={[0, sy, -0.148]} />
    </group>
  );
}

export function UpperArmVisual() {
  const g = useRobotGeometry();
  const face = 0.075 + 0.005;
  const screws: Array<[number, number]> = [
    [-0.035, 0.07],
    [0.035, 0.07],
    [-0.028, DIMS.upperArm - 0.07],
    [0.028, DIMS.upperArm - 0.07],
  ];
  return (
    <group>
      <Part geometry={g.upperArm} />
      {[1, -1].map((s) => (
        <group key={s}>
          <Part geometry={g.upperCover} material="mid" position={[0, 0, s * face]} threshold={40} />
          <Part geometry={g.slot} material="dark" position={[0, DIMS.upperArm * 0.48, s * (face + 0.005)]} outline={false} />
          {screws.map(([x, y], i) => (
            <Part key={i} geometry={g.screw} material="dark" position={[x, y, s * (face + 0.006)]} outline={false} />
          ))}
        </group>
      ))}
    </group>
  );
}

export function ElbowVisual() {
  const g = useRobotGeometry();
  return (
    <group>
      <Part geometry={g.forearmRear} />
      <Part geometry={g.elbowHousing} />
      {[1, -1].map((s) => (
        <group key={s}>
          <Part geometry={g.elbowCap} material="dark" position={[0, 0, s * 0.105]} />
          <Part geometry={g.elbowBoss} material="mid" position={[0, 0, s * 0.112]} />
        </group>
      ))}
      <Part geometry={g.rollSeam} material="dark" position={[DIMS.forearmToRoll - 0.006, 0, 0]} outline={false} />
    </group>
  );
}

export function ForearmVisual() {
  const g = useRobotGeometry();
  return (
    <group>
      <Part geometry={g.forearm} />
      <Part geometry={g.forkBridge} position={[0.3, 0, 0]} />
      <Part geometry={g.forkCheek} position={[0, 0, 0.058]} />
      <Part geometry={g.forkCheek} position={[0, 0, -0.058]} />
    </group>
  );
}

export function WristVisual() {
  const g = useRobotGeometry();
  return (
    <group>
      <Part geometry={g.knuckle} />
      <Part geometry={g.knuckleCap} material="dark" position={[0, 0, 0.073]} />
      <Part geometry={g.knuckleCap} material="dark" position={[0, 0, -0.073]} />
      <Part geometry={g.nose} />
    </group>
  );
}

/** Flange + parallel gripper. Finger groups are exposed so the controller can slide them. */
export function GripperVisual({
  fingerRefs,
}: {
  fingerRefs: React.RefObject<(THREE.Group | null)[]>;
}) {
  const g = useRobotGeometry();
  const mount = DIMS.flangeToTool;
  const fingerCenter = mount + 0.015 + (DIMS.gripper - 0.015) / 2;
  return (
    <group>
      <Part geometry={g.flange} material="dark" />
      <Part geometry={g.adapter} material="mid" />
      <Part geometry={g.gripperBody} position={[mount - 0.025, 0, 0]} />
      <Part geometry={g.gripperRail} material="dark" position={[mount + 0.013, 0, 0]} outline={false} />
      {[1, -1].map((s, i) => (
        <group
          key={s}
          ref={(el) => {
            if (fingerRefs.current) fingerRefs.current[i] = el;
          }}
          position={[0, s * 0.044, 0]}
        >
          <Part geometry={g.finger} position={[fingerCenter, 0, 0]} />
          <Part
            geometry={g.fingerPad}
            material="dark"
            position={[mount + DIMS.gripper - 0.03, -s * 0.011, 0]}
            outline={false}
          />
        </group>
      ))}
    </group>
  );
}
