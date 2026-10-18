import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { CuboidCollider, CylinderCollider, RigidBody, type RapierRigidBody } from "@react-three/rapier";
import * as THREE from "three";
import { Part } from "../robot/RobotGeometry";
import { roundedBox } from "../robot/geometryKit";
import { useRuntime } from "../robot/runtime";
import { CUBE_SIZE, pickAndPlaceWaypoints, STATIONS } from "../../robotics/graspController";

// Rapier body types (avoid importing the wasm package's enum directly).
const DYNAMIC = 0;
const KINEMATIC_POSITION = 2;

/**
 * Physics layer for the demo: static floor + base colliders and one dynamic cube.
 * The arm itself stays kinematic (driven by IK); while grasped, the cube is attached
 * with an explicit temporary constraint (kinematic follow of the TCP frame), not by
 * simulated contact friction. On release it becomes dynamic again with the TCP velocity.
 */
export function GraspableCube() {
  const rt = useRuntime();
  const body = useRef<RapierRigidBody>(null);
  const geometry = useMemo(() => roundedBox(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 0.004), []);
  const s = useMemo(
    () => ({
      rel: new THREE.Matrix4(),
      tcp: new THREE.Matrix4(),
      m: new THREE.Matrix4(),
      pos: new THREE.Vector3(),
      quat: new THREE.Quaternion(),
      scl: new THREE.Vector3(),
      prevTcp: new THREE.Vector3(),
      vel: new THREE.Vector3(),
      euler: new THREE.Euler(),
    }),
    [],
  );

  /** TCP pose from the controller's FK (row-major 3x3 + position). */
  const tcpMatrix = (out: THREE.Matrix4) => {
    const fk = rt.controller.fk;
    const R = fk.tcpRotation;
    const p = fk.tcpPosition;
    out.set(R[0], R[1], R[2], p[0], R[3], R[4], R[5], p[1], R[6], R[7], R[8], p[2], 0, 0, 0, 1);
    return out;
  };

  const cubeYaw = (b: RapierRigidBody) => {
    const r = b.rotation();
    s.quat.set(r.x, r.y, r.z, r.w);
    const v = new THREE.Vector3(1, 0, 0).applyQuaternion(s.quat);
    return Math.atan2(-v.z, v.x);
  };

  useFrame((_, dt) => {
    const b = body.current;
    const c = rt.controller;
    if (!b) return;

    if (rt.cube.requested) {
      rt.cube.requested = false;
      const t = b.translation();
      const from: [number, number, number] = [t.x, t.y, t.z];
      // Place at whichever station is farther from the cube.
      const d0 = Math.hypot(t.x - STATIONS[0][0], t.z - STATIONS[0][2]);
      const d1 = Math.hypot(t.x - STATIONS[1][0], t.z - STATIONS[1][2]);
      const place = d0 > d1 ? STATIONS[0] : STATIONS[1];
      const started = c.startGrasp(
        pickAndPlaceWaypoints(from, cubeYaw(b), place),
        () => {
          // Attach only when the TCP is genuinely at the cube with the tool pointing down.
          const ct = b.translation();
          const tcp = c.fk.tcpPosition;
          const near = Math.hypot(tcp[0] - ct.x, tcp[1] - ct.y, tcp[2] - ct.z) < 0.025;
          const down = c.fk.toolDirection[1] < -0.95;
          if (!near || !down) {
            rt.cube.status = "MISS";
            return false;
          }
          const cr = b.rotation();
          s.m.compose(s.pos.set(ct.x, ct.y, ct.z), s.quat.set(cr.x, cr.y, cr.z, cr.w), s.scl.set(1, 1, 1));
          s.rel.copy(tcpMatrix(s.tcp).invert()).multiply(s.m);
          b.setBodyType(KINEMATIC_POSITION as never, true);
          rt.cube.attached = true;
          rt.cube.status = "HOLD";
          return true;
        },
        () => {
          b.setBodyType(DYNAMIC as never, true);
          b.setLinvel({ x: s.vel.x, y: s.vel.y, z: s.vel.z }, true);
          b.setAngvel({ x: 0, y: 0, z: 0 }, true);
          rt.cube.attached = false;
          rt.cube.status = "PLACED";
        },
      );
      rt.cube.status = started ? "APPROACH" : "BUSY";
    }

    // TCP velocity (for a natural hand-off on release).
    const tcp = c.fk.tcpPosition;
    s.pos.set(tcp[0], tcp[1], tcp[2]);
    if (dt > 0) s.vel.copy(s.pos).sub(s.prevTcp).divideScalar(Math.max(dt, 1e-3));
    s.prevTcp.copy(s.pos);

    if (rt.cube.attached) {
      s.m.copy(tcpMatrix(s.tcp)).multiply(s.rel);
      s.m.decompose(s.pos, s.quat, s.scl);
      b.setNextKinematicTranslation({ x: s.pos.x, y: s.pos.y, z: s.pos.z });
      b.setNextKinematicRotation({ x: s.quat.x, y: s.quat.y, z: s.quat.z, w: s.quat.w });
    }
  }, -1);

  const start = STATIONS[0];
  return (
    <>
      {/* Static world: floor, base plate and pedestal (so the cube cannot fall into the robot). */}
      <RigidBody type="fixed" colliders={false}>
        <CuboidCollider args={[4, 0.05, 4]} position={[0, -0.05, 0]} />
        <CuboidCollider args={[0.3, 0.018, 0.3]} position={[0, 0.018, 0]} />
        <CylinderCollider args={[0.09, 0.2]} position={[0, 0.09, 0]} />
      </RigidBody>
      <RigidBody
        ref={body}
        colliders={false}
        position={start}
        rotation={[0, 0, 0]}
        friction={0.9}
        restitution={0.05}
        linearDamping={0.2}
        angularDamping={0.4}
      >
        <CuboidCollider args={[CUBE_SIZE / 2, CUBE_SIZE / 2, CUBE_SIZE / 2]} mass={0.12} />
        <Part geometry={geometry} material="mid" threshold={30} />
      </RigidBody>
    </>
  );
}
