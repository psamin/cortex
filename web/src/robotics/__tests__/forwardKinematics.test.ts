import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { ForwardKinematics, tcpOf } from "../forwardKinematics";
import { DIMS, ROBOT } from "../robotConfig";
import { mulberry } from "./rng";

const Z = [0, 0, 0, 0, 0, 0];
const close = (a: ArrayLike<number>, b: ArrayLike<number>, d = 6) => {
  for (let i = 0; i < b.length; i++) expect(a[i]).toBeCloseTo(b[i], d);
};
const reachX = DIMS.forearmToRoll + DIMS.rollToWrist + DIMS.wristToFlange + DIMS.flangeToTool + DIMS.gripper;

describe("forward kinematics", () => {
  it("zero configuration: upper arm vertical, forearm along +X", () => {
    const fk = new ForwardKinematics(ROBOT).compute(Z);
    close(fk.positions[0], [0, DIMS.baseHeight - DIMS.turretHeight, 0]);
    close(fk.positions[1], [0, DIMS.baseHeight, 0]);
    close(fk.positions[2], [0, DIMS.baseHeight + DIMS.upperArm, 0]);
    close(fk.tcpPosition, [reachX, DIMS.baseHeight + DIMS.upperArm, 0]);
    close(fk.toolDirection, [1, 0, 0]);
  });

  it("base yaw of +90deg swings the arm from +X to -Z", () => {
    close(tcpOf(ROBOT, [Math.PI / 2, 0, 0, 0, 0, 0]), [0, DIMS.baseHeight + DIMS.upperArm, -reachX]);
  });

  it("shoulder -90deg lays the upper arm along +X and points the forearm down", () => {
    const tcp = tcpOf(ROBOT, [0, -Math.PI / 2, 0, 0, 0, 0]);
    close(tcp, [DIMS.upperArm, DIMS.baseHeight - reachX, 0]);
  });

  it("wrist pitch +90deg points the tool straight up", () => {
    const fk = new ForwardKinematics(ROBOT).compute([0, 0, 0, 0, Math.PI / 2, 0]);
    close(fk.toolDirection, [0, 1, 0]);
    const wrist = fk.positions[4];
    close(fk.tcpPosition, [wrist[0], wrist[1] + DIMS.wristToFlange + DIMS.flangeToTool + DIMS.gripper, 0]);
  });

  it("flange roll does not move the TCP (tool on the roll axis)", () => {
    close(tcpOf(ROBOT, [0.3, -0.4, -0.6, 0.2, -0.5, 0]), tcpOf(ROBOT, [0.3, -0.4, -0.6, 0.2, -0.5, 2.1]), 9);
  });

  it("matches a Three.js Object3D hierarchy built from the same config (renderer parity)", () => {
    const root = new THREE.Object3D();
    const groups: THREE.Object3D[] = [];
    let parent = root;
    for (const j of ROBOT.joints) {
      const g = new THREE.Object3D();
      g.position.set(...j.offset);
      parent.add(g);
      groups.push(g);
      parent = g;
    }
    const tool = new THREE.Object3D();
    tool.position.set(...ROBOT.toolOffset);
    parent.add(tool);

    const fk = new ForwardKinematics(ROBOT);
    const rng = mulberry(7);
    for (let k = 0; k < 200; k++) {
      const q = ROBOT.joints.map((j) => j.min + rng() * (j.max - j.min));
      groups.forEach((g, i) => g.quaternion.setFromAxisAngle(new THREE.Vector3(...ROBOT.joints[i].axis), q[i]));
      root.updateMatrixWorld(true);
      fk.compute(q);
      const w = new THREE.Vector3();
      groups.forEach((g, i) => close(fk.positions[i], g.getWorldPosition(w).toArray(), 9));
      close(fk.tcpPosition, tool.getWorldPosition(w).toArray(), 9);
    }
  });

  it("stays numerically stable: finite output and orthonormal rotations", () => {
    const fk = new ForwardKinematics(ROBOT);
    const rng = mulberry(3);
    for (let k = 0; k < 1000; k++) {
      const q = ROBOT.joints.map(() => (rng() - 0.5) * 40);
      fk.compute(q);
      const R = fk.tcpRotation;
      for (const v of fk.tcpPosition) expect(Number.isFinite(v)).toBe(true);
      const det =
        R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
      expect(det).toBeCloseTo(1, 9);
    }
  });
});
