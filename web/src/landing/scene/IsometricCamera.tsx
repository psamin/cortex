import { useLayoutEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

export interface Framing {
  /** Screen x (0..1) where the look-at point should appear. */
  anchorX: number;
  /** Screen y (0..1, from top) where the look-at point should appear. */
  anchorY: number;
  /** Pixels per meter. */
  pixelsPerMeter: number;
}

/** Fixed three-quarter view: ~30 deg elevation, 45 deg azimuth. */
export const CAMERA_ELEVATION = THREE.MathUtils.degToRad(30);
export const CAMERA_AZIMUTH = THREE.MathUtils.degToRad(45);
export const LOOK_AT = new THREE.Vector3(0, 0.55, 0);

export function framingFor(width: number, height: number): Framing {
  if (width >= 1024) {
    // Robot on the right half, base around 2/3 across; scale to hero height.
    return { anchorX: 0.71, anchorY: 0.55, pixelsPerMeter: Math.min(height / 2.3, width / 3.35) };
  }
  // Stacked layout: canvas is its own block under the copy; keep the arm high in the block.
  return { anchorX: 0.6, anchorY: 0.4, pixelsPerMeter: Math.min(height / 1.95, width / 2.25) };
}

/**
 * Orthographic camera with an asymmetric frustum, so the robot can sit right-of-center
 * in a full-width canvas while the camera orientation stays fixed.
 */
export function IsometricCamera({ framing }: { framing?: (w: number, h: number) => Framing }) {
  const camera = useThree((s) => s.camera) as THREE.OrthographicCamera;
  const size = useThree((s) => s.size);

  useLayoutEffect(() => {
    const f = (framing ?? framingFor)(size.width, size.height);
    const W = size.width / f.pixelsPerMeter;
    const H = size.height / f.pixelsPerMeter;
    // R3F would otherwise reset the frustum to the canvas size on resize.
    (camera as THREE.OrthographicCamera & { manual?: boolean }).manual = true;
    camera.left = -f.anchorX * W;
    camera.right = (1 - f.anchorX) * W;
    camera.top = f.anchorY * H;
    camera.bottom = -(1 - f.anchorY) * H;
    camera.zoom = 1;
    camera.near = 0.1;
    camera.far = 60;
    const d = 20;
    camera.position.set(
      LOOK_AT.x + d * Math.cos(CAMERA_ELEVATION) * Math.sin(CAMERA_AZIMUTH),
      LOOK_AT.y + d * Math.sin(CAMERA_ELEVATION),
      LOOK_AT.z + d * Math.cos(CAMERA_ELEVATION) * Math.cos(CAMERA_AZIMUTH),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(LOOK_AT);
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height, framing]);

  return null;
}
