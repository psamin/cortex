import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

export type Axis = "x" | "y" | "z";

/** Re-orient a geometry built along +Y onto the requested axis. */
function orient(g: THREE.BufferGeometry, axis: Axis): THREE.BufferGeometry {
  if (axis === "x") g.rotateZ(-Math.PI / 2); // +Y -> +X
  if (axis === "z") g.rotateX(Math.PI / 2); // +Y -> +Z
  return g;
}

/**
 * Lathe from a [radius, along] profile, revolved around `axis`.
 * Stepped profiles give layered housings and seams for free.
 */
export function lathe(profile: Array<[number, number]>, axis: Axis = "y", segments = 56) {
  const pts = profile.map(([r, h]) => new THREE.Vector2(r, h));
  return orient(new THREE.LatheGeometry(pts, segments), axis);
}

export function cylinder(radius: number, length: number, axis: Axis = "y", segments = 48) {
  return orient(new THREE.CylinderGeometry(radius, radius, length, segments, 1), axis);
}

export function roundedBox(w: number, h: number, d: number, r: number) {
  return new RoundedBoxGeometry(w, h, d, 2, r);
}

/**
 * Tapered "bone" outline: two circles (radii r0 at origin, r1 at +L along Y) joined by
 * external tangents. The classic silhouette of an industrial robot link.
 */
export function boneShape(L: number, r0: number, r1: number): THREE.Shape {
  const phi = Math.asin((r0 - r1) / L);
  const s = new THREE.Shape();
  s.moveTo(r0 * Math.cos(phi), r0 * Math.sin(phi));
  s.lineTo(r1 * Math.cos(phi), L + r1 * Math.sin(phi));
  s.absarc(0, L, r1, phi, Math.PI - phi, false);
  s.lineTo(r0 * Math.cos(Math.PI - phi), r0 * Math.sin(Math.PI - phi));
  s.absarc(0, 0, r0, Math.PI - phi, 2 * Math.PI + phi, false);
  return s;
}

/** Extrude a shape (drawn in XY) symmetrically along Z with a crisp chamfer. */
export function extrude(shape: THREE.Shape, depth: number, bevel = 0.01, curveSegments = 40) {
  const g = new THREE.ExtrudeGeometry(shape, {
    depth: depth - 2 * bevel,
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 1,
    curveSegments,
  });
  g.translate(0, 0, -(depth - 2 * bevel) / 2);
  return g;
}

/** Extruded bone link along +Y (shoulder -> elbow). */
export function boneLink(L: number, r0: number, r1: number, depth: number, bevel = 0.012) {
  return extrude(boneShape(L, r0, r1), depth, bevel);
}

/** Yoke cheek: flat base, rounded top around an axis at height `axisY`. */
export function yokeShape(baseHalf: number, baseY: number, axisY: number, r: number) {
  const s = new THREE.Shape();
  s.moveTo(-baseHalf, baseY);
  s.lineTo(baseHalf, baseY);
  s.lineTo(r, axisY);
  s.absarc(0, axisY, r, 0, Math.PI, false);
  s.lineTo(-baseHalf, baseY);
  return s;
}

/** Rotate a +Y-built geometry so it runs along +X, then shift along X. */
export function alongX(g: THREE.BufferGeometry, shiftX = 0) {
  g.rotateZ(-Math.PI / 2);
  g.translate(shiftX, 0, 0);
  return g;
}
