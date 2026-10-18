import * as THREE from "three";

export const INK = "#171717";

export const PALETTE = {
  paper: "#FFFFFF",
  light: "#EDEDED",
  mid: "#D9D9D9",
  deep: "#BDBDBD",
  dark: "#242424",
  black: "#111111",
  grid: "#EAEAEA",
  label: "#858585",
} as const;

export type MaterialKey = "light" | "mid" | "deep" | "dark" | "black";

let cache: Record<MaterialKey, THREE.MeshToonMaterial> | null = null;

/**
 * Three-step toon ramp: lit faces read white, side faces light gray, shadowed faces
 * mid gray, which is the flat-shaded look of isometric technical illustrations.
 */
function ramp(): THREE.DataTexture {
  const steps = [150, 205, 255];
  const data = new Uint8Array(steps.length * 4);
  steps.forEach((v, i) => data.set([v, v, v, 255], i * 4));
  const tex = new THREE.DataTexture(data, steps.length, 1, THREE.RGBAFormat);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/** Shared materials (created once, reused by every part). */
export function robotMaterials() {
  if (cache) return cache;
  const gradientMap = ramp();
  const mk = (color: string) =>
    new THREE.MeshToonMaterial({
      color,
      gradientMap,
      // Push faces back slightly so edge lines never z-fight with their surfaces.
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  cache = {
    light: mk(PALETTE.light),
    mid: mk(PALETTE.mid),
    deep: mk(PALETTE.deep),
    dark: mk(PALETTE.dark),
    black: mk(PALETTE.black),
  };
  return cache;
}
