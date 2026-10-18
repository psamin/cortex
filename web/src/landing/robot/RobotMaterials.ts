import * as THREE from "three";
import { useEffect } from "react";
import { useTheme, type Theme } from "../../lib/theme";

/**
 * The drawing's palette per theme. Light: white faces, near-black ink. Dark keeps the same
 * structure inverted (dark grey faces, light ink), matching the sketch kit's dark tokens.
 */
const LIGHT = {
  ink: "#171717",
  paper: "#FFFFFF",
  light: "#EDEDED",
  mid: "#D9D9D9",
  deep: "#BDBDBD",
  dark: "#242424",
  black: "#111111",
  grid: "#EAEAEA",
  gridCell: "#EFEFEF",
  gridSection: "#E2E2E2",
  construction: "#BDBDBD",
  guide: "#D4D4D4",
  label: "#858585",
  shadow: "#000000",
  ghost: "#9A9A9A",
  ring: "#666666",
};
type ScenePalette = typeof LIGHT;
const DARK: ScenePalette = {
  ink: "#E6E6E6",
  paper: "#0E0F0F",
  light: "#3A3B3B",
  mid: "#2E2F2F",
  deep: "#242525",
  dark: "#D4D4D4",
  black: "#EDEDED",
  grid: "#1E1F1F",
  gridCell: "#1A1B1B",
  gridSection: "#262727",
  construction: "#4A4B4B",
  guide: "#3A3B3B",
  label: "#8C8C8C",
  shadow: "#000000",
  ghost: "#5C5C5C",
  ring: "#8C8C8C",
};

export function scenePalette(theme: Theme): ScenePalette {
  return theme === "dark" ? DARK : LIGHT;
}

/** Light-mode values, kept for code that has no theme (tests, static fallbacks). */
export const INK = LIGHT.ink;
export const PALETTE = LIGHT;

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

/** The current theme's palette; also recolours the shared materials when the theme flips. */
export function useScenePalette(): ScenePalette {
  const [theme] = useTheme();
  const pal = scenePalette(theme);
  useEffect(() => {
    const mats = robotMaterials();
    (Object.keys(mats) as MaterialKey[]).forEach((k) => mats[k].color.set(pal[k]));
  }, [pal]);
  return pal;
}
