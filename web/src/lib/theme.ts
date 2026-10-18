import { useEffect, useSyncExternalStore } from "react";

/**
 * One theme for the whole site (landing + studio). The value lives on <html data-theme>,
 * set before first paint by the inline script in index.html: the saved choice, else the
 * system preference. Every component reads it through useTheme(), so they never disagree.
 */
export type Theme = "light" | "dark";

const KEY = "cortex.theme";
const EVENT = "cortex:theme";

function stored(): Theme | null {
  try {
    const t = localStorage.getItem(KEY);
    return t === "dark" || t === "light" ? t : null;
  } catch {
    return null;
  }
}

function system(): Theme {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function apply(t: Theme) {
  if (document.documentElement.dataset.theme === t) return;
  document.documentElement.dataset.theme = t;
  window.dispatchEvent(new CustomEvent(EVENT, { detail: t }));
}

/** Choose a theme explicitly; it is remembered across visits. */
export function setTheme(t: Theme) {
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable: still applies for this visit */
  }
  apply(t);
}

function subscribe(cb: () => void) {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const theme = useSyncExternalStore(subscribe, currentTheme, () => "light" as Theme);
  return [theme, setTheme];
}

/** Mount once: follows the OS setting until the visitor picks one, and syncs other tabs. */
export function useThemeSync() {
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onSystem = () => stored() === null && apply(system());
    const onStorage = (e: StorageEvent) => e.key === KEY && apply(stored() ?? system());
    mq?.addEventListener("change", onSystem);
    window.addEventListener("storage", onStorage);
    return () => {
      mq?.removeEventListener("change", onSystem);
      window.removeEventListener("storage", onStorage);
    };
  }, []);
}

/** A token as a CSS colour string ("rgb(23 23 23)"), for canvas/WebGL code that can't use CSS. */
export function tokenColor(name: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
  return v.startsWith("#") || v.startsWith("rgb") ? v : `rgb(${v})`;
}
