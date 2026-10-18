import { useEffect, useState } from "react";
import { Link, Outlet, useMatch } from "react-router-dom";
import { ChevronRight, Moon, Sun } from "lucide-react";
import { HealthPills } from "../components/HealthPills";
import { useProjects } from "../api/hooks";

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (document.documentElement.dataset.theme as "dark" | "light") || "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("cortex.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return [theme, setTheme] as const;
}

export function Logo() {
  return (
    <svg viewBox="0 0 20 20" className="h-5 w-5" aria-hidden>
      <rect width="20" height="20" rx="4" className="fill-rail" />
      <path d="M5 14V6h3l2 4 2-4h3v8" className="stroke-servo" strokeWidth="1.8" fill="none" strokeLinejoin="round" />
    </svg>
  );
}

export function AppShell() {
  const [theme, setTheme] = useTheme();
  const m = useMatch("/p/:pid/*");
  const pid = m?.params.pid;
  const projects = useProjects();
  const project = pid ? projects.data?.find((p) => p.id === pid) : undefined;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line px-3">
        <Link to="/studio" className="flex items-center gap-2 rounded px-1 py-0.5 hover:bg-rail">
          <Logo />
          <span className="text-sm font-semibold tracking-[-0.01em]">Cortex</span>
        </Link>
        {pid && (
          <div className="flex min-w-0 items-center gap-1.5 text-sm">
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-4" />
            <span className="truncate font-medium text-ink-2">{project?.name ?? <span className="font-mono text-xs">{pid}</span>}</span>
            {project && (
              <span className="rounded-sm bg-rail px-1.5 py-px text-2xs font-medium uppercase tracking-wide text-ink-3">
                {project.mode === "sim" ? "sim" : "real robot"}
              </span>
            )}
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <HealthPills />
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded p-1.5 text-ink-3 hover:bg-rail hover:text-ink"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </header>
      <div className="min-h-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
