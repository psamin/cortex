import { useEffect, useState } from "react";
import { Link, Outlet, useMatch } from "react-router-dom";
import { Moon, Sun } from "lucide-react";
import { HealthPills } from "../components/HealthPills";
import { useProjects } from "../api/hooks";
import { BRAND } from "../lib/brand";

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (document.documentElement.dataset.theme as "dark" | "light") || "light",
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

/** Isometric cube mark. */
export function Logo({ className = "h-[18px] w-[18px]" }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
      <path d="M10 1.8 17.2 6v8L10 18.2 2.8 14V6z" />
      <path d="M2.8 6 10 10.2 17.2 6M10 10.2v8" />
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
      <div className="h-[10px] shrink-0" style={{ background: "var(--bar)" }} aria-hidden />
      <header className="flex h-14 shrink-0 items-center gap-6 border-b border-line px-5 sm:px-8">
        <Link to="/" className="flex items-center gap-2.5 text-ink">
          <Logo />
          <span className="text-[15px] font-medium tracking-[-0.01em]">{BRAND}</span>
        </Link>
        {pid && (
          <div className="flex min-w-0 items-center gap-3">
            <span className="h-4 w-px bg-[var(--line-strong)]" aria-hidden />
            <span className="truncate text-sm text-ink-2">{project?.name ?? <span className="font-mono text-xs">{pid}</span>}</span>
            {project && <span className="label hidden sm:inline">{project.mode === "sim" ? "Physics sim" : "Real robot"}</span>}
          </div>
        )}
        <nav className="ml-auto flex items-center gap-5">
          <Link to="/" className="label hidden hover:text-ink md:inline">
            Projects
          </Link>
          <HealthPills />
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-full p-1.5 text-ink-3 hover:bg-rail hover:text-ink"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </nav>
      </header>
      <div className="min-h-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
