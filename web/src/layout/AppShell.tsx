import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, Outlet, useMatch } from "react-router-dom";
import { ChevronDown, ChevronRight, Moon, Sun } from "lucide-react";
import { describeProvider } from "../components/HealthPills";
import { useEngine, useHealth, useProjects } from "../api/hooks";
import { cn } from "../lib/format";

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">(() => (document.documentElement.dataset.theme as "dark" | "light") || "light");
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
      <rect width="20" height="20" rx="5" className="fill-rail" />
      <path d="M5 14V6h3l2 4 2-4h3v8" className="stroke-servo" strokeWidth="1.8" fill="none" strokeLinejoin="round" />
    </svg>
  );
}

/** Click-to-open popover with click-outside and Escape. */
function Popover({ button, children, align = "right" }: { button: (open: boolean) => ReactNode; children: ReactNode; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <div onClick={() => setOpen((o) => !o)}>{button(open)}</div>
      {open && (
        <div
          onClick={(e) => (e.target as HTMLElement).closest("a") && setOpen(false)}
          className={cn("enter absolute top-full z-50 mt-1.5 min-w-[220px] rounded-xl bg-plate p-1.5 shadow-[0_0_0_1px_var(--line-strong),0_16px_32px_-12px_rgba(0,0,0,.5)]", align === "right" ? "right-0" : "left-0")}
        >
          {children}
        </div>
      )}
    </div>
  );
}

/** One quiet pill; the popover names every provider and labels local fallbacks. */
function HealthSummary() {
  const q = useHealth();
  if (q.isLoading) return <span className="skeleton h-6 w-24 rounded-full" />;
  if (q.error || !q.data)
    return (
      <span className="inline-flex h-6 items-center gap-1.5 rounded-full bg-fault/10 px-2.5 text-xs font-medium text-fault">
        <span className="h-1.5 w-1.5 rounded-full bg-fault" /> Backend offline
      </span>
    );
  const h = q.data;
  const subs = (["db", "storage", "vlm", "agent", "executor", "training"] as const).map((k) => [k, describeProvider(h[k])] as const);
  const fallbacks = subs.filter(([, d]) => d.fallback).length;
  return (
    <Popover
      button={(open) => (
        <button type="button" className={cn("inline-flex h-7 items-center gap-1.5 rounded-full px-2.5 text-xs text-ink-2 hover:bg-rail", open && "bg-rail")}>
          <span className={cn("h-1.5 w-1.5 rounded-full", fallbacks ? "bg-caution" : "bg-pass")} />
          {fallbacks ? `${fallbacks} local fallback${fallbacks === 1 ? "" : "s"}` : "All services connected"}
          <ChevronDown className="h-3 w-3 text-ink-4" />
        </button>
      )}
    >
      <ul className="space-y-0.5 p-1">
        {subs.map(([k, d]) => (
          <li key={k} className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs">
            <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", d.fallback ? "bg-caution" : "bg-pass")} />
            <span className="w-16 shrink-0 text-ink-3">{k}</span>
            <span className="min-w-0 flex-1 truncate text-ink-2" title={d.name}>
              {d.name}
            </span>
            {d.fallback && <span className="text-[10px] font-medium uppercase tracking-wide text-caution">fallback</span>}
          </li>
        ))}
      </ul>
      {h.notes?.length > 0 && <div className="border-t border-line-soft px-3 py-2 text-[11px] leading-relaxed text-ink-3">{h.notes.join(" ")}</div>}
    </Popover>
  );
}

function DetailsMenu({ pid }: { pid: string }) {
  return (
    <Popover
      button={(open) => (
        <button type="button" className={cn("inline-flex h-7 items-center gap-1 rounded-md px-2.5 text-xs text-ink-2 hover:bg-rail hover:text-ink", open && "bg-rail text-ink")}>
          Details <ChevronDown className="h-3 w-3" />
        </button>
      )}
    >
      {(
        [
          ["overview", "Overview", "plan, autopilot, next actions"],
          ["collect", "Collect", "demonstrations by hand or agent"],
          ["analyze", "Analyze", "quality checks and diagnoses"],
          ["improve", "Improve", "datasets, training, evaluations"],
        ] as const
      ).map(([path, label, sub]) => (
        <Link key={path} to={`/p/${pid}/${path}`} className="block rounded-lg px-3 py-2 hover:bg-rail">
          <div className="text-sm text-ink">{label}</div>
          <div className="text-[11px] text-ink-3">{sub}</div>
        </Link>
      ))}
    </Popover>
  );
}

export function AppShell() {
  const [theme, setTheme] = useTheme();
  const m = useMatch("/p/:pid/*");
  const pid = m?.params.pid;
  const projects = useProjects();
  const engine = useEngine();
  const project = pid ? projects.data?.find((p) => p.id === pid) : undefined;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line px-4">
        <Link to="/" className="flex items-center gap-2 rounded-md px-1 py-0.5 hover:bg-rail">
          <Logo />
          <span className="text-sm font-semibold tracking-[-0.01em]">Cortex</span>
        </Link>
        {pid && (
          <div className="flex min-w-0 items-center gap-1.5 text-sm">
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-4" />
            <Link to={`/p/${pid}`} className="truncate font-medium text-ink-2 hover:text-ink">
              {project?.name ?? <span className="font-mono text-xs">{pid}</span>}
            </Link>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          {pid && engine.data && (
            <span className="hidden h-6 items-center gap-1.5 rounded-full bg-rail px-2.5 text-xs text-ink-2 shadow-[0_0_0_1px_var(--line)] md:inline-flex" title={engine.data.model ? `model ${engine.data.model}` : "agent engine"}>
              <span className={cn("h-1.5 w-1.5 rounded-full", engine.data.engine === "rules" ? "bg-caution" : "bg-pass")} />
              {engine.data.label}
            </span>
          )}
          <HealthSummary />
          {pid && <DetailsMenu pid={pid} />}
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-md p-1.5 text-ink-3 hover:bg-rail hover:text-ink"
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
