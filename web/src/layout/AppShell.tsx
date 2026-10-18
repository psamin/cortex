import { BRAND } from "../landing/brand";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, Outlet, useMatch } from "react-router-dom";
import { ChevronDown } from "lucide-react";
import { ThemeToggle } from "../components/ThemeToggle";
import { describeProvider } from "../components/HealthPills";
import { useEngine, useHealth, useProjects } from "../api/hooks";
import { cn } from "../lib/format";

/** The product name, shown in the wordmark. Change it here only. */
export const BRAND_NAME = BRAND.name; // one source of truth: web/src/landing/brand.ts

/** The Kinetic mark: a small isometric cube with a solid top face. */
export function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 22 22" aria-hidden className="text-ink">
      <path d="M11 1.5 19.5 6.4v9.2L11 20.5 2.5 15.6V6.4z" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      <path d="M11 11 19.5 6.4M11 11v9.5M11 11 2.5 6.4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path d="M11 1.5 19.5 6.4 11 11 2.5 6.4z" fill="currentColor" />
    </svg>
  );
}

const statusLabel = "font-mono text-[10.5px] uppercase tracking-[0.14em]";

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
          className={cn("enter sk-depth absolute top-full z-50 mt-2 min-w-[240px] rounded border border-line-strong bg-plate p-1.5", align === "right" ? "right-0" : "left-0")}
        >
          {children}
        </div>
      )}
    </div>
  );
}

/** One quiet status pill, no dropdown; fallbacks are named in its tooltip. */
function HealthSummary() {
  const q = useHealth();
  if (q.isLoading) return <span className="skeleton h-6 w-24 rounded-full" />;
  if (q.error || !q.data)
    return (
      <span className={cn("inline-flex h-7 items-center gap-1.5 px-2 text-fault", statusLabel)}>
        <span className="h-1.5 w-1.5 rounded-full bg-fault" /> Backend offline
      </span>
    );
  const h = q.data;
  const subs = (["db", "storage", "vlm", "agent", "executor", "training"] as const).map((k) => [k, describeProvider(h[k])] as const);
  const fallbacks = subs.filter(([, d]) => d.fallback).length;
  return (
    <span
      className={cn("inline-flex h-7 items-center gap-1.5 px-2.5 text-ink-3", statusLabel)}
      title={fallbacks ? subs.filter(([, d]) => d.fallback).map(([k, d]) => `${k}: ${d.name}`).join("\n") : undefined}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", fallbacks ? "bg-caution" : "bg-pass")} />
      <span className="hidden whitespace-nowrap sm:inline">{fallbacks ? `${fallbacks} local fallback${fallbacks === 1 ? "" : "s"}` : "All services connected"}</span>
    </span>
  );
}

function DetailsMenu({ pid }: { pid: string }) {
  return (
    <Popover
      button={(open) => (
        <button type="button" className={cn("inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-[13px] text-ink-2 hover:text-ink", open && "text-ink")}>
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
        <Link key={path} to={`/p/${pid}/${path}`} className="block rounded-sm px-3 py-2 hover:bg-rail">
          <div className="text-sm text-ink">{label}</div>
          <div className={cn("mt-0.5 text-ink-3", statusLabel, "text-[10px]")}>{sub}</div>
        </Link>
      ))}
    </Popover>
  );
}

export function AppShell() {
  const m = useMatch("/p/:pid/*");
  const pid = m?.params.pid;
  const projects = useProjects();
  const engine = useEngine();
  const project = pid ? projects.data?.find((p) => p.id === pid) : undefined;
  const onStudioHome = !!useMatch("/studio");

  return (
    <div className="flex h-full flex-col">
      {/* The Kinetic accent bar: 10 px, full width. */}
      <div aria-hidden className="h-[10px] shrink-0 bg-servo" />
      <header className="flex h-14 shrink-0 items-center gap-4 border-b border-line px-4 sm:px-6">
        <Link to="/studio" className="flex shrink-0 items-center gap-2.5 rounded-sm py-0.5 text-ink" aria-label={`${BRAND_NAME} studio`}>
          <Logo />
          <span className="text-[15px] font-medium tracking-[-0.01em]">{BRAND_NAME}</span>
        </Link>
        {pid && (
          <div className="flex min-w-0 items-center gap-3 text-[13px]">
            <span aria-hidden className="hidden h-px w-6 shrink-0 bg-[var(--line-strong)] sm:block" />
            <Link to={`/p/${pid}`} className="truncate text-ink-2 hover:text-ink">
              {project?.name ?? <span className="font-mono text-xs">{pid}</span>}
            </Link>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1 sm:gap-2">
          {pid && engine.data && (
            <span className={cn("hidden h-7 items-center gap-1.5 px-2 text-ink-3 md:inline-flex", statusLabel)} title={engine.data.note ?? (engine.data.model ? `model ${engine.data.model}` : "agent engine")}>
              <span className={cn("h-1.5 w-1.5 rounded-full", engine.data.engine === "rules" || engine.data.note ? "bg-caution" : "bg-pass")} />
              {engine.data.label}
            </span>
          )}
          <Link to="/studio/agents" className="inline-flex h-7 items-center rounded-full px-2.5 text-[13px] text-ink-2 hover:text-ink">
            Agents
          </Link>
          <HealthSummary />
          {pid && <DetailsMenu pid={pid} />}
          {!onStudioHome && (
            <Link
              to="/studio"
              className="sk-pill sk-pill-primary hidden h-8 items-center px-4 text-[12.5px] font-medium sm:inline-flex"
            >
              New project
            </Link>
          )}
          <ThemeToggle />
        </div>
      </header>
      <div className="min-h-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
