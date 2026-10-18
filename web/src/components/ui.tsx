import {
  forwardRef,
  useEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";
import { AlertTriangle, CircleDashed, PlugZap, X } from "lucide-react";
import { ApiError, API_BASE } from "../api/client";
import type { JobStatus } from "../api/types";
import { cn, errorMessage } from "../lib/format";

// ------------------------------------------------------------------ Button
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
type BtnSize = "sm" | "md";

const btnBase =
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded font-medium select-none " +
  "transition-[background-color,color,box-shadow,transform] duration-150 ease-out active:scale-[0.97] " +
  "disabled:opacity-40 disabled:pointer-events-none";
const btnVariants: Record<BtnVariant, string> = {
  primary: "bg-servo text-servo-ink hover:bg-servo/90 shadow-[inset_0_1px_0_rgba(255,255,255,.2)]",
  secondary: "bg-rail text-ink hover:bg-rail/70 shadow-[0_0_0_1px_var(--line)]",
  ghost: "text-ink-2 hover:text-ink hover:bg-rail",
  danger: "text-fault hover:bg-fault/10 shadow-[0_0_0_1px_rgb(var(--fault)/.35)]",
};
const btnSizes: Record<BtnSize, string> = { sm: "h-7 px-2.5 text-xs", md: "h-8 px-3 text-sm" };

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: BtnSize; loading?: boolean }
>(function Button({ variant = "secondary", size = "md", loading, className, children, disabled, ...rest }, ref) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={disabled || loading}
      className={cn(btnBase, btnVariants[variant], btnSizes[size], className)}
      {...rest}
    >
      {loading && <CircleDashed className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
});

// ------------------------------------------------------------------ Badge
export type Tone = "neutral" | "servo" | "pass" | "fault" | "caution" | "hypothesis" | "fallback";
const tones: Record<Tone, string> = {
  neutral: "text-ink-2 bg-rail shadow-[0_0_0_1px_var(--line)]",
  servo: "text-servo bg-servo/10 shadow-[0_0_0_1px_rgb(var(--servo)/.3)]",
  pass: "text-pass bg-pass/10 shadow-[0_0_0_1px_rgb(var(--pass)/.25)]",
  fault: "text-fault bg-fault/10 shadow-[0_0_0_1px_rgb(var(--fault)/.25)]",
  caution: "text-caution bg-caution/10 shadow-[0_0_0_1px_rgb(var(--caution)/.25)]",
  hypothesis: "text-ink-2 prov-hypothesis",
  fallback: "text-caution prov-fallback",
};

export function Badge({
  tone = "neutral",
  children,
  className,
  title,
  icon,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
  title?: string;
  icon?: ReactNode;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex h-5 items-center gap-1 whitespace-nowrap rounded-sm px-1.5 text-2xs font-medium",
        tones[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

/** The canonical "this came from a model, not a measurement" mark. */
export function HypothesisBadge({ label = "unverified hypothesis" }: { label?: string }) {
  return (
    <Badge tone="hypothesis" title="Model-generated. Not a measurement. Confirm or reject it below.">
      {label}
    </Badge>
  );
}

// ------------------------------------------------------------------ Panel
export function Panel({
  title,
  meta,
  actions,
  children,
  className,
  bodyClassName,
  id,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("rounded-md bg-plate shadow-[0_0_0_1px_var(--line)]", className)}>
      {(title || actions) && (
        <header className="flex min-h-10 items-center gap-3 border-b border-line-soft px-3.5 py-2">
          <div className="flex min-w-0 flex-1 items-baseline gap-2">
            {title && <h2 className="truncate text-sm font-semibold tracking-[-0.005em]">{title}</h2>}
            {meta && <span className="truncate text-xs text-ink-3">{meta}</span>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName ?? "p-3.5"}>{children}</div>
    </section>
  );
}

// ------------------------------------------------------------------ states
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded", className)} />;
}

export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-7 w-full" />
      ))}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-6 py-8 text-center", className)}>
      {icon && <div className="text-ink-4">{icon}</div>}
      <p className="text-sm font-medium text-ink-2">{title}</p>
      {children && <p className="max-w-sm text-xs text-ink-3">{children}</p>}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

/** Python exception text, tracebacks, stack frames: never a headline. */
const TECHNICAL = /Traceback|\b[A-Z][A-Za-z]*(Error|Exception)\b|File ".*", line \d+|\n\s+at /;

/**
 * Split an error into a readable headline and optional technical details.
 * `fallback` replaces a headline that is itself technical (raw exception text).
 */
export function splitError(error: unknown, detail?: string | null, fallback = "Something went wrong."): { text: string; detail?: string } {
  let text = typeof error === "string" ? error : errorMessage(error);
  let det = detail ?? (error instanceof ApiError ? error.technical : undefined) ?? undefined;
  if (!text || TECHNICAL.test(text)) {
    det = [text, det].filter(Boolean).join("\n\n") || undefined;
    text = fallback;
  }
  return { text, detail: det };
}

/** "Details" toggle for technical text. */
export function ErrorDetails({ detail }: { detail?: string }) {
  const [open, setOpen] = useState(false);
  if (!detail) return null;
  return (
    <>
      <button type="button" onClick={() => setOpen((o) => !o)} className="mt-0.5 text-xs text-ink-3 hover:text-ink-2">
        {open ? "Hide details" : "Details"}
      </button>
      {open && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-md bg-rail px-2 py-1.5 font-mono text-[11px] text-ink-3 scroll-thin">{detail}</pre>}
    </>
  );
}

/** One-line inline error: readable text, technical text behind "Details". */
export function InlineError({
  error,
  detail,
  fallback,
  prefix,
  className,
}: {
  error: unknown;
  detail?: string | null;
  fallback?: string;
  prefix?: string;
  className?: string;
}) {
  const e = splitError(error, detail, fallback);
  return (
    <div className={cn("text-xs", className)} role="alert">
      <p className="text-fault">
        {prefix}
        {e.text}
      </p>
      <ErrorDetails detail={e.detail} />
    </div>
  );
}

export function ErrorState({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const unreachable = error instanceof ApiError && error.unreachable;
  const split = splitError(error);
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2.5 rounded bg-fault/5 text-sm shadow-[0_0_0_1px_rgb(var(--fault)/.25)]",
        compact ? "p-2.5" : "p-3.5",
      )}
    >
      {unreachable ? (
        <PlugZap className="mt-0.5 h-4 w-4 shrink-0 text-fault" />
      ) : (
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-fault" />
      )}
      <div className="min-w-0 flex-1">
        <p className="font-medium text-ink">{unreachable ? "Backend unreachable" : "Request failed"}</p>
        <p className="mt-0.5 break-words text-xs text-ink-3">
          {unreachable ? (
            <>
              No response from <span className="font-mono">{API_BASE}</span>. Start the API server or set
              VITE_API_URL.
            </>
          ) : (
            split.text
          )}
        </p>
        {!unreachable && <ErrorDetails detail={split.detail} />}
      </div>
      {onRetry && (
        <Button size="sm" variant="ghost" onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

/** Wraps a query result into loading / error / content. */
export function QueryState<T>({
  q,
  children,
  skeleton,
}: {
  q: { data: T | undefined; isLoading: boolean; error: unknown; refetch: () => unknown };
  children: (data: T) => ReactNode;
  skeleton?: ReactNode;
}) {
  if (q.isLoading) return <>{skeleton ?? <SkeletonRows />}</>;
  if (q.error) return <ErrorState error={q.error} onRetry={() => q.refetch()} compact />;
  if (q.data === undefined) return null;
  return <>{children(q.data)}</>;
}

// ------------------------------------------------------------------ form
/** Base input styling; default h-8/w-full unless the caller passes its own h-/w- utility. */
export function inputClass(extra?: string) {
  const hasW = /(^|\s)w-/.test(extra ?? "");
  const hasH = /(^|\s)h-/.test(extra ?? "");
  return cn(hasH ? "" : "h-8", hasW ? "" : "w-full", inputCls, extra);
}
export const inputCls =
  "rounded bg-well px-2.5 text-sm text-ink placeholder:text-ink-4 shadow-[inset_0_0_0_1px_var(--line)] " +
  "focus:outline-none focus:shadow-[inset_0_0_0_1px_rgb(var(--servo)/.7)]";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...rest },
  ref,
) {
  return <input ref={ref} className={inputClass(className)} {...rest} />;
});

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={inputClass(cn("appearance-none pr-7 bg-no-repeat", className))} style={selectChevron} {...rest}>
      {children}
    </select>
  );
}
const selectChevron = {
  backgroundImage:
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath d='M2 4l3 3 3-3' stroke='%23888' fill='none' stroke-width='1.3'/%3E%3C/svg%3E\")",
  backgroundPosition: "right 8px center",
};

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-xs font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="block text-2xs text-ink-3">{hint}</span>}
    </label>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 py-1">
      <input type="checkbox" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span
        aria-hidden
        className={cn(
          "relative mt-0.5 h-4 w-7 shrink-0 rounded-full transition-colors duration-150 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-servo/70",
          checked ? "bg-servo" : "bg-rail shadow-[inset_0_0_0_1px_var(--line-strong)]",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-3 w-3 rounded-full bg-white shadow transition-transform duration-150 ease-out",
            checked ? "translate-x-3.5" : "translate-x-0.5",
          )}
        />
      </span>
      <span className="min-w-0">
        <span className="block text-sm text-ink">{label}</span>
        {hint && <span className="block text-2xs text-ink-3">{hint}</span>}
      </span>
    </label>
  );
}

// ------------------------------------------------------------------ segmented
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size = "md",
  ariaLabel,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode }[];
  size?: "sm" | "md";
  ariaLabel?: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded bg-well p-0.5 shadow-[inset_0_0_0_1px_var(--line)]">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-sm px-2.5 font-medium capitalize transition-colors duration-100",
            size === "sm" ? "h-6 text-2xs" : "h-7 text-xs",
            o.value === value ? "bg-rail text-ink shadow-[0_0_0_1px_var(--line)]" : "text-ink-3 hover:text-ink-2",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ dialog
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
  width = 520,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: number;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      style={{ width }}
      className="enter max-w-[calc(100vw-32px)] rounded-lg bg-plate p-0 text-ink shadow-[0_0_0_1px_var(--line-strong),0_24px_48px_-12px_rgba(0,0,0,.5)]"
    >
      {open && (
        <div>
          <header className="flex items-center justify-between border-b border-line-soft px-4 py-3">
            <h2 className="text-base font-semibold">{title}</h2>
            <button type="button" onClick={onClose} className="rounded p-1 text-ink-3 hover:bg-rail hover:text-ink" aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          </header>
          <div className="max-h-[70vh] overflow-y-auto px-4 py-4 scroll-thin">{children}</div>
          {footer && <footer className="flex justify-end gap-2 border-t border-line-soft px-4 py-3">{footer}</footer>}
        </div>
      )}
    </dialog>
  );
}

// ------------------------------------------------------------------ data bits
export function Id({ children, className }: { children: string; className?: string }) {
  return (
    <span className={cn("font-mono text-2xs text-ink-3", className)} title={children}>
      {children}
    </span>
  );
}

export function Meter({
  value,
  tone = "servo",
  className,
  label,
}: {
  value: number;
  tone?: "servo" | "pass" | "fault" | "caution" | "ink";
  className?: string;
  label?: string;
}) {
  const color = {
    servo: "bg-servo",
    pass: "bg-pass",
    fault: "bg-fault",
    caution: "bg-caution",
    ink: "bg-ink-3",
  }[tone];
  const v = Math.max(0, Math.min(1, value));
  return (
    <div
      role="meter"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={v}
      aria-label={label}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-rail", className)}
    >
      <div className={cn("h-full rounded-full transition-[width] duration-300 ease-out", color)} style={{ width: `${v * 100}%` }} />
    </div>
  );
}

const statusTone: Record<string, Tone> = {
  proposed: "neutral",
  approved: "servo",
  queued: "neutral",
  running: "servo",
  succeeded: "pass",
  failed: "fault",
  cancelled: "neutral",
  accepted: "pass",
  rejected: "fault",
  recorded: "neutral",
  dismissed: "neutral",
  done: "pass",
  hypothesis: "hypothesis",
  human_confirmed: "pass",
  human_rejected: "fault",
};

export function StatusBadge({ status }: { status: JobStatus | string }) {
  const live = status === "running" || status === "queued";
  return (
    <Badge tone={statusTone[status] ?? "neutral"}>
      {live && <span className={cn("h-1.5 w-1.5 rounded-full bg-current", status === "running" && "live-dot")} />}
      {status.replace(/_/g, " ")}
    </Badge>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-sm bg-rail px-1 font-mono text-2xs text-ink-3 shadow-[0_0_0_1px_var(--line)]">{children}</kbd>
  );
}

export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("text-2xs font-medium uppercase tracking-[0.06em] text-ink-3", className)}>{children}</div>;
}
