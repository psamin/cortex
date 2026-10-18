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
import { CircleDashed, PlugZap, X } from "lucide-react";
import { ApiError, API_BASE } from "../api/client";
import type { JobStatus } from "../api/types";
import { cn, errorMessage } from "../lib/format";
import { IsoIcon, SketchScene, type IsoIconName } from "../sketch";

// ------------------------------------------------------------------ Button
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
type BtnSize = "sm" | "md";

// Kinetic pills: primary is a black pill, secondary a white pill with a hairline. Both sit on an
// isometric base plate that compresses on press (.sk-pill in sketch/sketch.css).
const btnBase =
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-full font-medium select-none " +
  "transition-colors duration-150 ease-out disabled:opacity-40 disabled:pointer-events-none";
const btnVariants: Record<BtnVariant, string> = {
  primary: "sk-pill sk-pill-primary",
  secondary: "sk-pill sk-pill-secondary",
  ghost: "text-ink-2 hover:text-ink hover:bg-rail",
  danger: "sk-pill sk-pill-danger",
};
const btnSizes: Record<BtnSize, string> = { sm: "h-7 px-3 text-xs", md: "h-8 px-4 text-[13px]" };

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
  neutral: "text-ink-2 bg-plate border border-line-strong",
  servo: "text-servo bg-plate border border-servo/40",
  pass: "text-pass bg-plate border border-pass/35",
  fault: "text-fault bg-plate border border-fault/35",
  caution: "text-caution bg-plate border border-caution/40",
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
        "inline-flex h-5 items-center gap-1 whitespace-nowrap rounded-sm px-1.5 font-mono text-[10px] font-medium uppercase tracking-[0.08em]",
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
  icon,
}: {
  /** Optional isometric glyph before the title (section-header style). */
  icon?: IsoIconName;
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("sk-depth relative rounded bg-plate border border-line-strong", className)}>
      {(title || actions) && (
        <header className="flex min-h-11 items-center gap-3 border-b border-line px-4 py-2">
          <div className="flex min-w-0 flex-1 items-center gap-2.5">
            {icon ? <IsoIcon name={icon} size={18} /> : title ? <span aria-hidden className="h-px w-6 shrink-0 bg-ink" /> : null}
            {title && <h2 className="truncate font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-ink">{title}</h2>}
            {meta && <span className="truncate text-xs text-ink-3">{meta}</span>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={bodyClassName ?? "p-4"}>{children}</div>
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
  icon: _icon,
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
      {/* The isometric sketch stands in for the old flat icon (`icon` is still accepted). */}
      <SketchScene width={200} className="mb-1" />
      <p className="kn-display text-[22px] text-ink">{title}</p>
      {children && <p className="max-w-sm text-xs leading-5 text-ink-3">{children}</p>}
      {action && <div className="mt-2">{action}</div>}
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
      <button type="button" onClick={() => setOpen((o) => !o)} className="mt-0.5 border-b border-line-strong text-xs text-ink-3 hover:border-ink hover:text-ink">
        {open ? "Hide details" : "Details"}
      </button>
      {open && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded-sm border border-line bg-well px-2 py-1.5 font-mono text-[11px] text-ink-3 scroll-thin">{detail}</pre>}
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
        "flex items-start gap-2.5 rounded border border-fault/30 bg-plate text-sm",
        compact ? "p-2.5" : "p-3.5",
      )}
    >
      {unreachable ? (
        <PlugZap className="mt-0.5 h-4 w-4 shrink-0 text-fault" />
      ) : (
        <IsoIcon name="warning" size={18} className="mt-px" />
      )}
      <div className="min-w-0 flex-1">
        <p className="font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-fault">{unreachable ? "Backend unreachable" : "Request failed"}</p>
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
  "rounded-sm border border-line-strong bg-plate px-2.5 text-sm text-ink placeholder:text-ink-4 " +
  "transition-colors focus:border-servo focus:outline-none";

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
      <span className="block font-mono text-[10.5px] font-medium uppercase tracking-[0.14em] text-ink-3">{label}</span>
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
          checked ? "bg-servo" : "bg-plate border border-line-strong",
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-3 w-3 rounded-full border border-black/15 bg-white transition-transform duration-150 ease-out",
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
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-full border border-line-strong bg-plate p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-full px-3 font-medium capitalize transition-colors duration-100",
            size === "sm" ? "h-6 text-2xs" : "h-7 text-xs",
            o.value === value ? "bg-ink text-bench" : "text-ink-3 hover:text-ink",
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
      className="enter max-w-[calc(100vw-32px)] rounded border border-line-strong bg-plate p-0 text-ink"
    >
      {open && (
        <div>
          <header className="flex items-center justify-between border-b border-line px-5 py-3.5">
            <h2 className="kn-display text-[24px]">{title}</h2>
            <button type="button" onClick={onClose} className="rounded p-1 text-ink-3 hover:bg-rail hover:text-ink" aria-label="Close">
              <X className="h-4 w-4" />
            </button>
          </header>
          <div className="max-h-[70vh] overflow-y-auto px-5 py-4 scroll-thin">{children}</div>
          {footer && <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">{footer}</footer>}
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
      className={cn("h-1 w-full overflow-hidden bg-[var(--line)]", className)}
    >
      <div className={cn("h-full transition-[width] duration-300 ease-out", color)} style={{ width: `${v * 100}%` }} />
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
    <kbd className="rounded-sm border border-line-strong bg-plate px-1 font-mono text-2xs text-ink-3">{children}</kbd>
  );
}

export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("kn-label", className)}>{children}</div>;
}

// ------------------------------------------------------------------ minimal copy
/** A small "i" that carries an explanation as a native tooltip (hover or focus). */
export function InfoTip({ children, className }: { children: string; className?: string }) {
  return (
    <span
      tabIndex={0}
      role="note"
      title={children}
      aria-label={children}
      className={cn(
        "inline-flex h-3.5 w-3.5 shrink-0 cursor-help items-center justify-center rounded-full border border-line-strong align-middle font-mono text-[9px] leading-none text-ink-3 hover:border-ink hover:text-ink focus:outline-none focus-visible:border-servo",
        className,
      )}
    >
      i
    </span>
  );
}

/** "Details" toggle for secondary explanations (same look as ErrorDetails). */
export function More({ children, label = "Details", className }: { children: ReactNode; label?: string; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={cn("text-xs", className)}>
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="border-b border-line-strong text-ink-3 hover:border-ink hover:text-ink">
        {open ? "Hide" : label}
      </button>
      {open && <div className="mt-1.5 space-y-1 leading-5 text-ink-3">{children}</div>}
    </div>
  );
}

/** Compact mono metadata row: `A · B · C`, empty parts dropped. */
export function Readout({ items, className }: { items: (ReactNode | null | undefined | false)[]; className?: string }) {
  const parts = items.filter((x) => x !== null && x !== undefined && x !== false && x !== "");
  if (!parts.length) return null;
  return (
    <div className={cn("flex flex-wrap items-center gap-x-1.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-3", className)}>
      {parts.map((p, i) => (
        <span key={i} className="contents">
          {i > 0 && <span aria-hidden className="text-ink-4">·</span>}
          <span>{p}</span>
        </span>
      ))}
    </div>
  );
}
