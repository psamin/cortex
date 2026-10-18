import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { SectionHead } from "../sections/SectionHead";
import SNIPPETS from "./snippets.json";
import { GlowEdge } from "../../components/ui/spotlight-card";

type Client = (typeof SNIPPETS.clients)[number];
type Call = (typeof SNIPPETS.session)[number] & { alt?: string };

const SESSION = SNIPPETS.session as Call[];

/** One-line JSON with a space after separators, so long args can wrap. */
const fmt = (v: unknown) => JSON.stringify(v).replace(/([,:])(?=["{[\d-])/g, "$1 ");
const sessionText = () => SESSION.map((c) => `${c.tool} ${fmt(c.args)}`).join("\n");

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked: the text stays selectable */
    }
  };
  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? "Copied" : label}
      className="rounded-full px-3 py-1 uppercase text-ink transition-colors hover:bg-rail focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink"
    >
      <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function CardBar({ left, children }: { left: string; children?: ReactNode }) {
  return (
    <div className="flex h-10 items-center justify-between gap-3 border-b border-line pl-4 pr-2 kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3">
      <span className="truncate">{left}</span>
      {children}
    </div>
  );
}

function Code({ c }: { c: Client }) {
  const shell = c.file === "Terminal";
  return (
    <pre className="whitespace-pre-wrap sm:min-h-[188px] px-4 py-5 [overflow-wrap:anywhere] sm:overflow-x-auto sm:whitespace-pre sm:[overflow-wrap:normal] kn-mono text-[12.5px] leading-[1.7] text-ink">
      <code>
        {c.code.split("\n").map((line, i) => (
          <span key={i} className={line.startsWith("#") ? "text-ink-3" : undefined}>
            {shell && line && !line.startsWith("#") && !line.startsWith(" ") && <span className="select-none text-ink-4">$ </span>}
            {line}
            {"\n"}
          </span>
        ))}
      </code>
    </pre>
  );
}

/** Tabbed config card: one tab per client, arrow keys move between tabs (WAI-ARIA tabs pattern). */
function ClientTabs() {
  const [active, setActive] = useState(0);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const base = useId();
  const n = SNIPPETS.clients.length;
  const onKey = (e: KeyboardEvent) => {
    const next = e.key === "ArrowRight" ? (active + 1) % n : e.key === "ArrowLeft" ? (active - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : null;
    if (next === null) return;
    e.preventDefault();
    setActive(next);
    tabs.current[next]?.focus();
  };
  const c = SNIPPETS.clients[active];

  return (
    <div className="min-w-0">
      <div role="tablist" aria-label="MCP client" className="-mb-px flex overflow-x-auto" onKeyDown={onKey}>
        {SNIPPETS.clients.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => (tabs.current[i] = el)}
            role="tab"
            id={`${base}-tab-${t.id}`}
            aria-selected={i === active}
            aria-controls={`${base}-panel`}
            tabIndex={i === active ? 0 : -1}
            onClick={() => setActive(i)}
            className={`shrink-0 whitespace-nowrap border-x border-t px-3.5 py-2.5 kn-mono text-[10.5px] uppercase tracking-[0.14em] transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-4 focus-visible:outline-ink ${
              i === active ? "border-line bg-bench text-ink" : "border-transparent text-ink-3 hover:text-ink"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-panel`} aria-labelledby={`${base}-tab-${c.id}`} className="relative border border-line">
        <GlowEdge />
        <CardBar left={c.file}>
          <CopyButton text={c.code} label={`Copy ${c.label} config`} />
        </CardBar>
        <Code c={c} />
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-4 py-3 kn-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">
          <span>{c.id === "rest" ? "Same tools and audit log, without MCP" : `Streamable HTTP · ${SNIPPETS.mcp_url.replace("http://", "")}`}</span>
          {c.doc && (
            <a href={c.doc} target="_blank" rel="noreferrer" className="text-ink hover:underline">
              {c.label} docs ↗
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

function Session() {
  return (
    <div className="relative mt-16 border border-line lg:mt-20">
      <GlowEdge />
      <CardBar left={`A session · ${SESSION.length} tool calls`}>
        <CopyButton text={sessionText()} label="Copy the session's tool calls" />
      </CardBar>
      <ol>
        {SESSION.map((c, i) => (
          <li
            key={c.tool}
            className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-3 gap-y-1.5 border-b border-line-soft px-4 py-4 last:border-b-0 lg:grid-cols-[2.5rem_minmax(0,1.35fr)_minmax(0,1fr)] lg:gap-x-8"
          >
            <span className="kn-mono text-[10.5px] leading-[1.9] tracking-[0.12em] text-ink-3">{String(i + 1).padStart(2, "0")}</span>
            <div className="min-w-0">
              <code className="kn-mono text-[12.5px] font-medium text-ink">{c.tool}</code>
              <code className="mt-1 block break-words kn-mono text-[11.5px] leading-[1.6] text-ink-2">{fmt(c.args)}</code>
              {c.alt && <p className="mt-2 kn-mono text-[10.5px] leading-[1.6] text-ink-3">{c.alt}</p>}
            </div>
            <p className="col-start-2 kn-sans text-[13.5px] leading-[1.55] text-ink-2 lg:col-start-3 lg:pt-px">
              <span aria-hidden className="mr-2 text-ink-4">→</span>
              {c.returns}
              {c.tool === "cortex.propose_training_job" && (
                <span className="mt-2 flex">
                  <span className="inline-flex h-5 items-center rounded-full border border-servo px-2 kn-mono text-[9.5px] uppercase tracking-[0.14em] text-servo">
                    Human approval
                  </span>
                </span>
              )}
            </p>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** FIG.03: how an agent connects, and what it calls once it has. */
export function ConnectSection() {
  return (
    <section id="connect" aria-labelledby="connect-title" className="kn-gutter border-t border-line py-24 lg:py-32">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-end">
        <SectionHead titleId="connect-title" fig="FIG.03" label="Connect your agent" first="Add one MCP server." second="Then ask for a task.">
          Cortex runs an MCP server on <code className="kn-mono text-[13px] text-ink">localhost:3000/mcp</code>. Add it to
          your agent once. Every tool is also on REST at <code className="kn-mono text-[13px] text-ink">/api/tools</code>.
        </SectionHead>
        <ClientTabs />
      </div>
      <Session />
    </section>
  );
}
