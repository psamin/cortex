import { SectionHead } from "./SectionHead";

const RULES = [
  {
    k: "S.01",
    title: "Agents never send motor commands",
    body: "Agents pick targets through validated tools. A local IK controller executes them, and the simulator decides whether it worked.",
    readout: "Agent → tool → IK → sim",
  },
  {
    k: "S.02",
    title: "A person approves training",
    body: "Training jobs wait as proposals until someone approves the GPU spend in the studio. Agents without the operator token are told to ask.",
    readout: "Proposed → approved → queued",
  },
  {
    k: "S.03",
    title: "Real hardware is supervised",
    body: "The SO-101 bridge runs locally behind a safety controller: joint, velocity and workspace limits, a watchdog, an e‑stop and an operator gate. The API and MCP cannot reach it.",
    readout: "Local only · operator + e-stop",
  },
];

/** approvals and the hardware boundary. */
export function Safety() {
  return (
    <section id="safety" aria-labelledby="safety-title" className="kn-gutter border-t border-line py-24 lg:py-32">
      <div className="flex flex-wrap items-end justify-between gap-10">
        <SectionHead titleId="safety-title" label="Safety and approvals" first="Agents propose." second="People approve." />
        <EStop />
      </div>
      <ol className="mt-14 grid border-t border-line md:grid-cols-3">
        {RULES.map((r, i) => (
          <li key={r.k} className={`py-8 md:px-8 ${i ? "border-t border-line md:border-l md:border-t-0" : "md:pl-0"}`}>
            <div className="kn-mono text-[10px] uppercase tracking-[0.16em] text-ink">{r.k}</div>
            <h3 className="mt-5 kn-sans text-[17px] font-medium text-ink">{r.title}</h3>
            <p className="mt-3 kn-sans text-[14.5px] leading-[1.6] text-ink-2">{r.body}</p>
            <div className="mt-6 kn-mono text-[10px] uppercase tracking-[0.14em] text-ink-3">{r.readout}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** A small drafted e-stop: dashed reach circle, solid button, mono callout. */
function EStop() {
  return (
    <svg viewBox="0 0 270 120" className="hidden h-[120px] w-[270px] md:block" aria-hidden fontFamily='"IBM Plex Mono", monospace'>
      <circle cx="60" cy="60" r="52" fill="none" stroke="var(--kn-grey)" strokeWidth="0.8" strokeDasharray="3 4" />
      <line x1="2" y1="60" x2="118" y2="60" stroke="var(--kn-faint)" strokeWidth="0.6" strokeDasharray="8 3 2 3" />
      <line x1="60" y1="2" x2="60" y2="118" stroke="var(--kn-faint)" strokeWidth="0.6" strokeDasharray="8 3 2 3" />
      <circle cx="60" cy="60" r="30" fill="var(--kn-paper)" stroke="var(--kn-ink)" strokeWidth="1" />
      <circle cx="60" cy="60" r="20" fill="var(--kn-accent)" />
      <path d="M81 39 L104 22 L266 22" fill="none" stroke="var(--kn-ink)" strokeWidth="0.8" />
      <circle cx="81" cy="39" r="2.6" fill="var(--kn-paper)" stroke="var(--kn-ink)" strokeWidth="0.9" />
      <text x="112" y="14" fontSize="10" letterSpacing="1.2" fill="var(--kn-ink)">
        E-STOP / OPERATOR
      </text>
      <text x="112" y="40" fontSize="9" letterSpacing="0.8" fill="var(--kn-grey)">
        REQUIRED ON A REAL ARM
      </text>
    </svg>
  );
}
