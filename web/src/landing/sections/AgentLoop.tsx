import { useState } from "react";
import { SectionHead } from "./SectionHead";

const NODES = [
  { k: "A", tag: "Agent", name: "Claude Code, or any MCP client", note: "Calls tools · asks you to approve" },
  { k: "B", tag: "Tools", name: "cortex.* tools", note: "create_policy_project · robot_move · propose_training_job" },
  { k: "C", tag: "Physics", name: "MuJoCo · SO-101", note: "EnvHub scene · pinned commit" },
  { k: "D", tag: "Data", name: "LeRobot dataset", note: "v3 · verified episodes only" },
  { k: "E", tag: "Policy", name: "ACT checkpoint", note: "Model card · run command" },
];

const CMD = "claude mcp add --transport http cortex http://localhost:3000/mcp";

function Arrow() {
  return (
    <svg aria-hidden className="h-3 w-full min-w-[28px] text-[#171717]" viewBox="0 0 40 12" preserveAspectRatio="none">
      <line x1="0" y1="6" x2="38" y2="6" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <path d="M33 2 L38.5 6 L33 10" fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** FIG.03: agents call tools; tools run physics; physics fills the dataset; the dataset trains the policy. */
export function AgentLoop() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(CMD);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked: the command stays selectable */
    }
  };

  return (
    <section id="loop" className="kn-gutter border-t border-[#E6E6E6] py-24 lg:py-32">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-end">
        <SectionHead fig="FIG.03" label="The agent loop" first="Your agent drives." second="Cortex keeps it honest.">
          Connect any MCP agent. It describes the arm and the task, calls Cortex tools and reads the results back. The
          physics, the datasets and the training run on the Cortex side.
        </SectionHead>

        <div className="min-w-0 border border-[#E6E6E6]">
          <div className="flex h-10 items-center justify-between border-b border-[#E6E6E6] pl-4 pr-2 kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585]">
            <span>Terminal · connect once</span>
            <button type="button" onClick={copy} className="rounded-full px-3 py-1 uppercase text-[#171717] transition-colors hover:bg-[#F5F5F5]">
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre className="whitespace-pre-wrap px-4 py-5 [overflow-wrap:anywhere] sm:overflow-x-auto sm:whitespace-pre sm:[overflow-wrap:normal] kn-mono text-[12.5px] leading-[1.7] text-[#171717]">
            <code>
              <span className="select-none text-[#BDBDBD]">$ </span>
              {CMD}
            </code>
          </pre>
          <div className="border-t border-[#E6E6E6] px-4 py-3 kn-mono text-[10px] uppercase tracking-[0.14em] text-[#858585]">
            Then ask{" "}
            <span className="ml-1 normal-case tracking-normal text-[#171717]">&ldquo;Teach the SO-101 to put the cube in the bowl.&rdquo;</span>
          </div>
        </div>
      </div>

      <div
        className="mt-20"
        role="img"
        aria-label="Agent, then cortex tools, then MuJoCo, then a LeRobot dataset, then an ACT policy. Evaluation feeds back into the tools."
      >
        <ol className="flex flex-col lg:grid lg:grid-cols-[minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)] lg:items-stretch">
          {NODES.map((n, i) => (
            <li key={n.k} className="contents">
              <div className={`border p-4 ${i === 1 ? "border-[#2F4F46]" : "border-[#E6E6E6]"}`}>
                <div className="flex items-center justify-between kn-mono text-[10px] uppercase tracking-[0.16em]">
                  <span className={i === 1 ? "text-[#2F4F46]" : "text-[#171717]"}>{n.k}</span>
                  <span className="text-[#858585]">{n.tag}</span>
                </div>
                <div className="mt-6 kn-sans text-[15px] font-medium text-[#171717]">{n.name}</div>
                <div className="mt-2 break-words kn-mono text-[10.5px] leading-[1.6] text-[#858585]">{n.note}</div>
              </div>
              {i < NODES.length - 1 && (
                <div className="flex h-8 items-center justify-center lg:h-auto" aria-hidden>
                  <div className="h-full w-px bg-[#171717] lg:hidden" />
                  <div className="hidden w-full px-1 lg:block">
                    <Arrow />
                  </div>
                </div>
              )}
            </li>
          ))}
        </ol>
        <div className="mt-6 hidden items-center gap-4 lg:flex" aria-hidden>
          <span className="h-px flex-1 border-t border-dashed border-[#BDBDBD]" />
          <span className="kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585]">
            E → B · evaluate, find the weakest condition, collect again
          </span>
          <span className="h-px flex-1 border-t border-dashed border-[#BDBDBD]" />
        </div>
      </div>
    </section>
  );
}
