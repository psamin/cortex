import { SectionHead } from "./SectionHead";
import { GlowEdge } from "../../components/ui/spotlight-card";

const NODES = [
  { k: "A", tag: "Agent", name: "Claude Code, or any MCP client", note: "Calls tools · asks you to approve" },
  { k: "B", tag: "Tools", name: "cortex.* tools", note: "create_policy_project · robot_move · propose_training_job" },
  { k: "C", tag: "Physics", name: "MuJoCo · SO-101", note: "EnvHub scene · pinned commit" },
  { k: "D", tag: "Data", name: "LeRobot dataset", note: "v3 · verified episodes only" },
  { k: "E", tag: "Policy", name: "ACT checkpoint", note: "Model card · run command" },
];

function Arrow() {
  return (
    <svg aria-hidden className="h-3 w-full min-w-[28px] text-ink" viewBox="0 0 40 12" preserveAspectRatio="none">
      <line x1="0" y1="6" x2="38" y2="6" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <path d="M33 2 L38.5 6 L33 10" fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** agents call tools; tools run physics; physics fills the dataset; the dataset trains the policy. */
export function AgentLoop() {
  return (
    <section id="loop" aria-labelledby="loop-title" className="kn-gutter border-t border-line py-24 lg:py-32">
      <SectionHead titleId="loop-title" label="The agent loop" first="The agent runs the loop." second="Cortex keeps it honest.">
        The agent describes the arm and the task, calls Cortex tools and reads the results back. The
        physics, the datasets and the training run on the Cortex side.
      </SectionHead>

      <div
        className="mt-20"
        role="img"
        aria-label="Agent, then cortex tools, then MuJoCo, then a LeRobot dataset, then an ACT policy. Evaluation feeds back into the tools."
      >
        <ol className="flex flex-col lg:grid lg:grid-cols-[minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)_40px_minmax(0,1fr)] lg:items-stretch">
          {NODES.map((n, i) => (
            <li key={n.k} className="contents">
              <div className={`relative border p-4 ${i === 1 ? "border-servo" : "border-line"}`}>
                <GlowEdge />
                <div className="flex items-center justify-between kn-mono text-[10px] uppercase tracking-[0.16em]">
                  <span className={i === 1 ? "text-servo" : "text-ink"}>{n.k}</span>
                  <span className="text-ink-3">{n.tag}</span>
                </div>
                <div className="mt-6 kn-sans text-[15px] font-medium text-ink">{n.name}</div>
                <div className="mt-2 break-words kn-mono text-[10.5px] leading-[1.6] text-ink-3">{n.note}</div>
              </div>
              {i < NODES.length - 1 && (
                <div className="flex h-8 items-center justify-center lg:h-auto" aria-hidden>
                  <div className="h-full w-px bg-ink lg:hidden" />
                  <div className="hidden w-full px-1 lg:block">
                    <Arrow />
                  </div>
                </div>
              )}
            </li>
          ))}
        </ol>
        <div className="mt-6 hidden items-center gap-4 lg:flex" aria-hidden>
          <span className="h-px flex-1 border-t border-dashed border-ink-4" />
          <span className="kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3">
            E → B · evaluate, find the weakest condition, collect again
          </span>
          <span className="h-px flex-1 border-t border-dashed border-ink-4" />
        </div>
      </div>
    </section>
  );
}
