import { SectionHead } from "./SectionHead";

const STACK = [
  {
    name: "LeRobot",
    body: "Every dataset is a LeRobotDataset v3 with a lineage file. Training runs lerobot-train.",
    tag: "Real datasets",
  },
  {
    name: "MuJoCo · EnvHub SO-101",
    body: "Demos and evaluations are rollouts in lerobot/so101-tabletop-mujoco, pinned to one commit. Never replayed or mocked.",
    tag: "Sim",
  },
  {
    name: "ACT",
    body: "Action Chunking Transformer policies, trained on a GPU cluster. No result is reported without a checkpoint on disk.",
    tag: "Measured",
  },
  {
    name: "MCP",
    body: "One tool registry served over MCP and REST, so any MCP client can run the loop end to end.",
    tag: "Agent interface",
  },
  {
    name: "InsForge",
    body: "Postgres, file storage and the model gateway. Cortex falls back to SQLite and local disk when it is not configured.",
    tag: "Infra · fallback",
  },
];

const COLUMNS: { title: string; solid: boolean; items: string[] }[] = [
  {
    title: "What is measured",
    solid: true,
    items: [
      "Success comes from the simulator's verdict, never from the model's claim.",
      "Success rates carry Wilson 95% intervals over seeded branches.",
      "Training results exist only with a checkpoint on disk.",
    ],
  },
  {
    title: "What is labelled",
    solid: false,
    items: [
      "Every episode names its source: the agent or the scripted IK oracle.",
      "Sim and real are marked on every run, dataset and policy.",
      "Values Cortex assumed are kept apart from values you gave.",
    ],
  },
];

/** the stack, and the line between measured and labelled. */
export function BuiltOn() {
  return (
    <section id="stack" aria-labelledby="stack-title" className="kn-gutter border-t border-line py-24 lg:py-32">
      <SectionHead titleId="stack-title" label="Built on" first="Open tools underneath." second="Nothing hidden on top." />

      <dl className="mt-14 border-t border-ink">
        {STACK.map((s) => (
          <div
            key={s.name}
            className="grid gap-2 border-b border-line py-5 md:grid-cols-[minmax(0,16rem)_minmax(0,1fr)_9rem] md:items-baseline md:gap-8"
          >
            <dt className="kn-display text-[22px] text-ink">{s.name}</dt>
            <dd className="kn-sans text-[14.5px] leading-[1.6] text-ink-2">{s.body}</dd>
            <dd className="kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3 md:text-right">{s.tag}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-16 grid gap-10 md:grid-cols-2">
        {COLUMNS.map((c) => (
          <div key={c.title}>
            <div className="kn-label kn-label-rule">
              <span>{c.title}</span>
            </div>
            <ul className="mt-5 space-y-3">
              {c.items.map((t) => (
                <li key={t} className="flex gap-3 kn-sans text-[14.5px] leading-[1.6] text-ink-2">
                  <span
                    aria-hidden
                    className={`mt-[8px] h-[7px] w-[7px] shrink-0 rounded-full border ${
                      c.solid ? "border-ink bg-ink" : "border-dashed border-ink-3"
                    }`}
                  />
                  {t}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
