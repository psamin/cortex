/** The seven layers of the training loop. Copy is kept to what Cortex actually does today (MVP: physics sim). */
export type GlyphKind = "describe" | "simulate" | "demonstrate" | "train" | "evaluate" | "improve" | "deploy";

export interface LayerStep {
  n: string;
  label: string;
  glyph: GlyphKind;
  /** Mono readout on the callout leader. */
  readout: string;
  /** Provenance tag shown with the explanation. */
  tag: string;
  body: string;
}

export const STEPS: LayerStep[] = [
  {
    n: "01",
    label: "Describe",
    glyph: "describe",
    readout: "MCP · UPDATE_INTAKE",
    tag: "Agent · MCP",
    body: "The agent tells Cortex the arm, cameras, objects and layout in a chat over MCP: any environment, no CAD or config files. Cortex asks typed questions about whatever is still open.",
  },
  {
    n: "02",
    label: "Simulate",
    glyph: "simulate",
    readout: "JAW · TORQUE · REACH MEASURED",
    tag: "Sim",
    body: "Cortex builds that exact layout as a MuJoCo scene and measures whether the arm can do the job: jaw opening, holding torque, reach.",
  },
  {
    n: "03",
    label: "Demonstrate",
    glyph: "demonstrate",
    readout: "HOME → PICK → PLACE → HOME",
    tag: "Sim · verified",
    body: "The agent drives the arm itself: home, pick, place, home, then it resets and goes again. No teleoperation. Only simulator-verified successes become LeRobot episodes.",
  },
  {
    n: "04",
    label: "Train",
    glyph: "train",
    readout: "ACT · GPU SPEND APPROVED",
    tag: "Human approval",
    body: "An ACT policy trains on the agent's own demonstrations. The one human step: approving the GPU spend.",
  },
  {
    n: "05",
    label: "Evaluate",
    glyph: "evaluate",
    readout: "SEEDED BRANCHES · WILSON 95% CI",
    tag: "Sim · measured",
    body: "Seeded test runs measure success with confidence intervals. The agent reads the numbers, not a person.",
  },
  {
    n: "06",
    label: "Improve",
    glyph: "improve",
    readout: "WEAKEST CONDITION → MORE DEMOS",
    tag: "Sim",
    body: "The agent finds the weakest condition, collects more demos for it and retrains. The loop runs itself until the numbers hold.",
  },
  {
    n: "07",
    label: "Deploy",
    glyph: "deploy",
    readout: "PRODUCTION LINE · OPERATOR + E-STOP",
    tag: "Real · supervised",
    body: "The policy goes to production: physical arms on the line run the task. An operator and an e‑stop stay on the floor.",
  },
];
