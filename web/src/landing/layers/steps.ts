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
    body: "Any agent connects over MCP and describes the arm, cameras, objects, zones and layout: your environment, not a fixed demo scene. Typed questions come back for whatever is still open.",
  },
  {
    n: "02",
    label: "Simulate",
    glyph: "simulate",
    readout: "JAW · TORQUE · REACH MEASURED",
    tag: "Sim",
    body: "Cortex builds that exact layout as a MuJoCo scene and measures feasibility from the model: jaw opening, holding torque, reach.",
  },
  {
    n: "03",
    label: "Demonstrate",
    glyph: "demonstrate",
    readout: "HOME → PICK → PLACE → HOME",
    tag: "Sim · verified",
    body: "The agent drives the arm through validated tools: home, pick, place, home, then reset. Only simulator-verified successes become LeRobot episodes.",
  },
  {
    n: "04",
    label: "Train",
    glyph: "train",
    readout: "ACT · GPU SPEND APPROVED",
    tag: "Human approval",
    body: "An ACT policy is trained on the demonstrations. A human approves the GPU spend before anything launches.",
  },
  {
    n: "05",
    label: "Evaluate",
    glyph: "evaluate",
    readout: "SEEDED BRANCHES · WILSON 95% CI",
    tag: "Sim · measured",
    body: "Seeded evaluation branches report success with Wilson confidence intervals. Measured, not claimed.",
  },
  {
    n: "06",
    label: "Improve",
    glyph: "improve",
    readout: "WEAKEST CONDITION → MORE DEMOS",
    tag: "Sim",
    body: "Cortex finds the weakest measured condition and collects targeted demos for it; RL sandboxes are next. The loop repeats.",
  },
  {
    n: "07",
    label: "Deploy",
    glyph: "deploy",
    readout: "PHYSICAL ARM · OPERATOR + E-STOP",
    tag: "Real · supervised",
    body: "The policy moves to the physical arm, where it can keep training on real demonstrations and then run the task itself. Always with an operator present and an e‑stop in reach.",
  },
];
