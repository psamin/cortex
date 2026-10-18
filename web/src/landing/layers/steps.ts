/** The seven layers of the training loop. Copy is kept to what Cortex actually does today (MVP: physics sim). */
export type GlyphKind = "card" | "grid" | "loop" | "loss" | "ci" | "branch" | "chip";

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
    glyph: "card",
    readout: "MCP · UPDATE_INTAKE",
    tag: "Agent · MCP",
    body: "The agent tells Cortex its arm, cameras, objects, zones and task over MCP. Typed questions come back for whatever is still open.",
  },
  {
    n: "02",
    label: "Simulate",
    glyph: "grid",
    readout: "JAW · TORQUE · REACH MEASURED",
    tag: "Sim",
    body: "A MuJoCo scene is built from that description. Feasibility is measured from the model: jaw opening, holding torque, reach.",
  },
  {
    n: "03",
    label: "Demonstrate",
    glyph: "loop",
    readout: "HOME → PICK → PLACE → HOME",
    tag: "Sim · verified",
    body: "The agent drives the arm through validated tools: home, pick, place, home, then reset. Only simulator-verified successes become LeRobot episodes.",
  },
  {
    n: "04",
    label: "Train",
    glyph: "loss",
    readout: "ACT · GPU SPEND APPROVED",
    tag: "Human approval",
    body: "An ACT policy is trained on the demonstrations. A human approves the GPU spend before anything launches.",
  },
  {
    n: "05",
    label: "Evaluate",
    glyph: "ci",
    readout: "SEEDED BRANCHES · WILSON 95% CI",
    tag: "Sim · measured",
    body: "Seeded evaluation branches report success with Wilson confidence intervals. Measured, not claimed.",
  },
  {
    n: "06",
    label: "Improve",
    glyph: "branch",
    readout: "WEAKEST CONDITION → MORE DEMOS",
    tag: "Sim",
    body: "Cortex finds the weakest condition, then collects targeted demos or RL-distilled demos for it. The loop repeats.",
  },
  {
    n: "07",
    label: "Deploy",
    glyph: "chip",
    readout: "OPERATOR + E-STOP ON REAL ARM",
    tag: "Real · supervised",
    body: "You get the policy, a model card and the run command. A real arm runs it only with an operator present and an e‑stop in reach.",
  },
];
