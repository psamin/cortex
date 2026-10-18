/** Product brand. The single place the name and taglines live. */
export const BRAND = {
  name: "Cortex",
  tagline: "Build intelligence for the physical world",
  /** Page title: what it is, in the words people search with. */
  title: "Cortex: MCP server for training robot policies with LeRobot and MuJoCo",
  description:
    "Your coding agent describes the robot and the task. Cortex builds the simulation, collects verified demonstrations, trains a policy and measures it, with a human approving every step that costs money or moves hardware.",
  /** Entity definition for search and answer engines: the exact names people and agents search for. */
  definition:
    "Cortex is an open-source MCP server and studio for training robot policies with an AI agent, built on LeRobot, MuJoCo, ACT and the SO-101 arm.",
} as const;

/** Where the landing page's primary calls to action lead (the studio's chat-first entry). */
export const STUDIO_ENTRY = "/studio";
