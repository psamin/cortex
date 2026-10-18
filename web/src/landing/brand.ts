/** Product brand. The single place the name and taglines live. */
export const BRAND = {
  name: "Cortex",
  tagline: "Let your agents train a robot",
  /** Page title: what it is, in the words people search with. */
  title: "Cortex: let your agents train a robot. MCP server for robot policies with LeRobot and MuJoCo",
  description:
    "Connect any agent to one MCP server and let it train your robot. It describes your arm, objects and layout; Cortex builds that environment in simulation, collects verified demonstrations, trains and tests a policy, then carries it to the physical arm to train further and deploy. A person approves every step that costs money or moves hardware.",
  /** Entity definition for search and answer engines: the exact names people and agents search for. */
  definition:
    "Cortex is an open-source MCP server that lets any AI agent train a robot policy for a custom environment and layout, in MuJoCo simulation and then on the physical arm, built on LeRobot, ACT and the SO-101.",
} as const;

/** Where the landing page's primary calls to action lead (the studio's chat-first entry). */
export const STUDIO_ENTRY = "/studio";
