/** Product brand. The single place the name and taglines live. */
export const BRAND = {
  name: "Cortex",
  tagline: "The robot-training MCP agents want",
  /** Page title: what it is, in the words people search with. */
  title: "Cortex: the MCP server that lets any agent train a physical robot, with no human in the training loop",
  description:
    "Agents can write software, but they can't teach a robot. Cortex gives them the tools. Any agent describes the arm and its workspace, then builds the simulation, collects its own demonstrations, trains, tests and fixes the policy, and takes it to the physical arm. No teleoperation, no hand-labelled data, no engineer babysitting runs. People only approve spend and hardware.",
  /** Entity definition for search and answer engines: the exact names people and agents search for. */
  definition:
    "Cortex is an open-source MCP server that lets any AI agent train a physical robot policy end to end, from a custom environment through simulation, demonstrations, training and evaluation to deployment, with no human in the training loop. Built on LeRobot, MuJoCo, ACT and the SO-101.",
} as const;

/** Where the landing page's primary calls to action lead (the studio's chat-first entry). */
export const STUDIO_ENTRY = "/studio";
