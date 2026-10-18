/** Product brand. The single place the name and taglines live. */
export const BRAND = {
  name: "Cortex",
  tagline: "The robot-training MCP agents want",
  /** Page title: what it is, in the words people search with. */
  title: "Cortex: the MCP server that lets any agent train a physical robot, with no human in the training loop",
  description:
    "Agents can code, but they can't train a robot. Cortex gives them the full functionality to train a robot on any skill. An agent can describe the arm and its environment, build the simulation, collect its own demonstrations, train, test, fix the policy, and deploy on the physical robot. No manual teleoperation, no hand-labelled data, no human babysitting runs. The next step in agent interactivity.",
  /** Entity definition for search and answer engines: the exact names people and agents search for. */
  definition:
    "Cortex is an open-source MCP server that lets any AI agent train a physical robot policy end to end, from a custom environment through simulation, demonstrations, training and evaluation to deployment, with no human in the training loop. Built on LeRobot, MuJoCo, ACT and the SO-101.",
} as const;

/** Where the landing page's primary calls to action lead (the studio's chat-first entry). */
export const STUDIO_ENTRY = "/studio";
