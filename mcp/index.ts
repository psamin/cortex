import { MCPServer } from "mcp-use";
import { registerPrompts } from "./src/prompts.js";
import { registerTools } from "./src/tools.js";

// Thin MCP layer over the Cortex backend tool registry: every cortex.<x> tool
// forwards to POST {CORTEX_API_URL}/tools/<x>. Structured commands and
// references only; media is returned as storage keys, never streamed.
const server = new MCPServer({
  name: "cortex",
  title: "Cortex",
  version: "0.1.0",
  description:
    "Cortex robot-learning studio: create projects, record simulated demonstrations, analyze failures, build datasets, " +
    "propose training (human-approved) and evaluate policies in simulation.",
});

registerTools(server);
registerPrompts(server);

export default server;
