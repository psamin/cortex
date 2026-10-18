// Exercise the Cortex MCP server's native MCP features with the official TypeScript SDK client.
//   node scripts/native-demo.ts <tools|resources|progress|intake|listen> [--legacy] [--project prj_...]
// MCP_URL defaults to http://localhost:3000/mcp. --legacy speaks the 2025 protocol (initialize handshake,
// what most shipping clients do); the default negotiates 2026-07-28 (per-request envelope, input_required).
// The elicitation handler plays the human: it prints the form and accepts every default
// (--first: pick the first option of each select instead; --decline: decline the form).
// progress: --tool <name> --args '<json>' (default record_demonstrations, --count N); --no-fallback skips the
// second, progress-less call.
// Needs @modelcontextprotocol/client, which mcp-use installs.
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";

const argv = process.argv.slice(2);
const cmd = argv[0] ?? "tools";
const legacy = argv.includes("--legacy");
const flag = (name: string) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : undefined);
const base = process.env.MCP_URL ?? "http://localhost:3000/mcp";
const t0 = Date.now();
const log = (...a: unknown[]) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);

async function connect(query = "") {
  const client = new Client(
    { name: "cortex-native-demo", version: "0.1.0" },
    {
      capabilities: { elicitation: { form: {} } },
      ...(legacy ? {} : { versionNegotiation: { mode: { pin: "2026-07-28" } } }),
    },
  );
  client.setRequestHandler("elicitation/create", async (req) => {
    const p = req.params as { message: string; requestedSchema?: { properties?: Record<string, { default?: unknown; type?: string; enum?: unknown[]; items?: { enum?: unknown[] } }> } };
    log("ELICITATION:", p.message);
    const content: Record<string, unknown> = {};
    for (const [k, s] of Object.entries(p.requestedSchema?.properties ?? {})) {
      log(`  field ${k}: ${JSON.stringify(s)}`);
      const first = s.enum ? s.enum[0] : s.items?.enum ? [s.items.enum[0]] : undefined;
      const v = argv.includes("--first") && first !== undefined ? first : (s.default ?? first);
      if (v !== undefined) content[k] = v;
    }
    if (argv.includes("--decline")) {
      log("  human declines the form");
      return { action: "decline" } as never;
    }
    log("  human answers:", JSON.stringify(content));
    return { action: "accept", content } as never;
  });
  await client.connect(new StreamableHTTPClientTransport(new URL(base + query)));
  log(`connected (${legacy ? "2025 legacy" : "2026-07-28"} era, negotiated ${client.getNegotiatedProtocolVersion()})`);
  return client;
}

const text = (r: { content?: unknown[] }) => (r.content ?? []).map((c) => (c as { text?: string }).text ?? "").join("\n");

async function tools() {
  for (const q of ["", "?toolset=core", ...(flag("--project") ? [`?project=${flag("--project")}`] : [])]) {
    const c = await connect(q);
    const { tools } = await c.listTools();
    log(`tools/list ${q || "(plain)"}: ${tools.length} tools`);
    for (const t of tools) log(`  ${t.name.padEnd(40)} ${JSON.stringify(t._meta ?? {})}`);
    await c.close();
  }
}

async function resources() {
  const c = await connect();
  log("resources/list:", JSON.stringify((await c.listResources()).resources.map((r) => r.uri)));
  log("resources/templates/list:", JSON.stringify((await c.listResourceTemplates()).resourceTemplates.map((r) => r.uriTemplate)));
  const ref = { type: "ref/resource" as const, uri: "cortex://projects/{project_id}" };
  const projects = await c.complete({ ref, argument: { name: "project_id", value: "prj_" } });
  log("complete project_id 'prj_':", JSON.stringify(projects.completion));
  const project = flag("--project") ?? projects.completion.values[0];
  const lref = { type: "ref/resource" as const, uri: "cortex://projects/{project_id}/datasets/{version}/lineage" };
  const versions = await c.complete({ ref: lref, argument: { name: "version", value: "" }, context: { arguments: { project_id: project } } });
  log("complete version (project_id given):", JSON.stringify(versions.completion));
  const eref = { type: "ref/resource" as const, uri: "cortex://projects/{project_id}/evaluations/{evaluation_id}" };
  const evals = await c.complete({ ref: eref, argument: { name: "evaluation_id", value: "ev" }, context: { arguments: { project_id: project } } });
  log("complete evaluation_id 'ev':", JSON.stringify(evals.completion));
  const uris = [
    "cortex://projects",
    `cortex://projects/${project}`,
    `cortex://projects/${project}/episodes`,
    ...(versions.completion.values[0] ? [`cortex://projects/${project}/datasets/${versions.completion.values[0]}/lineage`] : []),
    ...(evals.completion.values[0] ? [`cortex://projects/${project}/evaluations/${evals.completion.values[0]}`] : []),
    `cortex://projects/${project}/policy/model-card`,
  ];
  for (const uri of uris) {
    const r = await c.readResource({ uri });
    const body = (r.contents[0] as { text?: string; mimeType?: string });
    log(`resources/read ${uri} [${body.mimeType}] ${body.text?.length} chars:\n${body.text?.slice(0, 900)}\n`);
  }
  await c.close();
}

async function progress() {
  const c = await connect();
  const project = flag("--project");
  if (!project) throw new Error("--project prj_... required");
  const tool = flag("--tool") ?? "record_demonstrations";
  const args = { project_id: project, ...(flag("--args") ? JSON.parse(flag("--args")!) : { conditions: [{ count: Number(flag("--count") ?? 1) }] }) };
  log(`tools/call cortex.${tool} ${JSON.stringify(args)} with a progressToken (timeout 60 s, reset on progress)`);
  const r = await c.callTool(
    { name: `cortex.${tool}`, arguments: args },
    { onprogress: (p) => log(`  progress ${p.progress}/${p.total} ${p.message ?? ""}`), timeout: 60_000, resetTimeoutOnProgress: true, maxTotalTimeout: 1_800_000 },
  );
  log("RESULT:\n" + text(r));
  log("structuredContent.followed:", JSON.stringify((r.structuredContent as Record<string, unknown>)?.followed));
  if (argv.includes("--no-fallback")) return void (await c.close());
  log(`tools/call cortex.${tool} WITHOUT a progressToken (old behaviour)`);
  const r2 = await c.callTool({ name: `cortex.${tool}`, arguments: { ...args, wait_s: 2 } });
  log("RESULT:\n" + text(r2));
  await c.close();
}

async function intake() {
  const c = await connect();
  log("tools/call cortex.start_intake");
  const r = await c.callTool({ name: "cortex.start_intake", arguments: { prompt: flag("--prompt") ?? "pick up the water bottle and put it on the coaster" } });
  log("RESULT:\n" + text(r));
  const s = r.structuredContent as Record<string, unknown> | undefined;
  log("complete:", s?.complete, "open questions:", JSON.stringify(((s?.questions as { path: string; required?: boolean }[]) ?? []).map((q) => `${q.path}${q.required === false ? "?" : ""}`)));
  await c.close();
}

async function listen() {
  const c = await connect();
  const project = flag("--project");
  if (!project) throw new Error("--project prj_... required");
  const uri = `cortex://projects/${project}`;
  c.setNotificationHandler("notifications/resources/updated", (n) => log("NOTIFICATION resources/updated", JSON.stringify(n.params)));
  c.setNotificationHandler("notifications/resources/list_changed", () => log("NOTIFICATION resources/list_changed"));
  const sub = await c.listen({ resourceSubscriptions: [uri], resourcesListChanged: true } as never);
  log("subscriptions/listen open for", uri);
  const r = await c.callTool({ name: "cortex.recommend_demonstrations", arguments: { project_id: project } });
  log("tools/call recommend_demonstrations:", text(r).slice(0, 200));
  await new Promise((res) => setTimeout(res, 3000));
  await (sub as { close?: () => Promise<void> }).close?.();
  await c.close();
}

const run = { tools, resources, progress, intake, listen }[cmd];
if (!run) throw new Error(`unknown command ${cmd}`);
await run();
