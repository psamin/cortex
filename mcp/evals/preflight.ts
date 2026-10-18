// Local publish preflight (#123): checks the live server's metadata the way Manufact's
// Publish > Checklist does for its "metadata quality" and "tool schemas" categories, so
// problems show up before a paid checklist run. It is not the Manufact checklist.
//
//   node evals/preflight.ts [--mcp-url http://localhost:3000/mcp]

const urlArg = process.argv.indexOf("--mcp-url");
const MCP_URL = urlArg > 0 ? process.argv[urlArg + 1] : (process.env.CORTEX_EVAL_MCP_URL ?? "http://localhost:3000/mcp");

let sessionId: string | undefined;
async function rpc(id: number, method: string, params: Record<string, unknown> = {}): Promise<any> {
  const headers: Record<string, string> = { "Content-Type": "application/json", Accept: "application/json, text/event-stream" };
  if (sessionId) headers["mcp-session-id"] = sessionId;
  const r = await fetch(MCP_URL, { method: "POST", headers, body: JSON.stringify({ jsonrpc: "2.0", id, method, params }) });
  sessionId ??= r.headers.get("mcp-session-id") ?? undefined;
  const raw = await r.text();
  const body = raw.startsWith("{") ? raw : raw.split("\n").find((l) => l.startsWith("data: "))?.slice(6) ?? "{}";
  const msg = JSON.parse(body);
  if (msg.error) throw new Error(`${method}: ${JSON.stringify(msg.error)}`);
  return msg.result;
}

const problems: string[] = [];
const warnings: string[] = [];
const need = (ok: boolean, what: string) => ok || problems.push(what);
const want = (ok: boolean, what: string) => ok || warnings.push(what);

const init = await rpc(1, "initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "cortex-preflight", version: "1" } });
const info = init.serverInfo ?? {};
need(!!info.name && !!info.version, "serverInfo has name and version");
need(!!info.title, "serverInfo.title is set");
need((info.description ?? "").length >= 40, "serverInfo.description is a real sentence");
need(Array.isArray(info.icons) && info.icons.length > 0, "serverInfo.icons is set");
want((info.icons ?? []).some((i: any) => i.mimeType === "image/png"), "a PNG icon is offered (some directories reject SVG)");
need(/^https:\/\//.test(info.websiteUrl ?? ""), "serverInfo.websiteUrl is an https URL");
need((init.instructions ?? "").length > 0, "server instructions are set");
for (const icon of info.icons ?? []) {
  const r = await fetch(icon.src).catch(() => undefined);
  need(!!r && r.ok, `icon ${icon.src} is reachable`);
}

const { tools } = await rpc(2, "tools/list");
const HINTS = ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"];
for (const t of tools) {
  const a = t.annotations ?? {};
  need(!!t.title || !!a.title, `${t.name}: has a title`);
  need((t.description ?? "").length >= 40, `${t.name}: description is at least a sentence`);
  for (const h of HINTS) need(typeof a[h] === "boolean", `${t.name}: annotations.${h} is set explicitly`);
  need(!(a.readOnlyHint && a.destructiveHint), `${t.name}: not both readOnly and destructive`);
  need(/^[a-zA-Z0-9_.-]{1,64}$/.test(t.name), `${t.name}: name uses [a-zA-Z0-9_.-] and is <= 64 chars`);
  const props = t.inputSchema?.properties ?? {};
  for (const [k, v] of Object.entries<any>(props)) want(!!v.description || !!v.enum, `${t.name}.${k}: parameter has a description`);
}
const { prompts } = await rpc(3, "prompts/list");
for (const p of prompts) need(!!p.description, `prompt ${p.name}: has a description`);

console.log(`${MCP_URL}: ${tools.length} tools, ${prompts.length} prompts`);
for (const w of warnings) console.log(`WARN  ${w}`);
for (const p of problems) console.log(`FAIL  ${p}`);
console.log(problems.length ? `${problems.length} problems` : `preflight passed (${warnings.length} warnings)`);
process.exitCode = problems.length ? 1 : 0;
