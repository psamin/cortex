/**
 * Build-time HTML for agents and crawlers that read index.html without running JavaScript.
 * vite.config.ts injects it, so it is generated from the same sources as the React landing page
 * (brand.ts, layers/steps.ts, connect/snippets.json) and cannot drift from them.
 * Pure string code: no DOM, no React.
 */
import { BRAND } from "./brand";
import { STEPS } from "./layers/steps";
import SNIPPETS from "./connect/snippets.json";
import FAQ from "./faq/faq.json";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export const PAGE_TITLE = BRAND.title;

/** Meta description: the entity definition first, so a quoted snippet still says what Cortex is. */
const META_DESCRIPTION = `${BRAND.definition} Your agent describes the task; Cortex simulates, collects demonstrations, trains and evaluates, with human approval.`;

const HERO_ALT =
  "Isometric line drawing of a six-axis robotic arm on a square base, the kind of tabletop manipulator Cortex trains pick-and-place policies for.";

/** schema.org graph: the app, its FAQ (same text as the visible accordion) and the connect-and-train how-to. */
export function jsonLd(site: string) {
  const app = {
    "@type": "SoftwareApplication",
    "@id": `${site}/#app`,
    name: BRAND.name,
    description: BRAND.definition,
    abstract: BRAND.description,
    applicationCategory: "DeveloperApplication",
    applicationSubCategory: "Robot learning",
    operatingSystem: "Windows, macOS, Linux",
    url: `${site}/`,
    image: `${site}/robot-still.png`,
    license: "https://opensource.org/licenses/MIT",
    isAccessibleForFree: true,
    offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    softwareHelp: { "@type": "CreativeWork", url: `${site}/llms.txt` },
    sameAs: ["https://github.com/psamin/speculativespeculativeagentaction"],
    keywords:
      "LeRobot, SO-101, SO-ARM101, MuJoCo, ACT, Action Chunking Transformer, imitation learning, behaviour cloning, robot policy training, sim-to-real, Model Context Protocol, MCP server, Claude Code",
    featureList: STEPS.map((s) => `${s.label}: ${s.body}`),
    additionalProperty: [
      { "@type": "PropertyValue", name: "MCP endpoint", value: SNIPPETS.mcp_url },
      { "@type": "PropertyValue", name: "MCP transport", value: "streamable-http" },
      { "@type": "PropertyValue", name: "MCP discovery", value: `${site}/.well-known/mcp.json` },
      { "@type": "PropertyValue", name: "REST tools", value: SNIPPETS.rest_url },
    ],
  };
  const faq = {
    "@type": "FAQPage",
    "@id": `${site}/#faq`,
    mainEntity: FAQ.items.map((f) => ({
      "@type": "Question",
      name: f.q,
      acceptedAnswer: { "@type": "Answer", text: f.a },
    })),
  };
  const claude = SNIPPETS.clients.find((c) => c.id === "claude")!;
  const howTo = {
    "@type": "HowTo",
    "@id": `${site}/#connect`,
    name: "Connect your agent to Cortex and train a robot policy",
    description: "Add the Cortex MCP server to an MCP client such as Claude Code, then run the loop from task description to an evaluated ACT policy in MuJoCo.",
    tool: [
      { "@type": "HowToTool", name: "An MCP client, e.g. Claude Code, Cursor or VS Code" },
      { "@type": "HowToTool", name: "Cortex (backend on :8000, MCP server on :3000)" },
    ],
    step: [
      { "@type": "HowToStep", position: 1, name: "Add the MCP server", text: claude.code, url: `${site}/#connect` },
      ...SNIPPETS.session.map((s, i) => ({
        "@type": "HowToStep",
        position: i + 2,
        name: s.tool,
        text: `Call ${s.tool} ${JSON.stringify(s.args)}. Returns ${s.returns}.`,
      })),
    ],
  };
  return { "@context": "https://schema.org", "@graph": [app, faq, howTo] };
}

/** <meta>, OpenGraph, Twitter, alternate links and JSON-LD for <head>. */
export function headTags(site: string): string {
  const d = esc(META_DESCRIPTION);
  const t = esc(PAGE_TITLE);
  return [
    `<meta name="description" content="${d}" />`,
    `<link rel="canonical" href="${site}/" />`,
    `<link rel="alternate" type="text/plain" href="/llms.txt" title="llms.txt" />`,
    `<link rel="alternate" type="application/json" href="/.well-known/mcp.json" title="MCP server" />`,
    `<meta property="og:type" content="website" />`,
    `<meta property="og:site_name" content="${esc(BRAND.name)}" />`,
    `<meta property="og:title" content="${t}" />`,
    `<meta property="og:description" content="${d}" />`,
    `<meta property="og:url" content="${site}/" />`,
    `<meta property="og:image" content="${site}/robot-still.png" />`,
    `<meta property="og:image:alt" content="${esc(HERO_ALT)}" />`,
    `<meta name="twitter:card" content="summary_large_image" />`,
    `<meta name="twitter:title" content="${t}" />`,
    `<meta name="twitter:description" content="${d}" />`,
    `<meta name="twitter:image" content="${site}/robot-still.png" />`,
    `<script type="application/ld+json">${JSON.stringify(jsonLd(site)).replace(/</g, "\\u003c")}</script>`,
  ].join("\n    ");
}

/** Plain semantic copy of the landing page. Hidden once JS runs (html.js), so browsers only ever see the app. */
export function staticMain(): string {
  const clients = SNIPPETS.clients
    .map((c) => `<h3>${esc(c.label)}${c.file === "Terminal" ? "" : ` (${esc(c.file)})`}</h3>\n<pre><code>${esc(c.code)}</code></pre>`)
    .join("\n");
  const session = SNIPPETS.session
    .map((s) => `<li><code>${esc(s.tool)} ${esc(JSON.stringify(s.args))}</code>: ${esc(s.returns)}</li>`)
    .join("\n");
  const steps = STEPS.map((s) => `<li><strong>${esc(s.label)}.</strong> ${esc(s.body)}</li>`).join("\n");
  const faq = FAQ.items
    .map((f) => `<section aria-labelledby="q-${f.id}"><h2 id="q-${f.id}">${esc(f.q)}</h2>\n<p>${esc(f.a)}</p></section>`)
    .join("\n");
  return `<main id="landing-static">
<header><p><a href="/">${esc(BRAND.name)}</a> · <a href="/studio">Open studio</a> · <a href="/llms.txt">llms.txt</a></p></header>
<h1>${esc(BRAND.name)}: ${esc(BRAND.tagline.toLowerCase())}</h1>
<p><strong>${esc(BRAND.definition)}</strong></p>
<p>${esc(BRAND.description)}</p>
<img src="/robot-still.png" width="820" height="610" alt="${esc(HERO_ALT)}" />
<section aria-labelledby="s-steps"><h2 id="s-steps">How Cortex trains a robot policy: seven steps</h2>
<ol>
${steps}
</ol></section>
<section aria-labelledby="s-connect"><h2 id="s-connect">How to connect your agent to Cortex over MCP</h2>
<p>Cortex runs a Model Context Protocol server (streamable HTTP) at ${esc(SNIPPETS.mcp_url)}. Every tool is also on REST at ${esc(SNIPPETS.rest_url)}.</p>
${clients}
<h3>A worked session: pick up the small water bottle and put it on the mat</h3>
<ol>
${session}
</ol></section>
${faq}
<section aria-labelledby="s-links"><h2 id="s-links">Machine-readable references for agents</h2>
<ul>
<li><a href="/llms.txt">llms.txt</a>: what Cortex is, how to connect, task recipes</li>
<li><a href="/llms-full.txt">llms-full.txt</a>: every tool with its arguments and JSON schema</li>
<li><a href="/.well-known/mcp.json">/.well-known/mcp.json</a>: MCP endpoint, REST mirror and tool list</li>
<li><a href="/server-card.json">MCP Server Card</a> (SEP-2127), listed in <a href="/.well-known/ai-catalog.json">/.well-known/ai-catalog.json</a></li>
<li><a href="https://github.com/psamin/speculativespeculativeagentaction">Source code on GitHub</a> (MIT)</li>
</ul></section>
</main>`;
}
