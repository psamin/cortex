import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy } from "lucide-react";
import { Button, ErrorState, Field, Input, Panel, Segmented } from "../components/ui";
import { ago } from "../lib/format";
import { agentsApi, type AgentKey } from "./api";

/** The MCP server the snippets point at (`npm run dev` in mcp/ serves :3000/mcp). */
const MCP_URL = ((import.meta.env.VITE_MCP_URL as string | undefined) || "http://localhost:3000/mcp").replace(/\/$/, "");
const HEADER = "X-Cortex-Agent-Key";

type Client = "claude" | "cursor" | "vscode" | "json";

export function snippet(client: Client, key: string): string {
  switch (client) {
    case "claude":
      return `claude mcp add --transport http cortex ${MCP_URL} --header "${HEADER}: ${key}"`;
    case "cursor": // ~/.cursor/mcp.json
      return JSON.stringify({ mcpServers: { cortex: { url: MCP_URL, headers: { [HEADER]: key } } } }, null, 2);
    case "vscode": // .vscode/mcp.json
      return JSON.stringify({ servers: { cortex: { type: "http", url: MCP_URL, headers: { [HEADER]: key } } } }, null, 2);
    case "json":
      return JSON.stringify({ mcpServers: { cortex: { type: "http", url: MCP_URL, headers: { [HEADER]: key } } } }, null, 2);
  }
}

const WHERE: Record<Client, string> = {
  claude: "Run in a terminal. Claude Code then sends the key on every call.",
  cursor: "Add to ~/.cursor/mcp.json (or .cursor/mcp.json in a project).",
  vscode: "Add to .vscode/mcp.json in your workspace.",
  json: "For any client that takes the common mcpServers JSON (e.g. .mcp.json).",
};

function CopyBlock({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="relative">
      <pre className="scroll-thin overflow-x-auto whitespace-pre rounded-sm bg-rail p-3 pr-12 font-mono text-[11.5px] leading-relaxed text-ink">{text}</pre>
      <button
        type="button"
        aria-label="Copy"
        onClick={() => {
          void navigator.clipboard?.writeText(text).then(() => {
            setDone(true);
            setTimeout(() => setDone(false), 1500);
          });
        }}
        className="absolute right-2 top-2 rounded-sm border border-line-strong bg-plate p-1.5 text-ink-2 hover:text-ink"
      >
        {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

function Snippets({ value }: { value: string }) {
  const [client, setClient] = useState<Client>("claude");
  return (
    <div className="space-y-2">
      <Segmented
        size="sm"
        ariaLabel="Client"
        value={client}
        onChange={setClient}
        options={[
          { value: "claude", label: "Claude Code" },
          { value: "cursor", label: "Cursor" },
          { value: "vscode", label: "VS Code" },
          { value: "json", label: "JSON" },
        ]}
      />
      <CopyBlock text={snippet(client, value)} />
      <p className="text-xs text-ink-3">{WHERE[client]}</p>
    </div>
  );
}

/** Settings → Agents: create a key (shown once), connect a client, list and revoke keys. */
export function KeysPanel() {
  const qc = useQueryClient();
  const keys = useQuery({ queryKey: ["agents", "keys"], queryFn: agentsApi.keys });
  const [label, setLabel] = useState("");
  const [fresh, setFresh] = useState<AgentKey | null>(null);
  const create = useMutation({
    mutationFn: () => agentsApi.createKey(label.trim() || "agent"),
    onSuccess: (k) => {
      setFresh(k);
      setLabel("");
      void qc.invalidateQueries({ queryKey: ["agents", "keys"] });
    },
  });
  const revoke = useMutation({
    mutationFn: agentsApi.revokeKey,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["agents", "keys"] }),
  });

  return (
    <div className="space-y-6">
      <Panel title="Connect an agent" meta="a key ties its calls to you">
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            create.mutate();
          }}
        >
          <div className="min-w-[220px] flex-1">
            <Field label="Label">
              <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="laptop Claude Code" />
            </Field>
          </div>
          <Button type="submit" variant="primary" loading={create.isPending}>
            Create agent key
          </Button>
        </form>
        {create.error && <div className="mt-3"><ErrorState error={create.error} compact /></div>}
        {fresh?.key && (
          <div className="mt-4 space-y-3 border-t border-line pt-4">
            <p className="text-[13px] text-ink">
              Your key, shown <strong>once</strong>. Copy it now; Cortex keeps only a hash.
            </p>
            <CopyBlock text={fresh.key} />
            <Snippets value={fresh.key} />
          </div>
        )}
        {!fresh?.key && (
          <div className="mt-4 space-y-2 border-t border-line pt-4">
            <p className="text-xs text-ink-3">Snippet shape (create a key to fill it in):</p>
            <Snippets value="ck_…" />
          </div>
        )}
      </Panel>

      <Panel title="Agent keys" meta={keys.data ? `${keys.data.filter((k) => k.status === "active").length} active` : undefined} bodyClassName="p-0">
        {keys.error ? (
          <div className="p-4"><ErrorState error={keys.error} onRetry={() => keys.refetch()} compact /></div>
        ) : !keys.data?.length ? (
          <p className="p-4 text-sm text-ink-3">No keys yet. Agents without a key are still shown, as the local owner's.</p>
        ) : (
          <ul className="divide-y divide-line">
            {keys.data.map((k) => (
              <li key={k.id} className="flex flex-wrap items-center gap-3 px-4 py-2.5 text-[13px]">
                <span className="font-mono text-xs text-ink">{k.prefix}…</span>
                <span className="text-ink-2">{k.label}</span>
                <span className="text-xs text-ink-3">created {ago(k.created_at)}</span>
                {k.status === "revoked" ? (
                  <span className="ml-auto font-mono text-[10.5px] uppercase tracking-[0.12em] text-fault">revoked</span>
                ) : (
                  <Button size="sm" variant="danger" className="ml-auto" loading={revoke.isPending && revoke.variables === k.id} onClick={() => revoke.mutate(k.id)}>
                    Revoke
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Panel>
      <p className="text-xs leading-5 text-ink-3">
        Keys only say whose agent is calling. They can never approve training or drive the real arm: those stay a human click in the studio.
      </p>
    </div>
  );
}
