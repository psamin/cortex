import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import { ArrowRight, Database, HardDrive, Maximize2, X } from "lucide-react";
import { useLastEntityAt } from "../api/events";
import { useGraph } from "../api/hooks";
import type { GraphEdge, GraphNode, ProjectGraph } from "../api/types";
import { refRoute } from "../agent/AgentPanel";
import { Button, EmptyState, ErrorState, Skeleton } from "../components/ui";
import { cn, evidenceValue, humanize } from "../lib/format";

// ------------------------------------------------------------------ visual encoding
// Five families (fixed hue order from the validated chart palette + the servo accent);
// shape separates types within a family. The canvas is always dark: this view is a showpiece.
const FAMILY = {
  control: "#f0a43a", // project, autopilot (servo)
  data: "#3987e5", // condition, episode, dataset
  insight: "#199e70", // analysis, cluster, recommendation
  model: "#d55181", // training, checkpoint
  eval: "#b2b5bc", // evaluation (branches are colored by measured success)
} as const;

type Shape = "circle" | "ring" | "square" | "diamond";
const TYPES: Record<string, { family: keyof typeof FAMILY; shape: Shape; r: number; label: string }> = {
  project: { family: "control", shape: "square", r: 9, label: "Project" },
  autopilot: { family: "control", shape: "diamond", r: 8, label: "Autopilot" },
  condition: { family: "data", shape: "ring", r: 4.5, label: "Condition" },
  episode: { family: "data", shape: "circle", r: 3, label: "Episode" },
  dataset: { family: "data", shape: "square", r: 7, label: "Dataset" },
  analysis: { family: "insight", shape: "circle", r: 6, label: "Analysis" },
  cluster: { family: "insight", shape: "ring", r: 5, label: "Failure cluster" },
  recommendation: { family: "insight", shape: "diamond", r: 4, label: "Recommendation" },
  training: { family: "model", shape: "square", r: 7, label: "Training" },
  checkpoint: { family: "model", shape: "diamond", r: 6, label: "Checkpoint" },
  evaluation: { family: "eval", shape: "square", r: 7, label: "Evaluation" },
  branch: { family: "eval", shape: "circle", r: 5, label: "Branch" },
};
const typeInfo = (t: string) => TYPES[t] ?? { family: "eval" as const, shape: "circle" as Shape, r: 4, label: humanize(t) };

/** red (0) → green (1) for measured success rates */
function successColor(rate: number) {
  const a = [236, 96, 96];
  const b = [74, 196, 128];
  const t = Math.max(0, Math.min(1, rate));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

function branchN(meta: Record<string, unknown>): number {
  const s = meta.success;
  if (typeof s === "string" && s.includes("/")) return Number(s.split("/")[1]) || 0;
  return Array.isArray(meta.seeds) ? meta.seeds.length : 0;
}

function radius(n: GraphNode): number {
  const base = typeInfo(n.type).r;
  if (n.type === "branch") return base + Math.sqrt(branchN(n.meta)) * 1.6;
  if (n.type === "condition" && typeof n.meta.target === "number") return base + Math.sqrt(n.meta.target);
  if (n.type === "recommendation" && typeof n.meta.score === "number") return base + Math.min(4, n.meta.score);
  if (n.type === "cluster" && Array.isArray(n.meta.episodes)) return base + Math.sqrt(n.meta.episodes.length);
  return base;
}

function color(n: GraphNode): string {
  if (n.type === "branch" && typeof n.meta.success_rate === "number") return successColor(n.meta.success_rate);
  return FAMILY[typeInfo(n.type).family];
}

export const isFallbackBackend = (b: string | null | undefined) => !!b && /fallback|sqlite|local/i.test(b);

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

type GNode = NodeObject<GraphNode & { born?: number; r: number; color: string }>;
type GLink = LinkObject<GNode, { kind: string }>;

const BG = "#0b0c0e";

// ------------------------------------------------------------------ graph canvas
function GraphCanvas({
  data,
  hidden,
  selectedId,
  onSelect,
  height,
  compact,
}: {
  data: ProjectGraph;
  hidden: Set<string>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  height: number;
  compact?: boolean;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const fg = useRef<ForceGraphMethods<GNode, GLink> | undefined>(undefined);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<string | null>(null);
  const cache = useRef(new Map<string, GNode>());
  const fitted = useRef(false);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Merge new data into cached node objects so existing nodes keep their positions
  // and new ones spawn next to a neighbour, then "pop" in.
  const graphData = useMemo(() => {
    const now = performance.now();
    const firstLoad = cache.current.size === 0;
    const next = new Map<string, GNode>();
    for (const n of data.nodes) {
      const prev = cache.current.get(n.id);
      if (prev) {
        Object.assign(prev, n, { r: radius(n), color: color(n) });
        next.set(n.id, prev);
      } else {
        const anchorId = data.edges.find((e) => e.target === n.id && cache.current.has(e.source))?.source ??
          data.edges.find((e) => e.source === n.id && cache.current.has(e.target))?.target;
        const anchor = anchorId ? cache.current.get(anchorId) : undefined;
        const node: GNode = { ...n, r: radius(n), color: color(n), born: firstLoad ? undefined : now };
        if (anchor?.x !== undefined && anchor.y !== undefined) {
          node.x = anchor.x + (Math.random() - 0.5) * 20;
          node.y = anchor.y + (Math.random() - 0.5) * 20;
        }
        next.set(n.id, node);
      }
    }
    cache.current = next;
    const nodes = [...next.values()].filter((n) => !hidden.has(n.type));
    const vis = new Set(nodes.map((n) => n.id));
    const links: GLink[] = data.edges
      .filter((e) => vis.has(e.source) && vis.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, kind: e.kind }));
    return { nodes, links };
  }, [data, hidden]);

  // keep the graph framed when the canvas is resized (drawer opens, agent panel toggles)
  useEffect(() => {
    if (!fitted.current) return;
    const t = setTimeout(() => fg.current?.zoomToFit(300, compact ? 20 : 60), 250);
    return () => clearTimeout(t);
  }, [width, compact]);

  const neighbours = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of data.edges) {
      if (!m.has(e.source)) m.set(e.source, new Set());
      if (!m.has(e.target)) m.set(e.target, new Set());
      m.get(e.source)!.add(e.target);
      m.get(e.target)!.add(e.source);
    }
    return m;
  }, [data.edges]);

  const focus = hover ?? selectedId;
  const isLit = useCallback(
    (id: string) => !focus || id === focus || !!neighbours.get(focus)?.has(id),
    [focus, neighbours],
  );

  useEffect(() => {
    const g = fg.current;
    if (!g) return;
    g.d3Force("charge")?.strength(compact ? -60 : -90);
    (g.d3Force("link") as { distance?: (d: number) => void } | undefined)?.distance?.(compact ? 26 : 36);
  }, [compact]);

  const paintNode = useCallback(
    (node: GNode, ctx: CanvasRenderingContext2D, scale: number) => {
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const r = node.r;
      const info = typeInfo(node.type);
      const lit = isLit(node.id);
      const dimmed = node.type === "episode" && node.status === "rejected";
      ctx.globalAlpha = lit ? (dimmed ? 0.4 : 1) : 0.28;

      // pop-in ring for freshly arrived nodes
      if (node.born) {
        const age = (performance.now() - node.born) / 1400;
        if (age < 1) {
          ctx.beginPath();
          ctx.arc(x, y, r + 2 + age * 14, 0, Math.PI * 2);
          ctx.strokeStyle = node.color;
          ctx.globalAlpha = (1 - age) * 0.8;
          ctx.lineWidth = 1.5 / scale;
          ctx.stroke();
          ctx.globalAlpha = lit ? 1 : 0.28;
        } else node.born = undefined;
      }

      ctx.beginPath();
      if (info.shape === "square") {
        const s = r * 1.7;
        ctx.roundRect(x - s / 2, y - s / 2, s, s, s * 0.22);
      } else if (info.shape === "diamond") {
        ctx.moveTo(x, y - r * 1.15);
        ctx.lineTo(x + r * 1.15, y);
        ctx.lineTo(x, y + r * 1.15);
        ctx.lineTo(x - r * 1.15, y);
        ctx.closePath();
      } else {
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      if (info.shape === "ring") {
        ctx.fillStyle = BG;
        ctx.fill();
        ctx.strokeStyle = node.color;
        ctx.lineWidth = Math.max(1.4, r * 0.32);
        ctx.stroke();
      } else {
        ctx.fillStyle = node.color;
        ctx.fill();
      }
      // hypotheses get a dashed halo (same provenance language as the rest of the studio)
      if (node.type === "recommendation" && node.meta.support === "hypothesis") {
        ctx.beginPath();
        ctx.arc(x, y, r * 1.6, 0, Math.PI * 2);
        ctx.setLineDash([2 / scale, 2 / scale]);
        ctx.strokeStyle = "rgba(255,255,255,.45)";
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (node.id === selectedId) {
        ctx.beginPath();
        ctx.arc(x, y, r * 1.6 + 2, 0, Math.PI * 2);
        ctx.strokeStyle = "#f0a43a";
        ctx.lineWidth = 2 / scale;
        ctx.stroke();
      }

      // labels: always for anchors, on zoom for the rest
      const anchor = ["project", "dataset", "training", "evaluation", "checkpoint", "autopilot", "analysis"].includes(node.type);
      if ((anchor && scale > 0.9) || scale > 2.2 || node.id === focus) {
        const fs = Math.max(10 / scale, 2.5);
        ctx.font = `500 ${fs}px "Inter Variable", system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const text = node.label.length > 28 ? node.label.slice(0, 27) + "…" : node.label;
        ctx.fillStyle = lit ? "rgba(236,237,239,.92)" : "rgba(236,237,239,.2)";
        ctx.fillText(text, x, y + r * 1.25 + 2 / scale);
      }
      ctx.globalAlpha = 1;
    },
    [isLit, selectedId, focus],
  );

  const tooltip = useCallback((n: GNode) => {
    const meta = Object.entries(n.meta)
      .filter(([k, v]) => k !== "storage" && v !== null && typeof v !== "object")
      .slice(0, 6)
      .map(([k, v]) => `<div style="display:flex;gap:10px;justify-content:space-between"><span style="color:#80848c">${esc(k)}</span><span style="font-family:'JetBrains Mono Variable',monospace">${esc(evidenceValue(v))}</span></div>`)
      .join("");
    return `<div style="background:#1f2125;color:#eceDef;border:1px solid rgba(255,255,255,.12);border-radius:6px;padding:8px 10px;font:12px/1.45 'Inter Variable',system-ui;max-width:280px;box-shadow:0 8px 20px -8px rgba(0,0,0,.6)">
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px"><span style="width:8px;height:8px;border-radius:2px;background:${n.color}"></span><span style="color:#80848c;text-transform:uppercase;font-size:10px;letter-spacing:.06em">${esc(typeInfo(n.type).label)}</span>${n.status ? `<span style="color:#b2b5bc;font-size:10px">· ${esc(n.status)}</span>` : ""}</div>
      <div style="font-weight:600;margin-bottom:4px">${esc(n.label)}</div>${meta}</div>`;
  }, []);

  return (
    <div ref={wrap} className="relative w-full overflow-hidden" style={{ height, background: BG }}>
      <ForceGraph2D<GNode, GLink>
        ref={fg}
        graphData={graphData}
        width={width}
        height={height}
        backgroundColor={BG}
        nodeId="id"
        nodeVal={(n) => n.r}
        nodeLabel={tooltip}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(n, c, ctx) => {
          ctx.fillStyle = c;
          ctx.beginPath();
          ctx.arc(n.x ?? 0, n.y ?? 0, n.r * 1.5 + 2, 0, Math.PI * 2);
          ctx.fill();
        }}
        linkColor={(l) => {
          const s = typeof l.source === "object" ? (l.source as GNode).id : String(l.source);
          const t = typeof l.target === "object" ? (l.target as GNode).id : String(l.target);
          const lit = !focus || s === focus || t === focus;
          return lit ? (l.kind.startsWith("evidence") ? "rgba(25,158,112,.45)" : "rgba(255,255,255,.16)") : "rgba(255,255,255,.04)";
        }}
        linkLineDash={(l) => (l.kind.startsWith("evidence") || l.kind === "reference" ? [2, 2] : null)}
        linkWidth={0.8}
        linkDirectionalArrowLength={compact ? 0 : 2.5}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={() => "rgba(255,255,255,.25)"}
        linkDirectionalParticles={(l) => (l.kind === "reference" ? 0 : 1)}
        linkDirectionalParticleSpeed={0.006}
        linkDirectionalParticleWidth={compact ? 1.4 : 1.8}
        linkDirectionalParticleColor={(l) => {
          const s = l.source as GNode;
          return typeof s === "object" && s.color ? s.color : "#f0a43a";
        }}
        autoPauseRedraw={false}
        cooldownTicks={120}
        d3VelocityDecay={0.3}
        onEngineStop={() => {
          if (!fitted.current) {
            fitted.current = true;
            fg.current?.zoomToFit(500, compact ? 20 : 60);
            // small graphs would otherwise be blown up to fill the frame
            setTimeout(() => {
              const max = compact ? 2 : 3;
              if ((fg.current?.zoom() ?? 0) > max) fg.current?.zoom(max, 300);
            }, 550);
          }
        }}
        onNodeHover={(n) => setHover(n ? n.id : null)}
        onNodeClick={(n) => onSelect(n.id)}
        onBackgroundClick={() => onSelect(null)}
        enableZoomInteraction={!compact}
      />
      {!compact && (
        <button
          type="button"
          onClick={() => fg.current?.zoomToFit(400, 60)}
          className="absolute bottom-3 right-3 rounded bg-white/5 p-1.5 text-white/60 shadow-[0_0_0_1px_rgba(255,255,255,.1)] hover:text-white"
          aria-label="Fit graph to view"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

/** Refetch the graph shortly after any entity change, so nodes pop in live as work lands. */
function useLiveGraph(pid: string) {
  const q = useGraph(pid);
  const qc = useQueryClient();
  const last = useLastEntityAt();
  useEffect(() => {
    if (!last) return;
    const t = setTimeout(() => qc.invalidateQueries({ queryKey: ["graph", pid] }), 600);
    return () => clearTimeout(t);
  }, [last, pid, qc]);
  return q;
}

// ------------------------------------------------------------------ full page
export function LineagePage() {
  const { pid = "" } = useParams();
  const q = useLiveGraph(pid);
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("node");
  const select = (id: string | null) =>
    setParams((p) => {
      if (id) p.set("node", id);
      else p.delete("node");
      return p;
    }, { replace: true });
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [height, setHeight] = useState(640);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setHeight(Math.max(420, Math.floor(e.contentRect.height))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const data = q.data;
  const counts = useMemo(() => {
    const c = new Map<string, number>();
    for (const n of data?.nodes ?? []) c.set(n.type, (c.get(n.type) ?? 0) + 1);
    return c;
  }, [data]);
  const selected = data?.nodes.find((n) => n.id === selectedId) ?? null;

  return (
    <div className="flex h-full flex-col gap-3 px-5 py-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-[-0.01em]">Lineage</h1>
          <p className="mt-0.5 text-sm text-ink-3">
            Every object in this project and how it was produced, live from the database. Click a node for its provenance.
          </p>
        </div>
        {data && <BackendBadge backend={data.backend} />}
      </div>

      {data && (
        <div className="flex flex-wrap gap-1">
          {[...counts.entries()]
            .sort((a, b) => Object.keys(TYPES).indexOf(a[0]) - Object.keys(TYPES).indexOf(b[0]))
            .map(([t, n]) => {
              const off = hidden.has(t);
              const info = typeInfo(t);
              return (
                <button
                  key={t}
                  type="button"
                  aria-pressed={!off}
                  onClick={() =>
                    setHidden((h) => {
                      const s = new Set(h);
                      s.has(t) ? s.delete(t) : s.add(t);
                      return s;
                    })
                  }
                  className={cn(
                    "inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-2xs font-medium transition-opacity duration-100",
                    off ? "bg-transparent text-ink-4 opacity-60 shadow-[0_0_0_1px_var(--line)]" : "bg-rail text-ink-2 shadow-[0_0_0_1px_var(--line)]",
                  )}
                >
                  <ShapeSwatch shape={info.shape} color={FAMILY[info.family]} />
                  {info.label}
                  <span className="num font-mono text-ink-4">{n}</span>
                </button>
              );
            })}
          <span className="ml-1 inline-flex h-6 items-center gap-1.5 px-1 text-2xs text-ink-3">
            <span className="h-2 w-8 rounded-full" style={{ background: `linear-gradient(90deg, ${successColor(0)}, ${successColor(1)})` }} />
            branch success 0→100%
          </span>
        </div>
      )}

      <div ref={box} className="relative min-h-[420px] flex-1 overflow-hidden rounded-md shadow-[0_0_0_1px_var(--line)]">
        {q.isLoading ? (
          <Skeleton className="h-full w-full rounded-none" />
        ) : q.error && !data ? (
          <div className="p-4">
            <ErrorState error={q.error} onRetry={() => q.refetch()} />
          </div>
        ) : data && data.nodes.length === 0 ? (
          <EmptyState title="Nothing in the graph yet" className="h-full">Record demos or start autopilot; nodes appear here as work lands.</EmptyState>
        ) : data ? (
          <div className="flex h-full">
            <div className="min-w-0 flex-1">
              <GraphCanvas data={data} hidden={hidden} selectedId={selectedId} onSelect={select} height={height} />
            </div>
            {selected && <NodeDrawer node={selected} graph={data} pid={pid} onSelect={select} />}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ShapeSwatch({ shape, color }: { shape: Shape; color: string }) {
  if (shape === "ring") return <span className="h-2.5 w-2.5 rounded-full" style={{ boxShadow: `inset 0 0 0 2px ${color}` }} />;
  if (shape === "square") return <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: color }} />;
  if (shape === "diamond") return <span className="h-2 w-2 rotate-45 rounded-[1px]" style={{ background: color }} />;
  return <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />;
}

function BackendBadge({ backend }: { backend: ProjectGraph["backend"] }) {
  const fb = isFallbackBackend(backend.db);
  return (
    <div className={cn("flex items-center gap-3 rounded px-2.5 py-1.5 text-2xs", fb ? "prov-fallback text-ink-2" : "bg-servo/10 text-ink shadow-[0_0_0_1px_rgb(var(--servo)/.35)]")}>
      <span className="flex items-center gap-1.5">
        <Database className={cn("h-3.5 w-3.5", fb ? "text-caution" : "text-servo")} />
        {fb ? "local fallback" : "InsForge Postgres"} <span className="font-mono text-ink-3">{backend.db}</span>
      </span>
      <span className="flex items-center gap-1.5">
        <HardDrive className={cn("h-3.5 w-3.5", isFallbackBackend(backend.storage) ? "text-caution" : "text-servo")} />
        <span className="font-mono text-ink-3">{backend.storage}</span>
      </span>
    </div>
  );
}

// ------------------------------------------------------------------ drawer
function NodeDrawer({
  node,
  graph,
  pid,
  onSelect,
}: {
  node: GraphNode;
  graph: ProjectGraph;
  pid: string;
  onSelect: (id: string | null) => void;
}) {
  const nav = useNavigate();
  const info = typeInfo(node.type);
  const fb = isFallbackBackend(node.source.backend);
  const storage = (node.meta.storage ?? null) as { bucket?: string; keys?: string[]; prefix?: string } | null;
  const meta = Object.entries(node.meta).filter(([k]) => k !== "storage");
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const inEdges = graph.edges.filter((e) => e.target === node.id);
  const outEdges = graph.edges.filter((e) => e.source === node.id);
  const studio = refRoute(pid, node.source.row_id ?? node.id);

  return (
    <aside className="enter flex w-[340px] shrink-0 flex-col border-l border-white/10 bg-[#121316] text-[#eceDef]">
      <header className="flex items-start gap-2 border-b border-white/10 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-[0.06em] text-white/50">
            <ShapeSwatch shape={info.shape} color={color(node)} />
            {info.label}
            {node.status && <span className="normal-case tracking-normal text-white/60">· {node.status}</span>}
          </div>
          <h2 className="mt-1 break-words text-base font-semibold">{node.label}</h2>
        </div>
        <button type="button" onClick={() => onSelect(null)} className="rounded p-1 text-white/50 hover:bg-white/10 hover:text-white" aria-label="Close">
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3 scroll-thin">
        {/* provenance: the showcase */}
        <section
          className={cn(
            "rounded-md p-3",
            fb ? "border border-dashed border-[rgb(226_196_84/.55)] bg-[rgb(226_196_84/.05)]" : "bg-[rgb(240_164_58/.08)] shadow-[0_0_0_1px_rgb(240_164_58/.4)]",
          )}
        >
          <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-[0.06em] text-white/50">
            <Database className="h-3 w-3" /> Provenance
          </div>
          <p className="mt-1.5 text-sm font-semibold">{fb ? "Local fallback" : "InsForge Postgres"}</p>
          <p className="font-mono text-[11px] text-white/60">{node.source.backend}</p>
          {node.source.table ? (
            <dl className="mt-2 grid grid-cols-[48px_1fr] gap-y-1 font-mono text-[11px]">
              <dt className="text-white/45">table</dt>
              <dd className="text-white">{node.source.table}</dd>
              <dt className="text-white/45">row</dt>
              <dd className="break-all text-white">{node.source.row_id}</dd>
            </dl>
          ) : (
            <p className="mt-2 text-[11px] text-white/50">Derived node: not a stored row (computed from the project plan or references).</p>
          )}
          {storage && (
            <div className="mt-3 border-t border-white/10 pt-2">
              <div className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-[0.06em] text-white/50">
                <HardDrive className="h-3 w-3" /> Storage
                {storage.bucket && <span className={cn("normal-case tracking-normal", isFallbackBackend(storage.bucket) ? "text-[rgb(226_196_84)]" : "text-white/70")}>· {storage.bucket}</span>}
              </div>
              {storage.prefix && <p className="mt-1 break-all font-mono text-[11px] text-white/80">{storage.prefix}</p>}
              {storage.keys?.map((k) => (
                <p key={k} className="mt-1 break-all font-mono text-[11px] text-white/80">
                  {k}
                </p>
              ))}
            </div>
          )}
        </section>

        {meta.length > 0 && (
          <section>
            <div className="mb-1.5 text-2xs font-medium uppercase tracking-[0.06em] text-white/50">Fields</div>
            <dl className="space-y-1 text-xs">
              {meta.map(([k, v]) => (
                <div key={k} className="grid grid-cols-[110px_1fr] gap-2">
                  <dt className="truncate text-white/45">{humanize(k)}</dt>
                  <dd className="break-words font-mono text-[11px] text-white/90">{v === null ? "—" : evidenceValue(v)}</dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        <EdgeList title="From" edges={inEdges} other={(e) => byId.get(e.source)} onSelect={onSelect} />
        <EdgeList title="To" edges={outEdges} other={(e) => byId.get(e.target)} onSelect={onSelect} />
      </div>

      {studio && (
        <footer className="border-t border-white/10 px-4 py-3">
          <Button size="sm" className="w-full" onClick={() => nav(studio)}>
            Open in studio <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </footer>
      )}
    </aside>
  );
}

function EdgeList({
  title,
  edges,
  other,
  onSelect,
}: {
  title: string;
  edges: GraphEdge[];
  other: (e: GraphEdge) => GraphNode | undefined;
  onSelect: (id: string) => void;
}) {
  if (!edges.length) return null;
  return (
    <section>
      <div className="mb-1.5 text-2xs font-medium uppercase tracking-[0.06em] text-white/50">
        {title} · {edges.length}
      </div>
      <ul className="space-y-0.5">
        {edges.slice(0, 40).map((e, i) => {
          const n = other(e);
          if (!n) return null;
          return (
            <li key={i}>
              <button type="button" onClick={() => onSelect(n.id)} className="flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-xs hover:bg-white/5">
                <ShapeSwatch shape={typeInfo(n.type).shape} color={color(n)} />
                <span className="min-w-0 flex-1 truncate text-white/85">{n.label}</span>
                <span className="font-mono text-[10px] text-white/40">{e.kind}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ------------------------------------------------------------------ compact (overview)
export function LineagePreview({ pid }: { pid: string }) {
  const q = useLiveGraph(pid);
  const nav = useNavigate();
  const empty = useMemo(() => new Set<string>(), []);
  return (
    <section className="overflow-hidden rounded-md bg-plate shadow-[0_0_0_1px_var(--line)]">
      <header className="flex items-center gap-3 border-b border-line-soft px-3.5 py-2">
        <h2 className="text-sm font-semibold">Lineage</h2>
        {q.data && (
          <span className="text-xs text-ink-3">
            {q.data.nodes.length} objects · {q.data.edges.length} links · {isFallbackBackend(q.data.backend.db) ? "local fallback" : "InsForge"}
          </span>
        )}
        <Button size="sm" variant="ghost" className="ml-auto" onClick={() => nav(`/p/${pid}/graph`)}>
          Open graph <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      </header>
      {q.isLoading ? (
        <Skeleton className="h-[280px] w-full rounded-none" />
      ) : q.error && !q.data ? (
        <div className="p-3.5">
          <ErrorState error={q.error} compact onRetry={() => q.refetch()} />
        </div>
      ) : q.data && q.data.nodes.length > 0 ? (
        <GraphCanvas data={q.data} hidden={empty} selectedId={null} onSelect={(id) => id && nav(`/p/${pid}/graph?node=${encodeURIComponent(id)}`)} height={280} compact />
      ) : (
        <EmptyState title="No objects yet" />
      )}
    </section>
  );
}

