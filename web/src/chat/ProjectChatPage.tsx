import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useProjectState } from "../api/hooks";
import { useChat } from "./ChatProvider";
import { Conversation } from "./Conversation";
import { ViewerFocusContext } from "./focus";
import { normalizeTool, prefixOf, toolLabel } from "./model";
import { Pipeline } from "./Pipeline";
import { Viewer, type RolloutFilter, type ViewerSelection } from "./Viewer";

/** The tools that build a scene from a prompt; while one runs, the viewer shows that the simulation is being built. */
const SCENE_TOOLS = new Set(["start_intake", "update_intake", "build_simulation"]);

function usePersisted(key: string, initial: boolean) {
  const [v, setV] = useState(() => {
    try {
      const s = localStorage.getItem(key);
      return s === null ? initial : s === "1";
    } catch {
      return initial;
    }
  });
  const set = (x: boolean) => {
    setV(x);
    try {
      localStorage.setItem(key, x ? "1" : "0");
    } catch {
      /* ignore */
    }
  };
  return [v, set] as const;
}

export function ProjectChatPage() {
  const { pid = "" } = useParams();
  const chat = useChat();
  const state = useProjectState(pid);
  const location = useLocation();
  const nav = useNavigate();
  // the simulation should dominate: on short screens the pipeline starts collapsed
  const [pipelineOpen, setPipelineOpen] = usePersisted("cortex.pipelineOpen", window.innerHeight >= 880);

  // viewer selection, shared with the pipeline (iteration timeline, failure analysis)
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filter, setFilter] = useState<RolloutFilter>("all");
  const [evaluationId, setEvaluationId] = useState<string | null>(null);
  const sel: ViewerSelection = { selectedId, setSelectedId, filter, setFilter, evaluationId, setEvaluationId };
  // "Watch in viewer" from a chat row: clear filters and select the first of those rollouts
  const focus = useMemo(
    () => ({
      focus: (ids: string[]) => {
        setEvaluationId(null);
        setFilter("all");
        setSelectedId(ids[0] ?? null);
      },
      filter: (f: RolloutFilter) => {
        setEvaluationId(null);
        setFilter(f);
        setSelectedId(null);
      },
    }),
    [],
  );

  // first message from the landing page
  const autoSent = useRef(false);
  useEffect(() => {
    const msg = (location.state as { autoSend?: string } | null)?.autoSend;
    if (!msg || autoSent.current || chat.loading) return;
    autoSent.current = true;
    nav(location.pathname, { replace: true, state: null });
    chat.send(msg);
  }, [chat.loading, location.state]); // eslint-disable-line react-hooks/exhaustive-deps

  // follow the agent: a new evaluation result loads its rollouts into the viewer
  const lastEval = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    if (chat.loading) return;
    let latest: string | null = null;
    for (let i = chat.items.length - 1; i >= 0 && !latest; i--) {
      const it = chat.items[i];
      if (it.kind === "tool") latest = it.refIds.find((r) => prefixOf(r) === "ev") ?? null;
    }
    if (lastEval.current === undefined) {
      lastEval.current = latest; // history load: don't jump
      return;
    }
    if (latest && latest !== lastEval.current) {
      lastEval.current = latest;
      setEvaluationId(latest);
      setFilter("eval");
      setSelectedId(null);
    }
  }, [chat.items, chat.loading]);

  const activity = chat.busy
    ? { busy: true, label: chat.status?.text ?? (chat.status?.state === "running" && chat.status.tool ? toolLabel(chat.status.tool) : "Agent is thinking") }
    : { busy: false, label: "Idle" };

  // without a scene built from the prompt the sim is the cube tabletop: the request was about something else
  const project = state.data?.project;
  const asked =
    project && !project.scene && project.spec.objects[0] && !project.task_prompt.toLowerCase().includes(project.spec.objects[0].toLowerCase())
      ? project.task_prompt
      : null;
  // a new task's scene is being set up (its first prompt, or the agent running the intake tools): the viewer says
  // so instead of showing the cube tabletop, another scene or old videos in between
  const firstPrompt = (location.state as { autoSend?: string } | null)?.autoSend;
  const sceneTool = chat.busy && !!chat.status?.tool && SCENE_TOOLS.has(normalizeTool(chat.status.tool));
  const lastUser = [...chat.items].reverse().find((it) => it.kind === "user");
  const building = sceneTool
    ? (lastUser?.kind === "user" ? lastUser.text : null) ?? asked ?? project?.task_prompt ?? ""
    : asked && (chat.busy || firstPrompt)
      ? asked
      : null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex min-h-0 flex-1">
        <div className="w-[35%] min-w-[380px] max-w-[620px] shrink-0 border-r border-line">
          <ViewerFocusContext.Provider value={focus}>
            <Conversation pid={pid} />
          </ViewerFocusContext.Provider>
        </div>
        <div className="min-w-0 flex-1">
          <Viewer pid={pid} goal={project?.spec.goal ?? project?.task_prompt ?? "…"} asked={asked} building={building}
            sceneDigest={project?.scene?.digest ?? null} activity={activity} sel={sel}
            taskHint={project ? `${project.name} ${project.task_prompt}` : ""} />
        </div>
      </div>
      <Pipeline pid={pid} open={pipelineOpen} onToggle={() => setPipelineOpen(!pipelineOpen)} sel={sel} />
    </div>
  );
}
