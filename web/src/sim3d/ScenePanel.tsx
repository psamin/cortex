import { useState } from "react";
import { API_BASE } from "../api/client";
import type { Project } from "../api/types";
import { Panel } from "../components/ui";
import { SceneView } from "./SceneView";

/** The fields of Project.scene the overview needs (the API returns the whole SceneInfo). */
interface BuiltScene {
  digest?: string;
  pick?: string | null;
}

/**
 * Project overview: the simulated workcell. Uses the project's built scene when there is one,
 * otherwise the upstream SO-101 tabletop the plan's conditions run on.
 */
export function ScenePanel({ project }: { project: Project }) {
  const scene = (project as Project & { scene?: BuiltScene | null }).scene ?? null;
  const [failed, setFailed] = useState(false);
  if (!scene && project.mode !== "sim") return null;
  const src = scene
    ? `${API_BASE}/projects/${project.id}/simulation/frame?camera=top`
    : `${API_BASE}/projects/${project.id}/sim/frame?cube_region=center&lighting=normal&seed=0`;

  return (
    <Panel
      title="Scene"
      meta={scene ? `built scene ${scene.digest ?? ""}` : "SO-101 tabletop (no scene built yet)"}
      bodyClassName="p-0"
    >
      <div className="relative aspect-[16/9] max-h-[420px] w-full overflow-hidden rounded-b-md bg-well">
        <SceneView
          pid={project.id}
          sceneKey={scene?.digest}
          condition={scene ? undefined : { cube_region: "center", lighting: "normal" }}
          camera={
            failed ? (
              <div className="flex h-full items-center justify-center text-xs text-ink-4">Sim render unavailable</div>
            ) : (
              <img
                src={src}
                alt={scene ? "Overhead camera of the built scene" : "Front camera of the tabletop"}
                onError={() => setFailed(true)}
                className="h-full w-full object-contain"
              />
            )
          }
        />
      </div>
    </Panel>
  );
}
