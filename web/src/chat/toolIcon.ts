import type { IsoIconName } from "../sketch";
import { normalizeTool } from "./model";

/** Which isometric glyph a tool row or job row shows, by the kind of work it does. */
export function toolIcon(tool: string): IsoIconName {
  const t = normalizeTool(tool).toLowerCase();
  if (/approv|budget/.test(t)) return "approve";
  if (/^robot_|collect|demonstrat|record|episode|grasp|arm/.test(t)) return "robot";
  if (/import|dataset|hub/.test(t)) return "dataset";
  if (/train|checkpoint|gpu/.test(t)) return "train";
  if (/evaluat|compare|test|benchmark|rollout/.test(t)) return "evaluate";
  if (/improve|recommend|diagnos|analy/.test(t)) return "improve";
  if (/scene|frame|camera|observe|capture|simulation/.test(t)) return "camera";
  return "cube";
}
