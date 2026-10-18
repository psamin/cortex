import { createContext, useContext } from "react";
import type { RolloutFilter } from "./Viewer";

/** Lets chat rows drive the viewer: select rollouts ("Watch in viewer") or switch its filter. */
export const ViewerFocusContext = createContext<{ focus: (ids: string[]) => void; filter: (f: RolloutFilter) => void } | null>(null);
export const useViewerFocus = () => useContext(ViewerFocusContext);
