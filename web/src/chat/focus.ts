import { createContext, useContext } from "react";

/** Lets chat rows select rollouts in the viewer ("Watch in viewer"). */
export const ViewerFocusContext = createContext<{ focus: (ids: string[]) => void } | null>(null);
export const useViewerFocus = () => useContext(ViewerFocusContext);
