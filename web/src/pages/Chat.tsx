import { useLocation, useParams } from "react-router-dom";
import { ChatThread } from "../chat/ChatThread";

/** The project's main view: a full-height guided chat. */
export function ChatPage() {
  const { pid } = useParams<{ pid: string }>();
  const initial = (useLocation().state as { initial?: string } | null)?.initial;
  if (!pid) return null;
  return <ChatThread key={pid} pid={pid} initial={initial} />;
}
