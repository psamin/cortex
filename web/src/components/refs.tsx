import { useState } from "react";
import { useNavigate } from "react-router-dom";

/** Map an id prefix to where that object lives in the detail pages. */
export function refRoute(pid: string, id: string): string | null {
  const p = id.split("_")[0];
  const map: Record<string, string> = {
    ep: `/p/${pid}/analyze?ep=${id}`,
    an: `/p/${pid}/analyze`,
    fc: `/p/${pid}/analyze`,
    rec: `/p/${pid}/improve?tab=recommendations`,
    ds: `/p/${pid}/improve?tab=datasets`,
    tj: `/p/${pid}/improve?tab=training`,
    ck: `/p/${pid}/improve?tab=training`,
    ev: `/p/${pid}/improve?tab=evaluations`,
    br: `/p/${pid}/improve?tab=evaluations`,
    ac: `/p/${pid}/collect`,
    at: `/p/${pid}/collect`,
    prj: `/p/${pid}/overview`,
  };
  return map[p] ?? null;
}

export function RefChip({ pid, id }: { pid: string; id: string }) {
  const nav = useNavigate();
  const to = refRoute(pid, id);
  return (
    <button
      type="button"
      disabled={!to}
      onClick={() => to && nav(to)}
      className="rounded-sm border border-line-strong bg-plate px-1.5 py-px font-mono text-[10.5px] text-ink-2 hover:border-ink hover:text-ink disabled:cursor-default disabled:hover:border-line-strong disabled:hover:text-ink-2"
      title={to ? `Open ${id}` : id}
    >
      {id}
    </button>
  );
}

export function useApprover() {
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("cortex.approver") ?? "";
    } catch {
      return "";
    }
  });
  const save = (n: string) => {
    setName(n);
    try {
      localStorage.setItem("cortex.approver", n);
    } catch {
      /* storage unavailable */
    }
  };
  return [name, save] as const;
}
