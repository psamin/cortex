import { useNavigate } from "react-router-dom";

/** Map an id prefix to where that object lives in the studio. */
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
    prj: `/p/${pid}`,
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
      className="rounded-sm bg-rail px-1.5 py-px font-mono text-[10.5px] text-ink-2 shadow-[0_0_0_1px_var(--line)] hover:text-servo disabled:cursor-default disabled:hover:text-ink-2"
      title={to ? `Open ${id}` : id}
    >
      {id}
    </button>
  );
}
