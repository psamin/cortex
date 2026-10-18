import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { refRoute } from "../components/refs";
import { splitIds } from "./timeline";

/**
 * Light markdown for assistant text: paragraphs, lists, headings, code, bold, links. No HTML injection.
 * Ids ("job_eee5…") show as chips with a plain noun ("this step"); the id is in the chip's title.
 */
export function Markdown({ text, pid, muted }: { text: string; pid?: string; muted?: boolean }) {
  const inline = (s: string) => inlineMd(s, pid);
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (line.trim().startsWith("```")) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) body.push(lines[i++]);
      i++;
      blocks.push(
        <pre key={key++} className="overflow-x-auto rounded-sm border border-line bg-well px-3 py-2 font-mono text-xs leading-relaxed text-ink-2 scroll-thin">
          {body.join("\n")}
        </pre>,
      );
      continue;
    }
    const ul = /^\s*[-*•]\s+/;
    const ol = /^\s*\d+[.)]\s+/;
    if (ul.test(line) || ol.test(line)) {
      const ordered = ol.test(line);
      const re = ordered ? ol : ul;
      const entries: string[] = [];
      while (i < lines.length && re.test(lines[i])) entries.push(lines[i++].replace(re, ""));
      const cls = "my-1 space-y-1 pl-5 " + (ordered ? "list-decimal" : "list-disc") + " marker:text-ink-4";
      blocks.push(
        ordered ? (
          <ol key={key++} className={cls}>
            {entries.map((e, k) => (
              <li key={k}>{inline(e)}</li>
            ))}
          </ol>
        ) : (
          <ul key={key++} className={cls}>
            {entries.map((e, k) => (
              <li key={k}>{inline(e)}</li>
            ))}
          </ul>
        ),
      );
      continue;
    }
    const h = line.match(/^#{1,6}\s+(.*)$/);
    if (h) {
      blocks.push(
        <p key={key++} className="pt-2 font-mono text-[11px] font-medium uppercase tracking-[0.16em] text-ink">
          {inline(h[1])}
        </p>,
      );
      i++;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !ul.test(lines[i]) && !ol.test(lines[i]) && !/^#{1,6}\s/.test(lines[i]) && !lines[i].trim().startsWith("```"))
      para.push(lines[i++]);
    blocks.push(<p key={key++}>{inline(para.join(" "))}</p>);
  }
  return <div className={muted ? "space-y-2 text-[14px] leading-relaxed text-ink-2" : "space-y-2.5 text-[15px] leading-relaxed text-ink"}>{blocks}</div>;
}

/** Text with each id as a chip: its noun, the id in the title, a link where the id has a page. */
function chips(s: string, pid: string | undefined, key: number): ReactNode[] {
  return splitIds(s).map((p, i) => {
    if (typeof p === "string") return p;
    const to = pid ? refRoute(pid, p.id) : null;
    const cls = "whitespace-nowrap rounded-sm bg-rail px-1 py-px text-[0.92em] text-ink-2 shadow-[0_0_0_1px_var(--line)]";
    return to ? (
      <Link key={`${key}-${i}`} to={to} title={p.id} className={`${cls} hover:text-ink`}>
        {p.noun}
      </Link>
    ) : (
      <span key={`${key}-${i}`} title={p.id} className={cls}>
        {p.noun}
      </span>
    );
  });
}

function inlineMd(s: string, pid: string | undefined): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(...chips(s.slice(last, m.index), pid, k++));
    const t = m[0];
    if (t.startsWith("**")) out.push(<strong key={k++} className="font-medium text-ink">{chips(t.slice(2, -2), pid, k++)}</strong>);
    else if (t.startsWith("`") && splitIds(t.slice(1, -1)).some((p) => typeof p !== "string")) out.push(...chips(t.slice(1, -1), pid, k++));
    else if (t.startsWith("`"))
      out.push(
        <code key={k++} className="rounded-sm border border-line bg-well px-1 py-px font-mono text-[0.85em] text-ink-2">
          {t.slice(1, -1)}
        </code>,
      );
    else {
      const label = t.slice(1, t.indexOf("]"));
      out.push(
        <a key={k++} href={m[2]} target="_blank" rel="noreferrer" className="border-b border-ink-3 text-ink hover:border-ink">
          {label}
        </a>,
      );
    }
    last = m.index + t.length;
  }
  if (last < s.length) out.push(...chips(s.slice(last), pid, k++));
  return out;
}
