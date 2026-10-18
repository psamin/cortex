import type { ReactNode } from "react";

/** Light markdown for assistant text: paragraphs, lists, headings, code, bold, links. No HTML injection. */
export function Markdown({ text }: { text: string }) {
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
        <pre key={key++} className="overflow-x-auto rounded-md bg-well px-3 py-2 font-mono text-xs leading-relaxed text-ink-2 scroll-thin">
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
        <p key={key++} className="pt-1 font-semibold text-ink">
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
  return <div className="space-y-2.5 text-[15px] leading-relaxed text-ink">{blocks}</div>;
}

function inline(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) out.push(<strong key={k++} className="font-semibold text-ink">{t.slice(2, -2)}</strong>);
    else if (t.startsWith("`"))
      out.push(
        <code key={k++} className="rounded bg-rail px-1 py-px font-mono text-[0.85em] text-ink-2">
          {t.slice(1, -1)}
        </code>,
      );
    else {
      const label = t.slice(1, t.indexOf("]"));
      out.push(
        <a key={k++} href={m[2]} target="_blank" rel="noreferrer" className="text-servo underline underline-offset-2">
          {label}
        </a>,
      );
    }
    last = m.index + t.length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}
