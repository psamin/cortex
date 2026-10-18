import type { ReactNode } from "react";

// Shared look for the Cortex views: the studio's monochrome technical-drawing style, light and dark.
export const CSS = `
.cx{--fg:#16181d;--muted:#6b7079;--line:#d9dce1;--bg:#fff;--soft:#f5f6f8;--ok:#1f7a4d;--bad:#b4362f;--accent:#16181d;
  font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);background:var(--bg);
  padding:14px 16px;border-radius:10px}
@media (prefers-color-scheme:dark){.cx{--fg:#eceef1;--muted:#9aa0a9;--line:#30343b;--bg:#15171b;--soft:#1d2025;--ok:#4fbf86;--bad:#ef6a60;--accent:#eceef1}}
.cx *{box-sizing:border-box}
.cx .eyebrow{font:11px/1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  display:flex;align-items:center;gap:8px;margin-bottom:8px}
.cx .eyebrow:before{content:"";width:22px;height:1px;background:var(--muted)}
.cx h2{font-size:17px;font-weight:600;margin:0 0 10px}
.cx .mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.cx .muted{color:var(--muted)}
.cx .row{display:flex;gap:16px;flex-wrap:wrap}
.cx .stat{min-width:84px}
.cx .stat b{display:block;font-size:20px;font-weight:600}
.cx .stat span{font:11px ui-monospace,Consolas,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.cx table{width:100%;border-collapse:collapse;font-size:12px}
.cx td,.cx th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.cx th{font:11px ui-monospace,Consolas,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:400}
.cx .pill{display:inline-block;padding:1px 8px;border:1px solid var(--line);border-radius:99px;font-size:11px}
.cx .ok{color:var(--ok)} .cx .bad{color:var(--bad)}
.cx .btn{appearance:none;border:1px solid var(--accent);background:var(--accent);color:var(--bg);border-radius:8px;
  padding:8px 14px;font:600 13px system-ui,sans-serif;cursor:pointer}
.cx .btn[disabled]{opacity:.5;cursor:default}
.cx .btn.ghost{background:transparent;color:var(--fg);border-color:var(--line)}
.cx a{color:inherit}
.cx .frame{width:100%;max-width:360px;aspect-ratio:1;border:1px solid var(--line);border-radius:8px;object-fit:cover;background:var(--soft)}
.cx .bar{height:8px;background:var(--soft);border-radius:99px;position:relative;overflow:hidden;min-width:120px}
.cx .bar i{position:absolute;top:0;bottom:0;background:var(--fg);opacity:.18}
.cx .bar b{position:absolute;top:0;bottom:0;width:2px;background:var(--fg)}
`;

export function Shell({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children: ReactNode }) {
  return (
    <div className="cx">
      <style>{CSS}</style>
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {children}
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="stat">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="cx">
      <style>{CSS}</style>
      <div className="muted">{children}</div>
    </div>
  );
}

export const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);
