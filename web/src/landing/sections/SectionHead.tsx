import type { ReactNode } from "react";

/** "—— FIG.0n  LABEL" plus a two-tone Newsreader headline. */
export function SectionHead({ fig, label, first, second, titleId, children }: { fig: string; label: string; first: string; second: string; titleId?: string; children?: ReactNode }) {
  return (
    <div className="max-w-[44rem]">
      <div className="kn-label kn-label-rule">
        <span className="text-ink">{fig}</span>
        <span>{label}</span>
      </div>
      <h2 id={titleId} className="kn-display mt-6 text-[clamp(2rem,3.3vw,3.25rem)] text-ink">
        {first}
        <br />
        <span className="kn-second">{second}</span>
      </h2>
      {children && <div className="mt-6 max-w-[34rem] kn-sans text-[15px] leading-[1.65] text-ink-2">{children}</div>}
    </div>
  );
}
