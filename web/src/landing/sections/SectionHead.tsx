import type { ReactNode } from "react";

/** "——  SECTION TITLE" plus a two-tone Newsreader headline. */
export function SectionHead({ label, first, second, titleId, children }: { label: string; first: string; second: string; titleId?: string; children?: ReactNode }) {
  return (
    <div className="max-w-[44rem]">
      <div className="kn-label kn-label-rule">
        <span className="text-ink">{label}</span>
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
