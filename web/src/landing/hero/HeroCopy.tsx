import { Link } from "react-router-dom";
import { BRAND, STUDIO_ENTRY } from "../brand";
import { scrollToSection } from "../scroll";

/** Plain DOM: renders immediately, independent of the 3D scene. */
export function HeroCopy() {
  return (
    <div className="flex h-full flex-col px-6 pb-4 pt-24 sm:px-10 sm:pt-28 lg:justify-center lg:pb-16 lg:pl-[clamp(2.5rem,5.5vw,6rem)] lg:pr-0 lg:pt-24">
      <div className="reveal flex items-center gap-3 kn-mono text-[10.5px] uppercase tracking-[0.18em] text-[#858585]" style={{ animationDelay: "80ms" }}>
        <span className="h-px w-9 bg-[#171717]" />
        <span className="text-[#171717]">FIG.00</span>
        <span>Physical AI infrastructure</span>
      </div>

      <h1
        className="reveal mt-5 kn-serif sm:mt-7 text-[clamp(2.6rem,5.4vw,5.1rem)] font-light leading-[1.02] tracking-[-0.022em] text-[#171717]"
        style={{ animationDelay: "160ms" }}
      >
        Build intelligence
        <br />
        <span className="text-[#A6A6A6]">for the physical world.</span>
      </h1>

      <p
        className="reveal mt-5 max-w-[30rem] sm:mt-7 kn-sans text-[15px] leading-[1.65] text-[#555]"
        style={{ animationDelay: "260ms" }}
      >
        {BRAND.description}
      </p>

      <div className="reveal mt-7 flex flex-wrap sm:mt-9 items-center gap-3" style={{ animationDelay: "340ms" }}>
        <Link
          to={STUDIO_ENTRY}
          className="inline-flex h-11 items-center rounded-full bg-[#171717] px-6 kn-sans text-[13.5px] font-medium text-white transition-colors hover:bg-black focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#171717]"
          data-testid="cta-primary"
        >
          Start Building
        </Link>
        <a
          href="#platform"
          onClick={(e) => scrollToSection(e, "platform")}
          className="inline-flex h-11 items-center gap-2 rounded-full border border-[#DADADA] bg-white px-6 kn-sans text-[13.5px] font-medium text-[#171717] transition-colors hover:border-[#171717] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#171717]"
          data-testid="cta-secondary"
        >
          Explore Platform
          <span aria-hidden className="text-[#858585]">→</span>
        </a>
      </div>

      <dl
        className="reveal mt-14 hidden max-w-[30rem] grid-cols-3 border-t border-[#e6e6e6] pt-5 kn-mono text-[10px] uppercase tracking-[0.12em] text-[#858585] lg:grid"
        style={{ animationDelay: "420ms" }}
      >
        {[
          ["01", "Simulate"],
          ["02", "Train"],
          ["03", "Deploy"],
        ].map(([n, t]) => (
          <div key={n} className="flex gap-2">
            <dt className="text-[#171717]">{n}</dt>
            <dd>{t}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
