import { Link } from "react-router-dom";
import { BRAND, STUDIO_ENTRY } from "../brand";
import { BrandMark } from "../hero/SiteNav";
import { scrollToSection } from "../scroll";

const LINKS = [
  ["How it works", "platform"],
  ["Agent loop", "loop"],
  ["Built on", "stack"],
  ["Safety", "safety"],
];

/** FIG.06: the last ask, then the footer. */
export function FinalCta() {
  return (
    <>
      <section className="kn-gutter border-t border-[#E6E6E6] py-28 lg:py-36">
        <div>
          <div className="kn-label kn-label-rule">
            <span className="text-[#171717]">FIG.06</span>
            <span>Start</span>
          </div>
          <h2 className="kn-display mt-6 max-w-[48rem] text-[clamp(2.4rem,4.6vw,4.6rem)] text-[#171717]">
            Teach your arm a task.
            <br />
            <span className="kn-second">Start in simulation.</span>
          </h2>
          <div className="mt-10 flex flex-wrap items-center gap-3">
            <Link
              to={STUDIO_ENTRY}
              className="inline-flex h-11 items-center rounded-full bg-[#171717] px-6 kn-sans text-[13.5px] font-medium text-white transition-colors hover:bg-black"
            >
              Start Building
            </Link>
            <a
              href="#platform"
              onClick={(e) => scrollToSection(e, "platform")}
              className="inline-flex h-11 items-center gap-2 rounded-full border border-[#DADADA] bg-white px-6 kn-sans text-[13.5px] font-medium text-[#171717] transition-colors hover:border-[#171717]"
            >
              See how it works
              <span aria-hidden className="text-[#858585]">
                →
              </span>
            </a>
          </div>
          <p className="mt-10 kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585]">
          MVP · the whole loop runs in MuJoCo · the real-arm path is supervised
        </p>
        </div>
      </section>

      <footer className="kn-gutter border-t border-[#E6E6E6] py-10">
        <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap items-center gap-2.5 text-[#171717]">
            <BrandMark size={18} />
            <span className="kn-sans text-[14px] font-medium">{BRAND.name}</span>
            <span className="ml-3 kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585]">Physical AI infrastructure</span>
          </div>
          <nav aria-label="Footer" className="flex flex-wrap gap-x-7 gap-y-3 kn-sans text-[13px] text-[#555]">
            {LINKS.map(([t, id]) => (
              <a key={id} href={`#${id}`} onClick={(e) => scrollToSection(e, id)} className="hover:text-[#171717]">
                {t}
              </a>
            ))}
            <Link to={STUDIO_ENTRY} className="text-[#171717] hover:underline">
              Open the studio →
            </Link>
          </nav>
        </div>
      </footer>
    </>
  );
}
