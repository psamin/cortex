import { Link } from "react-router-dom";
import { BRAND, STUDIO_ENTRY } from "../brand";
import { BrandMark } from "../hero/SiteNav";
import { scrollToSection } from "../scroll";

const LINKS = [
  ["How it works", "platform"],
  ["Connect", "connect"],
  ["Agent loop", "loop"],
  ["Built on", "stack"],
  ["Safety", "safety"],
  ["FAQ", "faq"],
];

/** The last ask, then the footer. */
export function FinalCta() {
  return (
    <section id="start" aria-labelledby="start-title" className="kn-gutter border-t border-line py-28 lg:py-36">
      <div>
        <div className="kn-label kn-label-rule">
          <span className="text-ink">Start</span>
        </div>
        <h2 id="start-title" className="kn-display mt-6 max-w-[48rem] text-[clamp(2.4rem,4.6vw,4.6rem)] text-ink">
          Give your agent a robot.
          <br />
          <span className="kn-second">It handles the training.</span>
        </h2>
        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Link
            to={STUDIO_ENTRY}
            className="inline-flex h-11 items-center rounded-full bg-ink px-6 kn-sans text-[13.5px] font-medium text-bench transition-colors hover:bg-ink-2"
          >
            Start Building
          </Link>
          <a
            href="#platform"
            onClick={(e) => scrollToSection(e, "platform")}
            className="inline-flex h-11 items-center gap-2 rounded-full border border-line-strong bg-bench px-6 kn-sans text-[13.5px] font-medium text-ink transition-colors hover:border-ink"
          >
            See how it works
            <span aria-hidden className="text-ink-3">
              →
            </span>
          </a>
        </div>
        <p className="mt-10 kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3">
          MVP · the loop runs in MuJoCo today · people approve spend and anything that moves hardware
        </p>
      </div>
    </section>
  );
}

/** Site footer: brand, section links and the way into the studio. */
export function SiteFooter() {
  return (
    <footer className="kn-gutter border-t border-line py-10">
      <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-2.5 text-ink">
          <BrandMark size={18} />
          <span className="kn-sans text-[14px] font-medium">{BRAND.name}</span>
          <span className="ml-3 kn-mono text-[10px] uppercase tracking-[0.16em] text-ink-3">
            The robot-training MCP agents want
          </span>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-7 gap-y-3 kn-sans text-[13px] text-ink-2">
          {LINKS.map(([t, id]) => (
            <a key={id} href={`#${id}`} onClick={(e) => scrollToSection(e, id)} className="hover:text-ink">
              {t}
            </a>
          ))}
          <Link to={STUDIO_ENTRY} className="text-ink hover:underline">
            Open the studio →
          </Link>
        </nav>
      </div>
    </footer>
  );
}
