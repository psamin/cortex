import { useEffect } from "react";
import { Link } from "react-router-dom";
import "@fontsource/newsreader/300.css";
import "@fontsource/newsreader/400.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./landing.css";
import { BRAND, STUDIO_ENTRY } from "./brand";
import { HeroCopy } from "./hero/HeroCopy";
import { HeroSection } from "./hero/HeroSection";
import { BrandMark, SiteNav } from "./hero/SiteNav";

const STEPS = [
  {
    n: "01",
    id: "simulation",
    title: "Simulate",
    body: "Describe the task in plain language. The agent builds a MuJoCo scene of your workcell, with the real arm, objects, zones and obstacles, and checks that it is reachable.",
  },
  {
    n: "02",
    id: "train",
    title: "Train",
    body: "Collect scripted and agent-driven demonstrations in simulation, record every episode with its provenance, and train policies on the result.",
  },
  {
    n: "03",
    id: "deploy",
    title: "Deploy",
    body: "Analyze failures, compare runs in the lineage graph, and promote the policies that hold up to the robot on your bench.",
  },
];

const FOOTER = [
  { id: "research", title: "Research", body: "Policy evaluation, failure analysis and data lineage, built into every run." },
  { id: "docs", title: "Docs", body: "The studio, the REST API and the MCP server share one set of tools." },
  { id: "company", title: "Company", body: `${BRAND.name} builds infrastructure for physical AI.` },
];

/** Public home page: the isometric arm hero plus a short platform overview. */
export function LandingPage() {
  useEffect(() => {
    const prev = document.title;
    document.title = `${BRAND.name}: ${BRAND.tagline}`;
    return () => {
      document.title = prev;
    };
  }, []);

  return (
    <div className="kn-page min-h-full">
      <main>
        <SiteNav />
        <HeroSection copy={<HeroCopy />} />

        <section id="platform" className="scroll-mt-6 border-t border-[#e6e6e6] px-6 py-20 sm:px-10 lg:px-[clamp(2.5rem,5.5vw,6rem)]">
          <div className="flex items-center gap-3 kn-mono text-[10.5px] uppercase tracking-[0.18em] text-[#858585]">
            <span className="h-px w-9 bg-[#171717]" />
            <span className="text-[#171717]">FIG.02</span>
            <span>Platform</span>
          </div>
          <h2 className="mt-6 max-w-[40rem] kn-serif text-[clamp(2rem,3.4vw,3.2rem)] font-light leading-[1.08] tracking-[-0.02em]">
            One loop, <span className="text-[#A6A6A6]">from scene to policy.</span>
          </h2>
          <ol className="mt-12 grid gap-10 border-t border-[#e6e6e6] pt-8 md:grid-cols-3">
            {STEPS.map((s) => (
              <li key={s.n} id={s.id} className="scroll-mt-6">
                <div className="flex gap-2 kn-mono text-[10px] uppercase tracking-[0.12em] text-[#858585]">
                  <span className="text-[#171717]">{s.n}</span>
                  <span>{s.title}</span>
                </div>
                <p className="mt-4 max-w-[26rem] kn-sans text-[14.5px] leading-[1.65] text-[#555]">{s.body}</p>
              </li>
            ))}
          </ol>
          <div className="mt-12">
            <Link
              to={STUDIO_ENTRY}
              className="inline-flex h-11 items-center gap-2 rounded-full bg-[#171717] px-6 kn-sans text-[13.5px] font-medium text-white transition-colors hover:bg-black"
            >
              Open the studio
              <span aria-hidden className="text-[#A6A6A6]">→</span>
            </Link>
          </div>
        </section>
      </main>

      <footer className="border-t border-[#e6e6e6] px-6 py-12 sm:px-10 lg:px-[clamp(2.5rem,5.5vw,6rem)]">
        <div className="grid gap-8 md:grid-cols-4">
          <div className="flex items-center gap-2.5 text-[#171717]">
            <BrandMark size={18} />
            <span className="kn-sans text-[14px] font-medium">{BRAND.name}</span>
          </div>
          {FOOTER.map((f) => (
            <div key={f.id} id={f.id} className="scroll-mt-6">
              <div className="kn-mono text-[10px] uppercase tracking-[0.12em] text-[#171717]">{f.title}</div>
              <p className="mt-2 kn-sans text-[13px] leading-[1.6] text-[#858585]">{f.body}</p>
            </div>
          ))}
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
