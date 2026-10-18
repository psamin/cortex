import { useEffect } from "react";
import "@fontsource/newsreader/300.css";
import "@fontsource/newsreader/400.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./landing.css";
import { BRAND } from "./brand";
import { HeroCopy } from "./hero/HeroCopy";
import { HeroSection } from "./hero/HeroSection";
import { SiteNav } from "./hero/SiteNav";
import { LayersSection } from "./layers/LayersSection";
import { ConnectSection } from "./connect/ConnectSection";
import { AgentLoop } from "./sections/AgentLoop";
import { BuiltOn } from "./sections/BuiltOn";
import { Safety } from "./sections/Safety";
import { FinalCta, SiteFooter } from "./sections/FinalCta";
import { FaqSection } from "./faq/FaqSection";

/** Public home page: the arm hero, the exploded training loop, then connect, the agent loop, stack, safety, FAQ and CTA. */
export function LandingPage() {
  useEffect(() => {
    const prev = document.title;
    document.title = BRAND.title;
    return () => {
      document.title = prev;
    };
  }, []);

  return (
    <div className="kn-page min-h-full">
      <SiteNav />
      <main>
        <HeroSection copy={<HeroCopy />} />
        <LayersSection />
        <ConnectSection />
        <AgentLoop />
        <BuiltOn />
        <Safety />
        <FaqSection />
        <FinalCta />
      </main>
      <SiteFooter />
    </div>
  );
}

export default LandingPage;
