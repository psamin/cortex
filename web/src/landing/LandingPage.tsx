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
import { FinalCta } from "./sections/FinalCta";

/** Public home page: the arm hero, the exploded training loop, then the agent loop, stack, safety and CTA. */
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
        <LayersSection />
        <ConnectSection />
        <AgentLoop />
        <BuiltOn />
        <Safety />
      </main>
      <FinalCta />
    </div>
  );
}

export default LandingPage;
