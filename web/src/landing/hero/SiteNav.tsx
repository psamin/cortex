import { Link } from "react-router-dom";
import { BRAND, STUDIO_ENTRY } from "../brand";
import { scrollToSection } from "../scroll";

const LINKS: [string, string][] = [
  ["How it works", "platform"],
  ["Agent loop", "loop"],
  ["Built on", "stack"],
  ["Safety", "safety"],
];

/** Minimal top navigation. Section links scroll in-page; account actions open the studio. */
export function SiteNav() {
  return (
    <header className="absolute inset-x-0 top-0 z-30">
      <nav
        aria-label="Primary"
        className="mx-auto flex h-[72px] items-center justify-between px-6 sm:px-10 lg:px-[clamp(2.5rem,5.5vw,6rem)]"
      >
        <Link to="/" className="flex items-center gap-2.5 text-[#171717]" aria-label={`${BRAND.name} home`}>
          <BrandMark />
          <span className="kn-sans text-[15px] font-medium tracking-[-0.01em]">{BRAND.name}</span>
        </Link>
        <ul className="hidden items-center gap-9 kn-sans text-[13px] text-[#3a3a3a] md:flex">
          {LINKS.map(([l, id]) => (
            <li key={id}>
              <a
                href={`#${id}`}
                onClick={(e) => scrollToSection(e, id)}
                className="transition-colors hover:text-[#171717] focus-visible:underline"
              >
                {l}
              </a>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-5">
          <Link
            to={STUDIO_ENTRY}
            className="inline-flex h-9 items-center rounded-full bg-[#171717] px-4 kn-sans text-[12.5px] font-medium text-white hover:bg-black"
            data-testid="cta-access"
          >
            Open studio
          </Link>
        </div>
      </nav>
    </header>
  );
}

export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden>
      <path d="M11 1.5 19.5 6.4v9.2L11 20.5 2.5 15.6V6.4z" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path d="M11 11 19.5 6.4M11 11v9.5M11 11 2.5 6.4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <path d="M11 1.5 19.5 6.4 11 11 2.5 6.4z" fill="currentColor" />
    </svg>
  );
}
