import { useEffect, useRef, useState } from "react";
import { LayerStack } from "./LayerStack";
import { Plate } from "./Plate";
import { STEPS } from "./steps";

/** Scroll timeline (fraction of the pinned scroll range). */
const EXPLODE_END = 0.14;
const STEP_START = 0.17;
const PER_STEP = (1 - STEP_START) / STEPS.length;

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));
const smooth = (t: number) => t * t * (3 - 2 * t);

function stepAt(p: number) {
  if (p < STEP_START) return -1;
  return Math.min(STEPS.length - 1, Math.floor((p - STEP_START) / PER_STEP));
}

type Mode = "pinned" | "static" | "mobile";

function readMode(): Mode {
  if (typeof window === "undefined") return "pinned";
  if (!window.matchMedia("(min-width: 1024px) and (min-height: 600px)").matches) return "mobile";
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "static" : "pinned";
}

function useMode(): Mode {
  const [mode, setMode] = useState<Mode>(readMode);
  useEffect(() => {
    const qs = [window.matchMedia("(min-width: 1024px) and (min-height: 600px)"), window.matchMedia("(prefers-reduced-motion: reduce)")];
    const on = () => setMode(readMode());
    qs.forEach((q) => q.addEventListener("change", on));
    return () => qs.forEach((q) => q.removeEventListener("change", on));
  }, []);
  return mode;
}

function Heading() {
  return (
    <>
      <div className="kn-label kn-label-rule">
        <span className="text-[#171717]">FIG.02</span>
        <span>How it works</span>
      </div>
      <h2 id="platform-title" className="kn-display mt-6 text-[clamp(2.2rem,3.5vw,3.5rem)] text-[#171717]">
        Seven layers,
        <br />
        <span className="kn-second">one training loop.</span>
      </h2>
    </>
  );
}

function Tag({ children, on }: { children: string; on?: boolean }) {
  return (
    <span
      className={`inline-flex h-6 items-center rounded-full border px-2.5 kn-mono text-[10px] uppercase tracking-[0.14em] transition-colors ${
        on ? "border-[#2F4F46] text-[#2F4F46]" : "border-[#E6E6E6] text-[#858585]"
      }`}
    >
      {children}
    </span>
  );
}

/** FIG.02: the scroll-driven exploded stack. Section id stays "platform" for the hero's Explore link. */
export function LayersSection() {
  const mode = useMode();
  return (
    <section id="platform" aria-labelledby="platform-title" className="relative scroll-mt-0 border-t border-[#E6E6E6]">
      {mode === "pinned" ? <Pinned /> : mode === "static" ? <StaticExploded /> : <MobileStack />}
    </section>
  );
}

function Pinned() {
  const ref = useRef<HTMLDivElement>(null);
  const [p, setP] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    let cur = -1;
    const tick = () => {
      const r = el.getBoundingClientRect();
      const range = Math.max(1, r.height - window.innerHeight);
      const target = clamp01(-r.top / range);
      const next = cur < 0 ? target : cur + (target - cur) * 0.16;
      const settled = Math.abs(target - next) < 0.0005 ? target : next;
      if (settled !== cur) {
        cur = settled;
        setP(settled);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const explode = smooth(clamp01(p / EXPLODE_END));
  const active = stepAt(p);
  const focus = (p - STEP_START) / PER_STEP;
  const mix = smooth(clamp01((p - EXPLODE_END) / (STEP_START + PER_STEP * 0.3 - EXPLODE_END)));
  const s = active >= 0 ? STEPS[active] : null;

  const goTo = (i: number) => {
    const el = ref.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY;
    const range = el.offsetHeight - window.innerHeight;
    window.scrollTo({ top: top + range * (STEP_START + PER_STEP * (i + 0.5)), behavior: "smooth" });
  };

  return (
    <div ref={ref} style={{ height: "760vh" }} data-testid="layers-pinned">
      <div className="sticky top-0 flex h-[100svh] overflow-hidden">
        <div className="relative z-10 flex w-[40%] shrink-0 flex-col justify-center py-16 pl-[clamp(2.5rem,5.5vw,6rem)] pr-6">
          <Heading />

          <ol className="mt-10 max-w-[22rem] border-t border-[#E6E6E6]">
            {STEPS.map((st, i) => {
              const on = i === active;
              const done = active > i;
              return (
                <li key={st.n} className="border-b border-[#E6E6E6]">
                  <button
                    type="button"
                    onClick={() => goTo(i)}
                    aria-current={on ? "step" : undefined}
                    className="group flex h-8 w-full items-center gap-4 text-left kn-mono text-[10.5px] uppercase tracking-[0.18em]"
                  >
                    <span
                      className="h-px transition-all duration-500 ease-out"
                      style={{ width: on ? 28 : 10, background: on ? "#2F4F46" : done ? "#171717" : "#BDBDBD" }}
                    />
                    <span className={on ? "text-[#2F4F46]" : done ? "text-[#171717]" : "text-[#858585] group-hover:text-[#171717]"}>
                      {st.n}
                    </span>
                    <span className={on ? "text-[#171717]" : done ? "text-[#171717]" : "text-[#858585] group-hover:text-[#171717]"}>
                      {st.label}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>

          <div className="mt-8 min-h-[150px] max-w-[26rem]" aria-live="polite">
            {s ? (
              <div key={s.n} className="kn-panel">
                <div className="flex items-center gap-3">
                  <span className="kn-mono text-[10.5px] uppercase tracking-[0.18em] text-[#171717]">
                    Step {s.n} <span className="text-[#858585]">/ 07</span>
                  </span>
                  <Tag on>{s.tag}</Tag>
                </div>
                <p className="mt-4 kn-sans text-[16px] leading-[1.6] text-[#3a3a3a]">{s.body}</p>
              </div>
            ) : (
              <p className="kn-sans text-[15px] leading-[1.6] text-[#858585]">
                Each layer is one step an agent runs through Cortex, from describing the arm to handing over a policy. Keep
                scrolling to take the loop apart.
              </p>
            )}
          </div>
        </div>

        <div className="relative flex min-w-0 flex-1 items-center justify-center pr-[clamp(1.5rem,3vw,3.5rem)]">
          <LayerStack explode={explode} active={active} focus={focus} mix={mix} className="h-[calc(100svh-120px)] max-h-[860px] w-auto max-w-full" />
        </div>

        {/* Scroll readout */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center gap-4 px-[clamp(2.5rem,5.5vw,6rem)] pb-6 kn-mono text-[10px] uppercase tracking-[0.16em] text-[#858585]">
          <span>{active < 0 ? (explode < 0.98 ? "Separating layers" : "Exploded view") : `Layer ${STEPS[active].n} of 07`}</span>
          <span className="relative h-px flex-1 bg-[#E6E6E6]">
            <span className="absolute inset-y-0 left-0 bg-[#171717]" style={{ width: `${p * 100}%` }} />
          </span>
          <span className="tabular-nums text-[#171717]">{p.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

/** Reduced motion on desktop: the stack is shown already exploded, with every step written out. */
function StaticExploded() {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-10 py-24 pl-[clamp(2.5rem,5.5vw,6rem)] pr-[clamp(1.5rem,3vw,3.5rem)]">
      <div>
        <Heading />
        <ol className="mt-12 border-t border-[#E6E6E6]">
          {STEPS.map((s) => (
            <li key={s.n} className="border-b border-[#E6E6E6] py-5">
              <div className="flex items-center gap-3 kn-mono text-[10.5px] uppercase tracking-[0.18em]">
                <span className="text-[#171717]">{s.n}</span>
                <span className="text-[#171717]">{s.label}</span>
                <Tag>{s.tag}</Tag>
              </div>
              <p className="mt-2 max-w-[30rem] kn-sans text-[14.5px] leading-[1.6] text-[#555]">{s.body}</p>
            </li>
          ))}
        </ol>
      </div>
      <div className="sticky top-6 self-start">
        <LayerStack explode={1} active={-1} uniformGap={200} className="h-auto w-full" />
      </div>
    </div>
  );
}

/** Phones and short screens: a plain vertical stack, one plate per step, joined by construction lines. */
function MobileStack() {
  const [active, setActive] = useState(-1);
  const refs = useRef<(HTMLLIElement | null)[]>([]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(Number((e.target as HTMLElement).dataset.i));
      },
      { rootMargin: "-45% 0px -45% 0px" },
    );
    refs.current.forEach((el) => el && io.observe(el));
    return () => io.disconnect();
  }, []);

  return (
    <div className="px-6 py-20 sm:px-10">
      <Heading />
      <p className="mt-5 max-w-[30rem] kn-sans text-[15px] leading-[1.6] text-[#555]">
        Each layer is one step an agent runs through Cortex, from describing the arm to handing over a policy.
      </p>
      <ol className="mt-10">
        {STEPS.map((s, i) => (
          <li key={s.n} ref={(el) => void (refs.current[i] = el)} data-i={i} className="relative">
            {i > 0 && <div aria-hidden className="mx-auto h-8 w-px border-l border-dashed border-[#BDBDBD]" />}
            <svg viewBox="-205 -118 410 250" className="mx-auto block w-full max-w-[300px]" aria-hidden>
              <Plate glyph={s.glyph} tone={active === i ? "active" : "idle"} />
            </svg>
            <div className="mt-4 flex flex-wrap items-center gap-3 kn-mono text-[10.5px] uppercase tracking-[0.18em]">
              <span className={active === i ? "text-[#2F4F46]" : "text-[#171717]"}>{s.n}</span>
              <span className="text-[#171717]">{s.label}</span>
              <Tag on={active === i}>{s.tag}</Tag>
            </div>
            <p className="mt-2 kn-sans text-[14.5px] leading-[1.6] text-[#555]">{s.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
