import { SectionHead } from "../sections/SectionHead";
import FAQ from "./faq.json";

/** Renders `cortex.foo` and the install command as code inside an answer. */
function Answer({ text }: { text: string }) {
  const parts = text.split(/(claude mcp add [^\s]+ [^\s]+ [^\s]+ [^\s]+|cortex\.[a-z_*]+|https?:\/\/[^\s,]+[^\s,.])/g);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 ? (
          <code key={i} className="kn-mono text-[12.5px] text-ink [overflow-wrap:anywhere]">
            {p}
          </code>
        ) : (
          p
        ),
      )}
    </>
  );
}

/**
 * the questions people and agents actually ask. Native <details>, so every answer is in the HTML
 * (and in the FAQPage JSON-LD built from the same faq.json) even while collapsed.
 */
export function FaqSection() {
  return (
    <section id="faq" aria-labelledby="faq-title" className="kn-gutter border-t border-line py-24 lg:py-32">
      <div className="grid gap-12 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <SectionHead titleId="faq-title" label="Questions" first="Asked often." second="Answered plainly." />
        <div className="border-t border-ink">
          {FAQ.items.map((f, i) => (
            <details key={f.id} id={`faq-${f.id}`} open={i === 0} className="group border-b border-line">
              <summary className="flex cursor-pointer list-none items-baseline justify-between gap-6 py-5 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-4 focus-visible:outline-ink [&::-webkit-details-marker]:hidden">
                <h3 className="kn-sans text-[16px] font-medium leading-[1.4] text-ink">{f.q}</h3>
                <span aria-hidden className="kn-mono text-[13px] text-ink-3 transition-transform duration-200 group-open:rotate-45">
                  +
                </span>
              </summary>
              <p className="max-w-[40rem] pb-6 pr-8 kn-sans text-[14.5px] leading-[1.7] text-ink-2">
                <Answer text={f.a} />
              </p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
