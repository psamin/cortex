/**
 * Shown when WebGL is unavailable: a 2x still captured from the real render
 * (regenerate with the snippet in README if the robot design changes).
 */
export function HeroFallback() {
  return (
    <div
      className="absolute inset-0 flex items-start justify-center pt-4 lg:items-center lg:justify-end lg:pr-[1.5%] lg:pt-12"
      data-testid="hero-fallback"
    >
      <img
        src="/robot-still.png"
        alt="Isometric line drawing of a six-axis robotic arm on a square base"
        width={820}
        height={610}
        className="h-auto w-[92%] max-w-[820px] lg:w-[57%] dark:invert"
        decoding="async"
        style={{
          maskImage: "radial-gradient(ellipse 72% 70% at 55% 50%, #000 62%, transparent 100%)",
          WebkitMaskImage: "radial-gradient(ellipse 72% 70% at 55% 50%, #000 62%, transparent 100%)",
        }}
      />
    </div>
  );
}
