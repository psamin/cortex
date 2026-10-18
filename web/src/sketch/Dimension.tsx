import type { ReactNode } from "react";

/**
 * A dimension line with end ticks and a mono value centred on it (|——— 240 MM ———|).
 * Horizontal fills its container's width; vertical fills its height.
 */
export function Dimension({
  value,
  orientation = "h",
  className,
}: {
  value?: ReactNode;
  orientation?: "h" | "v";
  className?: string;
}) {
  return (
    <div className={["sk-dim", orientation === "v" ? "sk-dim-v" : "sk-dim-h", className].filter(Boolean).join(" ")} aria-hidden>
      <span className="sk-dim-tick" />
      <span className="sk-dim-line" />
      {value != null && <span className="sk-dim-value">{value}</span>}
      <span className="sk-dim-line" />
      <span className="sk-dim-tick" />
    </div>
  );
}
