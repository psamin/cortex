import type { Config } from "tailwindcss";

const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bench: v("bench"), // page canvas
        plate: v("plate"), // raised panel
        rail: v("rail"), // popover / hover surface
        well: v("well"), // inset inputs
        ink: { DEFAULT: v("ink"), 2: v("ink-2"), 3: v("ink-3"), 4: v("ink-4") },
        servo: { DEFAULT: v("servo"), ink: v("servo-ink") }, // the one accent
        pass: v("pass"),
        fault: v("fault"),
        caution: v("caution"),
        line: { DEFAULT: "var(--line)", soft: "var(--line-soft)", strong: "var(--line-strong)" },
      },
      borderColor: { line: "var(--line)", "line-soft": "var(--line-soft)", "line-strong": "var(--line-strong)" },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
        serif: ['"Newsreader"', '"Iowan Old Style"', '"Times New Roman"', "serif"],
      },
      fontSize: {
        "2xs": ["11px", "16px"],
        xs: ["12px", "16px"],
        sm: ["13px", "20px"],
        base: ["14px", "22px"],
        lg: ["16px", "24px"],
        xl: ["18px", "26px"],
        "2xl": ["22px", "28px"],
        "3xl": ["28px", "34px"],
      },
      borderRadius: { sm: "4px", DEFAULT: "6px", md: "8px", lg: "10px" },
      transitionTimingFunction: { out: "cubic-bezier(0.23, 1, 0.32, 1)" },
    },
  },
  plugins: [],
} satisfies Config;
