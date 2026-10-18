import { Moon, Sun } from "lucide-react";
import { cn } from "../lib/format";
import { useTheme } from "../lib/theme";

/** Light / dark switch. The same control in the landing nav and the studio header. */
export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className={cn("rounded-full p-1.5 text-ink-3 transition-colors hover:bg-rail hover:text-ink", className)}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      data-testid="theme-toggle"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}
