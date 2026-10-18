import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { headTags, PAGE_TITLE, staticMain } from "./src/landing/staticHtml";

/** Fills index.html's agent-readable parts (meta tags, JSON-LD, no-JS landing copy) from the landing page's sources. */
function agentReadableHtml(site: string): Plugin {
  return {
      name: "cortex-agent-readable-html",
      transformIndexHtml(html) {
        return html
          .replace("<!--app-title-->", PAGE_TITLE)
          .replace("<!--agent-head-->", headTags(site))
          .replace("<!--landing-static-->", staticMain());
      },
    };
  }

export default defineConfig(({ mode }) => {
  /** Public origin for absolute URLs (canonical, og:image, JSON-LD). Set CORTEX_SITE_URL when deploying. */
  const site = (loadEnv(mode, ".", "CORTEX_").CORTEX_SITE_URL || "http://localhost:5173").replace(/\/$/, "");
  return {
    plugins: [react(), agentReadableHtml(site)],
    server: { port: 5173, strictPort: true },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            react: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"],
            charts: ["recharts"],
          },
        },
      },
    },
  };
});
