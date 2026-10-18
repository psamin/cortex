import { Component, lazy, Suspense, type ReactNode } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { ProjectLayout } from "./layout/ProjectLayout";
import { HomePage } from "./pages/Home";
import { PlanPage } from "./pages/Plan";
import { CollectPage } from "./pages/Collect";
import { AnalyzePage } from "./pages/Analyze";
import { ImprovePage } from "./pages/improve/Improve";
import { ProjectChatPage } from "./chat/ProjectChatPage";
import { RobotPage } from "./pages/Robot";
import { SetupPage } from "./pages/Setup";

// Public landing page at /. Lazy so the studio bundle never pulls in its fonts or sections; the landing page in turn
// lazy-loads its 3D hero and falls back to a still image when three.js or WebGL is missing (see landing/hero/HeroSection).
// Only if the landing module itself cannot load do we send the visitor to the studio, so the task box stays reachable.
const LandingPage = lazy(() =>
  import("./landing/LandingPage").catch((e) => {
    if (import.meta.env.DEV) console.warn("[landing] landing page failed to load; opening the studio instead.", e);
    return { default: () => <Navigate to="/studio" replace /> };
  }),
);

/** If the landing page crashes while rendering, show a plain page with the way into the studio, never a blank screen. */
class LandingFallback extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="flex min-h-full flex-col items-start justify-center gap-4 bg-white px-6 py-24 text-[#171717] sm:px-10">
        <h1 className="text-2xl font-medium">Cortex</h1>
        <p className="max-w-md text-[15px] leading-relaxed text-[#555]">
          The landing page failed to render. The studio is unaffected.
        </p>
        <Link to="/studio" className="inline-flex h-11 items-center rounded-full bg-[#171717] px-6 text-[13.5px] font-medium text-white">
          Open studio
        </Link>
      </main>
    );
  }
}

export function App() {
  return (
    <Routes>
      <Route
        index
        element={
          <LandingFallback>
            <Suspense fallback={null}>
              <LandingPage />
            </Suspense>
          </LandingFallback>
        }
      />
      <Route element={<AppShell />}>
        <Route path="studio" element={<HomePage />} />
        <Route path="p/:pid" element={<ProjectLayout />}>
          <Route index element={<ProjectChatPage />} />
          <Route path="overview" element={<PlanPage />} />
          <Route path="collect" element={<CollectPage />} />
          <Route path="analyze" element={<AnalyzePage />} />
          <Route path="improve" element={<ImprovePage />} />
          <Route path="robot" element={<RobotPage />} />
          <Route path="setup" element={<SetupPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/studio" replace />} />
      </Route>
    </Routes>
  );
}
