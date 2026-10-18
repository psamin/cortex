import { Component, lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
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

// Public landing page at /. Lazy so the studio bundle never pulls in three.js or gsap.
const LandingPage = lazy(() => import("./landing/LandingPage"));

/** If the landing page can't load or crashes (e.g. its 3D deps aren't installed), go to the studio
 *  instead of showing a blank page: the landing page must never block the product. */
class LandingFallback extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <Navigate to="/studio" replace /> : this.props.children;
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
