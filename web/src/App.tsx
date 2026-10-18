import { lazy, Suspense } from "react";
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

// Public landing page at /. Lazy so the studio bundle never pulls in three.js or gsap.
const LandingPage = lazy(() => import("./landing/LandingPage"));

export function App() {
  return (
    <Routes>
      <Route
        index
        element={
          <Suspense fallback={null}>
            <LandingPage />
          </Suspense>
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
        </Route>
        <Route path="*" element={<Navigate to="/studio" replace />} />
      </Route>
    </Routes>
  );
}
