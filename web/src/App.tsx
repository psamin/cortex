import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { ProjectLayout } from "./layout/ProjectLayout";
import { HomePage } from "./pages/Home";
import { NewProjectPage } from "./pages/NewProject";
import { PlanPage } from "./pages/Plan";
import { CollectPage } from "./pages/Collect";
import { AnalyzePage } from "./pages/Analyze";
import { ImprovePage } from "./pages/improve/Improve";
import { LineagePage } from "./pages/Lineage";

// Public home page (Kinetic landing). Lazy so the studio bundle never pulls in three.js for it.
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
        <Route path="new" element={<NewProjectPage />} />
        <Route path="p/:pid" element={<ProjectLayout />}>
          <Route index element={<PlanPage />} />
          <Route path="collect" element={<CollectPage />} />
          <Route path="analyze" element={<AnalyzePage />} />
          <Route path="improve" element={<ImprovePage />} />
          <Route path="graph" element={<LineagePage />} />
        </Route>
        <Route path="*" element={<Navigate to="/studio" replace />} />
      </Route>
    </Routes>
  );
}
