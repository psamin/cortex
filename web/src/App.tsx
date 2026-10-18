import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { ProjectLayout } from "./layout/ProjectLayout";
import { HomePage } from "./pages/Home";
import { PlanPage } from "./pages/Plan";
import { CollectPage } from "./pages/Collect";
import { AnalyzePage } from "./pages/Analyze";
import { ImprovePage } from "./pages/improve/Improve";
import { ProjectChatPage } from "./chat/ProjectChatPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="p/:pid" element={<ProjectLayout />}>
          <Route index element={<ProjectChatPage />} />
          <Route path="overview" element={<PlanPage />} />
          <Route path="collect" element={<CollectPage />} />
          <Route path="analyze" element={<AnalyzePage />} />
          <Route path="improve" element={<ImprovePage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
