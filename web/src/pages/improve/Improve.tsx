import { useSearchParams } from "react-router-dom";
import { Segmented } from "../../components/ui";
import { RecommendationsTab } from "./Recommendations";
import { DatasetsTab } from "./Datasets";
import { TrainingTab } from "./Training";
import { EvaluationsTab } from "./Evaluations";

const TABS = ["recommendations", "datasets", "training", "evaluations"] as const;
type Tab = (typeof TABS)[number];

export function ImprovePage() {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab");
  const tab: Tab = (TABS as readonly string[]).includes(raw ?? "") ? (raw as Tab) : "recommendations";
  return (
    <div className="space-y-4 px-5 py-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="kn-display text-[36px] leading-tight">Improve</h1>
        </div>
        <Segmented
          value={tab}
          onChange={(t) => setParams({ tab: t }, { replace: true })}
          ariaLabel="Improve section"
          options={TABS.map((t) => ({ value: t, label: t }))}
        />
      </div>
      {tab === "recommendations" && <RecommendationsTab />}
      {tab === "datasets" && <DatasetsTab />}
      {tab === "training" && <TrainingTab />}
      {tab === "evaluations" && <EvaluationsTab />}
    </div>
  );
}
