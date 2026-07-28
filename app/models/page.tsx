import type { Metadata } from "next";
import { EntryCard } from "../components/EntryCard";
import { MetricTiles } from "../components/MetricTiles";
import { modelsInCorpus, humanize } from "@/lib/content";

export const metadata: Metadata = { title: "Models" };

export default function ModelsPage() {
  const models = modelsInCorpus();
  return (
    <main className="page-container inner-page">
      <header className="page-heading">
        <div>
          <span className="eyebrow">Generated dossiers</span>
          <h1>Model lineages</h1>
          <p>
            Model pages are projections of logs, evals, and findings. They
            preserve branching checkpoints rather than imposing one universal
            training pipeline.
          </p>
        </div>
      </header>

      <div className="model-list">
        {models.map(({ id, entries }) => {
          const evals = entries.filter((entry) => entry.type === "evals");
          const latest = evals[0];
          const stages = [...new Set(entries.map((entry) => entry.training_stage).filter(Boolean))];
          return (
            <section className="model-dossier" key={id}>
              <div className="model-header">
                <div>
                  <span className="model-glyph" aria-hidden="true">∷</span>
                  <div><span className="eyebrow">Model family</span><h2>{id}</h2></div>
                </div>
                <div className="model-counts">
                  <span><strong>{entries.length}</strong> linked records</span>
                  <span><strong>{evals.length}</strong> eval runs</span>
                </div>
              </div>

              <div className="lineage-bar">
                <span className="lineage-mark" aria-hidden="true">BRANCH</span>
                {stages.map((stage, index) => (
                  <span key={stage}>
                    {index > 0 && <i />}
                    <code>{humanize(String(stage))}</code>
                  </span>
                ))}
              </div>

              {latest && <MetricTiles metrics={latest.metrics} limit={3} />}

              <div className="model-records">
                {entries.slice(0, 3).map((entry) => (
                  <EntryCard compact entry={entry} key={entry.id} />
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </main>
  );
}
