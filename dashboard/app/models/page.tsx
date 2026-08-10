import type { Metadata } from "next";
import { EntryCard } from "../components/EntryCard";
import { MetricTiles } from "../components/MetricTiles";
import { allMock, anyMock, byKindThenDate, entryKind, modelsInCorpus, humanize } from "@/lib/content";
import { MockBadge, MockDataBanner } from "../components/MockDataBanner";

export const metadata: Metadata = { title: "Models" };

export default function ModelsPage() {
  // Ordered by written evidence, not alphabetically. A model with three
  // write-ups behind it and a model with one bare artifact link sorted
  // adjacently, so the page opened on whichever name happened to sort first.
  const models = modelsInCorpus()
    .map((model) => ({
      ...model,
      entries: [...model.entries].sort(byKindThenDate),
      written: model.entries.filter((entry) => entryKind(entry) === "written").length,
    }))
    .sort((a, b) => b.written - a.written || b.entries.length - a.entries.length);
  // A dossier is a projection of its linked entries, so a model whose entire
  // record set is fabricated is itself a fixture - including its model id.
  const someMock = models.some(({ entries }) => anyMock(entries));
  return (
    <main className="page-container inner-page">
      {someMock && <MockDataBanner scope="some" />}
      <header className="page-heading">
        <div>
          <span className="eyebrow">Generated dossiers</span>
          <h1>Model lineages</h1>
          <p>
            Everything in the corpus that mentions a given model or adapter,
            gathered in one place. Nothing here is written directly — each
            dossier is assembled from the logs, evals and findings that name
            that model, so it is a view of the record rather than a claim.
          </p>
        </div>
      </header>

      <div className="model-list">
        {models.map(({ id, entries, written }) => {
          const evals = entries.filter((entry) => entry.type === "evals");
          const latest = evals[0];
          const stages = [...new Set(entries.map((entry) => entry.training_stage).filter(Boolean))];
          return (
            <section className="model-dossier" key={id}>
              <div className="model-header">
                <div>
                  <span className="model-glyph" aria-hidden="true">∷</span>
                  <div>
                    <span className="eyebrow">Model family</span>
                    <h2>{id}</h2>
                    {allMock(entries) && <MockBadge />}
                  </div>
                </div>
                <div className="model-counts">
                  <span><strong>{entries.length}</strong> linked records</span>
                  <span><strong>{evals.length}</strong> eval runs</span>
                  {/* Says outright how much of the dossier anyone has actually
                      written up, so a wall of linked artifacts is not mistaken
                      for a body of analysis. */}
                  <span><strong>{written}</strong> written up</span>
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
