import type { Metadata } from "next";
import { CloudOff, Database, FileJson2, MessagesSquare } from "lucide-react";
import { DatasetViewer, DatasetViewerEntry } from "../components/DatasetViewer";
import { entriesOfType } from "@/lib/content";
import { MockDataBanner } from "../components/MockDataBanner";
import { allMock, anyMock } from "@/lib/content";

export const metadata: Metadata = { title: "Synthetic datasets" };

export default function DatasetsPage() {
  const allDatasets = entriesOfType("datasets");
  const datasets = allDatasets.filter((entry) => entry.dataset);
  // Declared but with no records resolved. Since these corpora are served from
  // the Hub with no local copy, an outage empties this page entirely - and
  // "no dataset found" reads as a bug rather than as the described gap it is.
  const unresolved = allDatasets.filter((entry) => !entry.dataset);
  const viewerEntries: DatasetViewerEntry[] = datasets.map((entry) => ({
    id: entry.id,
    title: entry.title,
    summary: entry.summary,
    status: entry.status,
    tags: entry.tags,
    dataset: entry.dataset,
    dataset_id: entry.dataset_id,
    dataset_version: entry.dataset_version,
    training_objective: entry.training_objective,
    generator_model: entry.generator_model,
    mock: entry.mock,
  }));
  const records = datasets.reduce(
    (sum, entry) => sum + (entry.dataset?.record_count || 0),
    0,
  );

  return (
    <main className="page-container inner-page datasets-page">
      {anyMock(datasets) && (
        <MockDataBanner scope={allMock(datasets) ? "all" : "some"} />
      )}
      <header className="page-heading compact-heading">
        <div>
          <span className="eyebrow">Training-data inspection</span>
          <h1>Synthetic datasets</h1>
          <p>
            Inspect SFT/AFT records as conversations, without losing the JSONL,
            split, construction metadata, or quality signals underneath.
          </p>
        </div>
        <div className="heading-stat">
          <Database size={19} /><strong>{datasets.length}</strong><span>datasets</span>
        </div>
        <div className="heading-stat">
          <MessagesSquare size={19} /><strong>{records}</strong><span>dialogues</span>
        </div>
        <div className="heading-stat">
          <FileJson2 size={19} /><strong>JSONL</strong><span>source format</span>
        </div>
      </header>

      {unresolved.map((entry) => (
        <div className="empty-state transcript-error" key={entry.id}>
          <CloudOff size={16} />
          <span>
            <strong>{entry.title}</strong> could not be loaded
            {entry.hf ? ` from ${entry.hf.repo_id}` : ""}.
            {entry.hf?.message ? ` ${entry.hf.message}.` : ""} The corpus is
            served from Hugging Face and is not mirrored in this repository, so
            there is nothing to fall back to. The dataset&rsquo;s own page still
            has its description and provenance.
          </span>
        </div>
      ))}

      {datasets.length > 0 && <DatasetViewer datasets={viewerEntries} />}
    </main>
  );
}

