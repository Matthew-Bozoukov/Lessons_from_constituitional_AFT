import type { Metadata } from "next";
import { Database, FileJson2, MessagesSquare } from "lucide-react";
import { DatasetViewer, DatasetViewerEntry } from "../components/DatasetViewer";
import { entriesOfType } from "@/lib/content";
import { MockDataBanner } from "../components/MockDataBanner";
import { allMock, anyMock } from "@/lib/content";

export const metadata: Metadata = { title: "Synthetic datasets" };

export default function DatasetsPage() {
  const datasets = entriesOfType("datasets").filter((entry) => entry.dataset);
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
      <DatasetViewer datasets={viewerEntries} />
    </main>
  );
}

