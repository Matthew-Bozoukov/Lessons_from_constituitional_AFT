import type { Metadata } from "next";
import Link from "next/link";
import { CloudOff, Database, Layers, MessagesSquare } from "lucide-react";
import { DatasetViewer, DatasetViewerEntry } from "../components/DatasetViewer";
import { entriesOfType } from "@/lib/content";
import { composition } from "@/lib/composition";

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
  // Only corpora that published a statistics sidecar state a record count. The
  // rest are read by byte range and never counted at build time, so this is a
  // sum over the ones that say - labelled as such rather than presented as the
  // size of the whole collection.
  const counted = datasets.filter((entry) => (entry.dataset?.record_count || 0) > 0);
  // Corpora whose `mixture_stats.json` actually states the blend, counted with
  // the same function the picker groups by. Counting entries that merely HAVE a
  // statistics file said 28 where the picker showed 19: nine of those files
  // carry a total and no `by_source`, so they name no blend at all.
  const withComposition = datasets.filter(
    (entry) =>
      composition(entry.dataset?.stats.categories, entry.dataset?.stats.categories_source)
        ?.constitutionShare != null,
  ).length;
  const records = counted.reduce((sum, entry) => sum + (entry.dataset?.record_count || 0), 0);

  return (
    <main className="page-container inner-page datasets-page">
      {/* No page-level fixture banner here, deliberately. This page renders ONE
          corpus at a time, so a collection-scoped "some entries are mock"
          warning would sit above a real corpus and cast doubt on it because a
          fixture exists elsewhere in the picker - the exact thing the banner is
          meant to prevent. The warning is raised inside the viewer, against the
          corpus actually being shown, and the picker marks the fixture row. */}
      <header className="page-heading compact-heading">
        <div>
          <span className="eyebrow">Training-data inspection</span>
          <h1>Synthetic datasets</h1>
          <p>
            These are the training corpora the fine-tunes were built from. Each
            one blends ordinary instruction data with some share of
            constitution-grounded synthetic conversations — that share is what
            the experiments vary, so it leads every entry below. Records are read
            straight from Hugging Face a page at a time, so what you see is the
            published file, not a copy of it that can drift.{" "}
            <Link href="/glossary">Unfamiliar terms are in the glossary</Link>.
          </p>
        </div>
        <div className="heading-stat">
          <Database size={19} /><strong>{datasets.length}</strong><span>corpora</span>
        </div>
        <div className="heading-stat">
          <MessagesSquare size={19} />
          <strong>{records.toLocaleString()}</strong>
          <span>records, across the {counted.length} that publish a count</span>
        </div>
        <div className="heading-stat">
          <Layers size={19} />
          <strong>{withComposition}</strong>
          <span>publish their blend</span>
        </div>
      </header>

      {/* Folded into one line. As six full-width error-styled blocks these took
          600px above the fold and read as six things being broken, when what
          they record is that six repos publish adapters, code and logs rather
          than a conversation corpus. The detail is still one click away. */}
      {unresolved.length > 0 && (
        <details className="unresolved-note">
          <summary>
            <CloudOff size={14} />
            {unresolved.length} {unresolved.length === 1 ? "corpus" : "corpora"} publish no
            records to browse
          </summary>
          <ul>
            {unresolved.map((entry) => (
              <li key={entry.id}>
                <strong>{entry.title}</strong>
                {entry.hf?.message ? ` — ${entry.hf.message}` : ""}
                {entry.hf?.url ? (
                  <>
                    {" "}
                    <a href={entry.hf.url} target="_blank" rel="noreferrer">
                      open on Hugging Face
                    </a>
                  </>
                ) : null}
              </li>
            ))}
          </ul>
        </details>
      )}

      {datasets.length > 0 && <DatasetViewer datasets={viewerEntries} />}
    </main>
  );
}

