"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Filter,
  LoaderCircle,
} from "lucide-react";
import {
  DatasetManifest,
  DialogueMessage,
  DialogueRecord,
  ResearchEntry,
  humanize,
} from "@/lib/content";
import { MockBadge } from "./MockDataBanner";
import { describeLoadError, loadChunk } from "@/lib/lazy";
import { DialogueTranscript } from "./DialogueTranscript";

function messagesFor(record: DialogueRecord): DialogueMessage[] {
  const candidate =
    record.messages || record.conversation || record.turns || record.dialogue;
  if (candidate) return candidate;
  if (record.prompt !== undefined || record.response !== undefined) {
    return [
      { role: "user", content: String(record.prompt || "") },
      { role: "assistant", content: String(record.response || "") },
    ];
  }
  return [];
}

function metadataFor(record: DialogueRecord) {
  return (record.metadata || {}) as Record<string, unknown>;
}

/**
 * A corpus labels its records either under `metadata` or at the top level, so
 * both are checked, nested first.
 *
 * These exist because the filter logic and the record list used to read
 * different paths - the filter fell back to the top level, the list did not.
 * That was invisible against a fixture carrying `metadata.category`, and showed
 * up the moment a real corpus put its labels at the top level: the filters
 * worked while every row read "Uncategorized". One source of truth now.
 */
function categoryOf(record: DialogueRecord) {
  const metadata = metadataFor(record);
  return String(metadata.category || record.category || "uncategorized");
}

function splitOf(record: DialogueRecord) {
  const metadata = metadataFor(record);
  return String(metadata.split || record.split || "unspecified");
}

export type DatasetViewerEntry = Pick<
  ResearchEntry,
  | "id"
  | "title"
  | "summary"
  | "status"
  | "tags"
  | "dataset"
  | "dataset_id"
  | "dataset_version"
  | "training_objective"
  | "generator_model"
  | "mock"
>;

export function DatasetViewer({ datasets }: { datasets: DatasetViewerEntry[] }) {
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasets[0]?.id || "");
  const selectedDataset =
    datasets.find((dataset) => dataset.id === selectedDatasetId) || datasets[0];
  const manifest = selectedDataset?.dataset as DatasetManifest | undefined;
  const [records, setRecords] = useState<DialogueRecord[]>([]);
  const [selectedRecordId, setSelectedRecordId] = useState("");
  const [loadedChunks, setLoadedChunks] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [category, setCategory] = useState("all");
  const [split, setSplit] = useState("all");

  const chunkSource =
    manifest?.source?.kind === "hf"
      ? `Hugging Face (${manifest.source.repo_id})`
      : "this site";

  // Records are never baked into the page: the first chunk arrives after mount
  // and further chunks only when the reader asks for them.
  useEffect(() => {
    let cancelled = false;
    async function loadInitialChunk() {
      if (!manifest?.chunks[0]) return;
      setLoading(true);
      setLoadError("");
      setRecords([]);
      setLoadedChunks(0);
      try {
        const nextRecords = await loadChunk<DialogueRecord>(manifest.chunks[0]);
        if (!cancelled) {
          setRecords(nextRecords);
          setSelectedRecordId(nextRecords[0]?.id || "");
          setLoadedChunks(1);
        }
      } catch (error) {
        if (!cancelled) setLoadError(describeLoadError(error, chunkSource));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadInitialChunk();
    return () => {
      cancelled = true;
    };
  }, [manifest?.chunks, chunkSource]);

  const categories = Object.keys(manifest?.stats.categories || {});
  const splits = Object.keys(manifest?.stats.splits || {});
  const filteredRecords = useMemo(
    () =>
      records.filter(
        (record) =>
          (category === "all" || categoryOf(record) === category) &&
          (split === "all" || splitOf(record) === split),
      ),
    [records, category, split],
  );

  const filtered = category !== "all" || split !== "all";
  const unloadedRecords = Math.max(0, (manifest?.record_count || 0) - records.length);

  const selectedRecord =
    filteredRecords.find((record) => record.id === selectedRecordId) ||
    filteredRecords[0];
  const selectedIndex = filteredRecords.findIndex(
    (record) => record.id === selectedRecord?.id,
  );

  async function loadMore() {
    if (!manifest || loadedChunks >= manifest.chunks.length) return;
    setLoading(true);
    setLoadError("");
    try {
      const nextRecords = await loadChunk<DialogueRecord>(manifest.chunks[loadedChunks]);
      setRecords((existing) => [...existing, ...nextRecords]);
      setLoadedChunks((value) => value + 1);
    } catch (error) {
      setLoadError(describeLoadError(error, chunkSource));
    } finally {
      setLoading(false);
    }
  }

  if (!selectedDataset || !manifest) {
    return <div className="empty-state">No indexed dialogue dataset found.</div>;
  }

  const metadata = selectedRecord ? metadataFor(selectedRecord) : {};

  return (
    <div className="dataset-workspace">
      <section className="dataset-toolbar">
        <div className="dataset-selector">
          <span className="eyebrow">Dataset</span>
          <select
            value={selectedDataset.id}
            onChange={(event) => setSelectedDatasetId(event.target.value)}
            aria-label="Select dataset"
          >
            {datasets.map((dataset) => (
              <option value={dataset.id} key={dataset.id}>{dataset.title}</option>
            ))}
          </select>
        </div>
        <div className="dataset-summary">
          <span><strong>{manifest.record_count}</strong> records</span>
          <span><strong>{manifest.stats.average_turns}</strong> avg turns</span>
          <span><strong>{Object.keys(manifest.stats.categories).length}</strong> categories</span>
          <a href={manifest.source_file} download><Download size={14} /> JSONL</a>
        </div>
      </section>

      <section className="dataset-description">
        <div>
          <span className="status status-draft">{humanize(selectedDataset.status)}</span>
          {/* Tied to the selected dataset, so switching between a fixture and a
              real corpus moves the marker with it. */}
          {selectedDataset.mock && <MockBadge />}
          <h1>{selectedDataset.title}</h1>
          <p>{selectedDataset.summary}</p>
        </div>
        <dl>
          <div><dt>Objective</dt><dd>{String(selectedDataset.training_objective || "unspecified").toUpperCase()}</dd></div>
          <div><dt>Version</dt><dd><code>{selectedDataset.dataset_version || "unknown"}</code></dd></div>
          <div><dt>Generator</dt><dd><code>{selectedDataset.generator_model || "unknown"}</code></dd></div>
        </dl>
      </section>

      {/*
        There is no free-text search here, deliberately.

        Records are fetched a chunk at a time, so a search box can only ever
        search what has already been loaded - 25 of 1,443 on arrival. It would
        return "no results" for terms that occur hundreds of times in the
        corpus, which is worse than offering nothing: a silent false negative
        reads as an answer. Restoring it needs a real index (per-chunk term or
        label indexes in the manifest, so the viewer can fetch only the chunks
        that can match), not a bigger first page.

        The two dropdowns have the same scope limit but are honest about it:
        they are labelled as filtering loaded records, and the status line and
        empty state below both state how many records have not been fetched.
      */}
      <section className="dataset-filters" aria-label="Dataset filters">
        <label>
          <Filter size={14} />
          <select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="Filter loaded records by category">
            <option value="all">All categories</option>
            {categories.map((value) => <option value={value} key={value}>{humanize(value)}</option>)}
          </select>
        </label>
        <label>
          <select value={split} onChange={(event) => setSplit(event.target.value)} aria-label="Filter loaded records by split">
            <option value="all">All splits</option>
            {splits.map((value) => <option value={value} key={value}>{humanize(value)}</option>)}
          </select>
        </label>
        <span className="loaded-status">
          {loading && <LoaderCircle size={13} className="spin" />}
          {loadError ? (
            loadError
          ) : (
            <>
              {records.length} loaded · {filteredRecords.length} shown
              {/* Filters apply to what has been fetched, not to the corpus.
                  With one chunk of 25 out of 1,443 records, a filter can read
                  "0 shown" while the corpus holds hundreds of matches - so the
                  count of records not yet searched is stated rather than left
                  for the reader to infer. */}
              {filtered && unloadedRecords > 0 && (
                <em>
                  {" "}
                  — filtering {records.length} fetched of {manifest.record_count};{" "}
                  {unloadedRecords.toLocaleString()} not yet searched
                </em>
              )}
            </>
          )}
        </span>
      </section>

      <div className="dataset-browser">
        <aside className="record-index">
          <div className="pane-heading"><Database size={14} /> Records</div>
          <div className="record-list">
            {filteredRecords.map((record) => {
              const active = record.id === selectedRecord?.id;
              return (
                <button
                  className={active ? "record-button is-active" : "record-button"}
                  type="button"
                  onClick={() => setSelectedRecordId(record.id)}
                  key={record.id}
                >
                  <span><code>{record.id}</code><small>{splitOf(record)}</small></span>
                  <strong>{humanize(categoryOf(record))}</strong>
                  <small>{messagesFor(record).length} messages</small>
                </button>
              );
            })}
            {/* A filter that matches nothing in the fetched page is not the
                same as a corpus that contains nothing. Say which one it is. */}
            {filteredRecords.length === 0 && (
              <p className="record-empty">
                {unloadedRecords > 0
                  ? `No match in the ${records.length} records fetched so far. ${unloadedRecords.toLocaleString()} remain unfetched — keep loading to search them.`
                  : "No record in this corpus matches the current filters."}
              </p>
            )}
            {loadedChunks < manifest.chunks.length && (
              <button className="load-more" type="button" onClick={loadMore} disabled={loading}>
                Load next {manifest.chunk_size}
              </button>
            )}
          </div>
        </aside>

        <section className="conversation-pane">
          <div className="pane-heading">
            <span>Conversation preview</span>
            <div className="record-nav">
              <button
                type="button"
                aria-label="Previous record"
                disabled={selectedIndex <= 0}
                onClick={() => setSelectedRecordId(filteredRecords[selectedIndex - 1]?.id)}
              ><ChevronLeft size={15} /></button>
              <code>{selectedIndex + 1} / {filteredRecords.length}</code>
              <button
                type="button"
                aria-label="Next record"
                disabled={selectedIndex < 0 || selectedIndex >= filteredRecords.length - 1}
                onClick={() => setSelectedRecordId(filteredRecords[selectedIndex + 1]?.id)}
              ><ChevronRight size={15} /></button>
            </div>
          </div>
          {selectedRecord ? (
            <DialogueTranscript messages={messagesFor(selectedRecord)} />
          ) : (
            <div className="empty-state">No records match the current filters.</div>
          )}
        </section>

        <aside className="record-metadata">
          <div className="pane-heading">Record metadata</div>
          {selectedRecord && (
            <>
              <div className="record-identity">
                <span>Record ID</span><code>{selectedRecord.id}</code>
              </div>
              <dl>
                {Object.entries(metadata).map(([key, value]) => (
                  <div key={key}>
                    <dt>{humanize(key)}</dt>
                    <dd>
                      {Array.isArray(value) ? (
                        <span className="meta-tags">
                          {value.map((item) => <code key={String(item)}>{String(item)}</code>)}
                        </span>
                      ) : typeof value === "object" ? (
                        <code>{JSON.stringify(value)}</code>
                      ) : (
                        <code>{String(value)}</code>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

