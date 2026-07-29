"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Filter,
  LoaderCircle,
  Search,
} from "lucide-react";
import {
  DatasetManifest,
  DialogueMessage,
  DialogueRecord,
  ResearchEntry,
  humanize,
} from "@/lib/content";
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
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [split, setSplit] = useState("all");

  useEffect(() => {
    let cancelled = false;
    async function loadInitialChunk() {
      if (!manifest?.chunks[0]) return;
      setLoading(true);
      setRecords([]);
      setLoadedChunks(0);
      const response = await fetch(manifest.chunks[0]);
      const nextRecords = (await response.json()) as DialogueRecord[];
      if (!cancelled) {
        setRecords(nextRecords);
        setSelectedRecordId(nextRecords[0]?.id || "");
        setLoadedChunks(1);
        setLoading(false);
      }
    }
    loadInitialChunk();
    return () => {
      cancelled = true;
    };
  }, [manifest?.chunks]);

  const categories = Object.keys(manifest?.stats.categories || {});
  const splits = Object.keys(manifest?.stats.splits || {});
  const filteredRecords = useMemo(
    () =>
      records.filter((record) => {
        const metadata = metadataFor(record);
        const recordCategory = String(metadata.category || record.category || "uncategorized");
        const recordSplit = String(metadata.split || record.split || "unspecified");
        const searchable = `${record.id} ${recordCategory} ${JSON.stringify(messagesFor(record))}`.toLowerCase();
        return (
          (category === "all" || recordCategory === category) &&
          (split === "all" || recordSplit === split) &&
          (!query || searchable.includes(query.toLowerCase()))
        );
      }),
    [records, category, split, query],
  );

  const selectedRecord =
    filteredRecords.find((record) => record.id === selectedRecordId) ||
    filteredRecords[0];
  const selectedIndex = filteredRecords.findIndex(
    (record) => record.id === selectedRecord?.id,
  );

  async function loadMore() {
    if (!manifest || loadedChunks >= manifest.chunks.length) return;
    setLoading(true);
    const response = await fetch(manifest.chunks[loadedChunks]);
    const nextRecords = (await response.json()) as DialogueRecord[];
    setRecords((existing) => [...existing, ...nextRecords]);
    setLoadedChunks((value) => value + 1);
    setLoading(false);
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
          <h1>{selectedDataset.title}</h1>
          <p>{selectedDataset.summary}</p>
        </div>
        <dl>
          <div><dt>Objective</dt><dd>{String(selectedDataset.training_objective || "unspecified").toUpperCase()}</dd></div>
          <div><dt>Version</dt><dd><code>{selectedDataset.dataset_version || "unknown"}</code></dd></div>
          <div><dt>Generator</dt><dd><code>{selectedDataset.generator_model || "unknown"}</code></dd></div>
        </dl>
      </section>

      <section className="dataset-filters" aria-label="Dataset filters">
        <label className="search-control">
          <Search size={15} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search loaded conversations"
            aria-label="Search conversations"
          />
        </label>
        <label>
          <Filter size={14} />
          <select value={category} onChange={(event) => setCategory(event.target.value)} aria-label="Filter by category">
            <option value="all">All categories</option>
            {categories.map((value) => <option value={value} key={value}>{humanize(value)}</option>)}
          </select>
        </label>
        <label>
          <select value={split} onChange={(event) => setSplit(event.target.value)} aria-label="Filter by split">
            <option value="all">All splits</option>
            {splits.map((value) => <option value={value} key={value}>{humanize(value)}</option>)}
          </select>
        </label>
        <span className="loaded-status">
          {loading && <LoaderCircle size={13} className="spin" />}
          {records.length} loaded · {filteredRecords.length} shown
        </span>
      </section>

      <div className="dataset-browser">
        <aside className="record-index">
          <div className="pane-heading"><Database size={14} /> Records</div>
          <div className="record-list">
            {filteredRecords.map((record) => {
              const recordMetadata = metadataFor(record);
              const active = record.id === selectedRecord?.id;
              return (
                <button
                  className={active ? "record-button is-active" : "record-button"}
                  type="button"
                  onClick={() => setSelectedRecordId(record.id)}
                  key={record.id}
                >
                  <span><code>{record.id}</code><small>{String(recordMetadata.split || "unspecified")}</small></span>
                  <strong>{humanize(String(recordMetadata.category || "uncategorized"))}</strong>
                  <small>{messagesFor(record).length} messages</small>
                </button>
              );
            })}
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

