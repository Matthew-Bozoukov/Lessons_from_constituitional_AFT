"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Filter,
  LoaderCircle,
} from "lucide-react";
import { DatasetManifest, ResearchEntry, humanize } from "@/lib/content";
import { composition } from "@/lib/composition";
import { CompositionPanel, CorpusPicker } from "./CorpusPicker";
import { MockBadge, MockDataBanner } from "./MockDataBanner";
import { describeLoadError, loadChunk, loadJsonlWindow } from "@/lib/lazy";
import { NormalizedRecord, normalizeRecord } from "@/lib/records";
import { DialogueTranscript } from "./DialogueTranscript";

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

/**
 * Slug-shaped values are prettified; prose is left exactly as published.
 * `humanize` on a constitution trait turned "Preserve human oversight; avoid
 * unilateral, power-accruing action" into Title Case, which misquotes it.
 */
function label(value: string) {
  return /\s/.test(value) ? value : humanize(value);
}

/** What the grouping field actually is, so a trait is not called a "source". */
const GROUPING_NAMES: Record<string, [one: string, many: string]> = {
  source: ["source", "sources"],
  corpus: ["corpus", "corpora"],
  doc_type: ["document type", "document types"],
  trait_name: ["trait", "traits"],
  category: ["category", "categories"],
};

function formatBytes(bytes: number) {
  if (!bytes) return "unknown size";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

/** Reading position, for whichever paging mode the corpus uses. */
type Cursor = { chunks: number; offset: number; done: boolean };

const START: Cursor = { chunks: 0, offset: 0, done: false };

/**
 * Fetch one page and say where the next one starts.
 *
 * Deliberately outside the component and free of React state: both callers -
 * the first page on mount and the reader pressing Load more - need identical
 * paging, and a version that touched state would have to be a hook, which is
 * how the two paths drift apart.
 */
async function fetchPage(
  manifest: DatasetManifest,
  from: Cursor,
): Promise<{ items: unknown[]; cursor: Cursor }> {
  if (manifest.stream) {
    const { url, window } = manifest.stream;
    const page = await loadJsonlWindow(url, from.offset, window);
    return {
      items: page.records,
      cursor: {
        chunks: from.chunks + 1,
        offset: page.nextOffset,
        // A window that yields nothing and does not advance would otherwise
        // leave a Load-more button that can never finish.
        done: page.done || page.nextOffset <= from.offset,
      },
    };
  }
  const url = manifest.chunks[from.chunks];
  if (!url) return { items: [], cursor: { ...from, done: true } };
  return {
    items: await loadChunk<unknown>(url),
    cursor: {
      chunks: from.chunks + 1,
      offset: 0,
      done: from.chunks + 1 >= manifest.chunks.length,
    },
  };
}

export function DatasetViewer({ datasets }: { datasets: DatasetViewerEntry[] }) {
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasets[0]?.id || "");
  const selectedDataset =
    datasets.find((dataset) => dataset.id === selectedDatasetId) || datasets[0];
  const manifest = selectedDataset?.dataset as DatasetManifest | undefined;
  const [records, setRecords] = useState<NormalizedRecord[]>([]);
  const [selectedRecordId, setSelectedRecordId] = useState("");
  const [cursor, setCursor] = useState<Cursor>(START);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  // Filters are stored against the corpus they were chosen for, and read back
  // as "all" for any other. Each corpus has its own source vocabulary, so a
  // filter carried across a switch matches nothing: picking a second dataset
  // landed the reader on "23 read · 0 shown" and an empty list, with the cause
  // sitting in a dropdown they had set on a different corpus entirely.
  const [filters, setFilters] = useState({ key: "", category: "all", split: "all" });
  const filterKey = selectedDataset?.id || "";
  const active = filters.key === filterKey;
  const category = active ? filters.category : "all";
  const split = active ? filters.split : "all";
  const setCategory = (value: string) =>
    setFilters({ key: filterKey, category: value, split });
  const setSplit = (value: string) =>
    setFilters({ key: filterKey, category, split: value });

  const stream = manifest?.stream;
  const source =
    manifest?.source?.kind === "hf"
      ? `Hugging Face (${manifest.source.repo_id})`
      : "this site";

  // Records are never baked into the page: the first page arrives after mount,
  // and further pages only when the reader asks for one. Switching corpus
  // re-runs this, because `manifest` is a different object.
  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    async function loadFirstPage(current: DatasetManifest) {
      setLoading(true);
      setLoadError("");
      setRecords([]);
      setCursor(START);
      setSelectedRecordId("");
      try {
        const page = await fetchPage(current, START);
        if (cancelled) return;
        const next = page.items.map((raw, index) => normalizeRecord(raw, index));
        setRecords(next);
        setSelectedRecordId(next[0]?.id || "");
        setCursor(page.cursor);
      } catch (error) {
        if (!cancelled) setLoadError(describeLoadError(error, source));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadFirstPage(manifest);
    return () => {
      cancelled = true;
    };
  }, [manifest, source]);

  // A plain function, not a callback: it runs from a click, so it closes over
  // the cursor as it is now. Record numbering continues across pages and is
  // taken from the list being appended to - deriving it from a captured
  // `records` would renumber every page from a stale length, and two records
  // would answer to `#1`.
  async function loadMore() {
    if (!manifest || cursor.done || loading) return;
    setLoading(true);
    setLoadError("");
    try {
      const page = await fetchPage(manifest, cursor);
      setRecords((existing) => [
        ...existing,
        ...page.items.map((raw, index) => normalizeRecord(raw, existing.length + index)),
      ]);
      setCursor(page.cursor);
    } catch (error) {
      setLoadError(describeLoadError(error, source));
    } finally {
      setLoading(false);
    }
  }

  const categories = useMemo(() => {
    const declared = Object.keys(manifest?.stats.categories || {});
    const seen = new Set(records.map((record) => record.category));
    return Array.from(new Set([...declared, ...seen])).sort();
  }, [manifest?.stats.categories, records]);
  const splits = Object.keys(manifest?.stats.splits || {});
  // Named from the records themselves: a mixture groups by `source`, a synthdoc
  // corpus by the constitution trait its scenario came from. Calling both
  // "sources" would misdescribe half the collection.
  const groupingNames = GROUPING_NAMES[records[0]?.category_field || ""] || [
    "category",
    "categories",
  ];
  const grouping = groupingNames[categories.length === 1 ? 0 : 1];
  const made = useMemo(
    () => composition(manifest?.stats.categories, manifest?.stats.categories_source),
    [manifest?.stats.categories, manifest?.stats.categories_source],
  );

  const filteredRecords = useMemo(
    () =>
      records.filter(
        (record) =>
          (category === "all" || record.category === category) &&
          (split === "all" || record.split === split),
      ),
    [records, category, split],
  );

  const filtered = category !== "all" || split !== "all";
  const selectedRecord =
    filteredRecords.find((record) => record.id === selectedRecordId) || filteredRecords[0];
  const selectedIndex = filteredRecords.findIndex(
    (record) => record.id === selectedRecord?.id,
  );

  if (!selectedDataset || !manifest) {
    return <div className="empty-state">No indexed dialogue dataset found.</div>;
  }

  const totalBytes = stream?.total_bytes || 0;
  const readBytes = Math.min(cursor.offset, totalBytes || cursor.offset);
  // Only ever stated when a published statistic says so. A corpus with no
  // stats sidecar reports what has been read and nothing more.
  const knownCount = manifest.record_count || 0;
  const unloaded = knownCount ? Math.max(0, knownCount - records.length) : 0;
  const reasoningRecords = records.filter((record) => record.has_reasoning).length;
  const emptyThinkRecords = records.filter((record) => record.empty_think).length;

  return (
    <div className="dataset-workspace">
      {/* The picker replaces a 43-option <select>. A dropdown gives one line of
          text at a time, which is the wrong instrument for choosing between
          corpora that differ by a blend ratio: "Qwen3 6 27b synthdocv2 mixture
          15 85" and "...20 80" are adjacent, near-identical strings, and the
          number that distinguishes them is the whole point. Shown as a list,
          the ladder is visible. */}
      <div className="dataset-columns">
        <aside className="dataset-rail">
          <CorpusPicker
            corpora={datasets}
            selectedId={selectedDataset.id}
            onSelect={setSelectedDatasetId}
          />
        </aside>

        <div className="dataset-main">
          {/* Scoped to the corpus on screen, so a real corpus is never warned
              about and a fixture is never quietly shown as real. */}
          {selectedDataset.mock && <MockDataBanner scope="entry" />}
          <section className="dataset-description">
            <div>
              <span className="status status-draft">{humanize(selectedDataset.status)}</span>
              {/* Tied to the selected dataset, so switching between a fixture
                  and a real corpus moves the marker with it. */}
              {selectedDataset.mock && <MockBadge />}
              <h1>{selectedDataset.title}</h1>
              <p>{selectedDataset.summary}</p>
              <div className="dataset-facts">
                {knownCount ? (
                  <span><strong>{knownCount.toLocaleString()}</strong> records</span>
                ) : (
                  <span><strong>{records.length}</strong> records read so far</span>
                )}
                {manifest.stats.average_turns > 0 && (
                  <span><strong>{manifest.stats.average_turns}</strong> avg turns</span>
                )}
                {categories.length > 0 && (
                  <span><strong>{categories.length}</strong> {grouping}</span>
                )}
                {stream && <span><strong>{formatBytes(totalBytes)}</strong> on the Hub</span>}
                {stream && <code>{stream.path}</code>}
                <a href={manifest.source_file} download>
                  <Download size={14} /> Download JSONL
                </a>
              </div>
            </div>
            {made ? (
              <CompositionPanel made={made} />
            ) : (
              <p className="composition-note">
                This corpus publishes no statistics file, so its composition is not stated here.
                The records below are read straight from the JSONL either way.
              </p>
            )}
          </section>

      {/*
        There is no free-text search here, deliberately.

        Records are fetched a page at a time, so a search box can only ever
        search what has already been loaded. It would return "no results" for
        terms that occur hundreds of times in the corpus, which is worse than
        offering nothing: a silent false negative reads as an answer.

        The two dropdowns have the same scope limit but are honest about it:
        they are labelled as filtering loaded records, and the status line below
        states how much of the corpus has not been read.
      */}
      <section className="dataset-filters" aria-label="Dataset filters">
        <label>
          <Filter size={14} />
          <select
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            aria-label={`Filter loaded records by ${groupingNames[0]}`}
          >
            <option value="all">All {groupingNames[1]}</option>
            {categories.map((value) => (
              <option value={value} key={value}>{label(value)}</option>
            ))}
          </select>
        </label>
        {splits.length > 0 && (
          <label>
            <select
              value={split}
              onChange={(event) => setSplit(event.target.value)}
              aria-label="Filter loaded records by split"
            >
              <option value="all">All splits</option>
              {splits.map((value) => <option value={value} key={value}>{label(value)}</option>)}
            </select>
          </label>
        )}
        <span className="loaded-status">
          {loading && <LoaderCircle size={13} className="spin" />}
          {loadError ? (
            loadError
          ) : (
            <>
              {records.length} read · {filteredRecords.length} shown
              {/* Filters apply to what has been fetched, not to the corpus, so
                  a filter can read "0 shown" while the corpus holds hundreds of
                  matches. State what has not been searched rather than leaving
                  the reader to infer it - by record count where a published
                  statistic gives one, by bytes where none does. */}
              {filtered && !cursor.done && (
                <em>
                  {" "}
                  —{" "}
                  {unloaded
                    ? `${unloaded.toLocaleString()} records not yet read`
                    : `${formatBytes(Math.max(0, totalBytes - readBytes))} of the file not yet read`}
                </em>
              )}
            </>
          )}
        </span>
      </section>

      {/* The think block is not decoration: Qwen3.6 renders the tag on every
          assistant turn and leaves it EMPTY where no reasoning is supervised,
          so the two counts below are a real property of the training data. */}
      {(reasoningRecords > 0 || emptyThinkRecords > 0) && (
        <p className="dataset-think-note">
          <Brain size={13} /> Of the {records.length} records read,{" "}
          <strong>{reasoningRecords}</strong> carry a reasoning trace and{" "}
          <strong>{emptyThinkRecords}</strong> have an empty <code>&lt;think&gt;</code>{" "}
          block, where reasoning is rendered but not supervised.
        </p>
      )}

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
                  <span>
                    <code>{record.id}</code>
                    {record.has_reasoning && (
                      <small className="think-flag" title="Carries a reasoning trace">think</small>
                    )}
                  </span>
                  <strong>{label(record.category)}</strong>
                  <small>{record.messages.length} messages</small>
                </button>
              );
            })}
            {/* A filter that matches nothing in the fetched page is not the
                same as a corpus that contains nothing. Say which one it is. */}
            {filteredRecords.length === 0 && !loading && (
              <p className="record-empty">
                {cursor.done
                  ? "No record in this corpus matches the current filters."
                  : `No match in the ${records.length} records read so far — keep loading to search further into the file.`}
              </p>
            )}
            {!cursor.done && (
              <button
                className="load-more"
                type="button"
                onClick={loadMore}
                disabled={loading}
              >
                {loading ? "Loading…" : "Load more records"}
                {stream && totalBytes > 0 && (
                  <small>
                    {formatBytes(readBytes)} of {formatBytes(totalBytes)} read
                  </small>
                )}
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
            <DialogueTranscript messages={selectedRecord.messages} />
          ) : (
            <div className="empty-state">
              {loading ? "Reading the corpus from the Hub…" : "No records match the current filters."}
            </div>
          )}
        </section>

        <aside className="record-metadata">
          <div className="pane-heading">Record metadata</div>
          {selectedRecord && (
            <>
              <div className="record-identity">
                <span>Record</span><code>{selectedRecord.id}</code>
              </div>
              <dl>
                {Object.entries(selectedRecord.metadata).map(([key, value]) => (
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
      </div>
    </div>
  );
}
