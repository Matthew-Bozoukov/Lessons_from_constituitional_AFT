"use client";

// ABOUTME: The /datasets reading surface: one corpus at a time from a collapsible picker,
// ABOUTME: or two side by side when one corpus's prompts are a subset of the other's.

import { useEffect, useMemo, useState } from "react";
import {
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Filter,
  LoaderCircle,
  Search,
  TriangleAlert,
  X,
} from "lucide-react";
import { DatasetManifest, humanize } from "@/lib/content";
import { composition } from "@/lib/composition";
import { CompositionPanel, CorpusPicker } from "./CorpusPicker";
import { MockBadge, MockDataBanner } from "./MockDataBanner";
import { describeLoadError, loadJsonlAll, loadJsonlWindow } from "@/lib/lazy";
import { NormalizedRecord, normalizeRecord } from "@/lib/records";
import { Pairing, pairCorpora } from "@/lib/pairing";
import { DialogueTranscript } from "./DialogueTranscript";

/** One corpus in the picker: built live by TrainingDataExplorer from a Hub listing. */
export type DatasetViewerEntry = {
  id: string;
  title: string;
  summary: string;
  /** Rendered as the badge: the corpus kind (synth, mixture, ablation) or `smoke`. */
  status: string;
  tags: string[];
  /** Absent when the repo publishes nothing browsable; the explorer lists why. */
  dataset?: DatasetManifest;
  mock?: boolean;
};

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

/** Reading position in the byte-range stream. */
type Cursor = { offset: number; done: boolean };

const START: Cursor = { offset: 0, done: false };

/**
 * Above this combined size, pairing waits for a click.
 *
 * Single-corpus reading streams a window at a time; pairing cannot, because the subset
 * rule is a statement about the WHOLE of both files (see lib/pairing.ts) and a rule
 * checked over the first page would answer "not pairable" for corpora that pair
 * perfectly. So the reader is told the real cost before it is spent, in the same shape
 * the eval explorer uses for large rollout files.
 */
const PAIR_AUTO_BYTES = 8_000_000;

/**
 * Fetch one page and say where the next one starts.
 *
 * Deliberately outside the component and free of React state: both callers -
 * the first page on mount and the reader pressing Load more - need identical
 * paging, and a version that touched state would have to be a hook, which is
 * how the two paths drift apart. The file size rides back with every page:
 * a live-discovered corpus is not listed at build time, so the first
 * `content-range` is where the reader learns how much there is.
 */
async function fetchPage(
  manifest: DatasetManifest,
  from: Cursor,
): Promise<{ items: unknown[]; cursor: Cursor; totalBytes: number }> {
  const { url, window } = manifest.stream;
  const page = await loadJsonlWindow(url, from.offset, window);
  return {
    items: page.records,
    cursor: {
      offset: page.nextOffset,
      // A window that yields nothing and does not advance would otherwise
      // leave a Load-more button that can never finish.
      done: page.done || page.nextOffset <= from.offset,
    },
    totalBytes: page.totalBytes,
  };
}

/** The outcome of reading and aligning one pair of corpora. */
type PairState = { error?: string; result?: Pairing };

/** A line in the record list: one record, or one prompt as both corpora publish it. */
type ViewRow = { id: string; a?: NormalizedRecord; b?: NormalizedRecord };

export function DatasetViewer({ datasets }: { datasets: DatasetViewerEntry[] }) {
  const [selectedDatasetId, setSelectedDatasetId] = useState(datasets[0]?.id || "");
  const selectedDataset =
    datasets.find((dataset) => dataset.id === selectedDatasetId) || datasets[0];
  const manifest = selectedDataset?.dataset as DatasetManifest | undefined;
  const [records, setRecords] = useState<NormalizedRecord[]>([]);
  const [selectedRecordId, setSelectedRecordId] = useState("");
  const [cursor, setCursor] = useState<Cursor>(START);
  const [fileBytes, setFileBytes] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");

  // The corpus list is a drawer, not a fixed column. It opens on arrival, because a
  // reader who has not chosen anything yet needs to see what there is; picking a corpus
  // closes it and gives the width back to the records, which is what the page is for.
  const [pickerOpen, setPickerOpen] = useState(true);
  // Which side a click in the drawer assigns to while comparing.
  const [pickerTarget, setPickerTarget] = useState<"a" | "b">("a");
  // Two states, not one: the reader turns comparing ON before they have chosen what to
  // compare against, and a mode derived from "is a second corpus set" would switch itself
  // back off under them while the picker was still open.
  const [compareOn, setCompareOn] = useState(false);
  const [compareId, setCompareId] = useState("");
  // Keyed by the pair, so a result belongs to the two corpora it was computed from and
  // a switch back to an earlier pair costs nothing.
  const [pairs, setPairs] = useState<Record<string, PairState>>({});
  // Set to the pair the reader authorised a full read for (see PAIR_AUTO_BYTES).
  const [forcedPair, setForcedPair] = useState("");

  const compareDataset = compareOn && compareId
    ? datasets.find((dataset) => dataset.id === compareId)
    : undefined;
  const compareManifest = compareDataset?.dataset as DatasetManifest | undefined;
  const pairId = manifest && compareManifest ? `${selectedDataset?.id}|${compareId}` : "";
  // Derived, not stored: how big the pair is and whether the reader has agreed to read
  // it are facts about the current selection, and keeping them in state would mean an
  // effect writing state the render could have worked out for itself.
  const pairBytes = pairId
    ? (manifest?.stream?.total_bytes || 0) + (compareManifest?.stream?.total_bytes || 0)
    : 0;
  const needsConsent = pairBytes > PAIR_AUTO_BYTES && forcedPair !== pairId;
  const pair: PairState = (pairId && !needsConsent && pairs[pairId]) || {};
  const pairLoading = Boolean(pairId) && !needsConsent && !pairs[pairId];

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
      setFileBytes(0);
      setSelectedRecordId("");
      try {
        const page = await fetchPage(current, START);
        if (cancelled) return;
        const next = page.items.map((raw, index) => normalizeRecord(raw, index));
        setRecords(next);
        setSelectedRecordId(next[0]?.id || "");
        setCursor(page.cursor);
        setFileBytes(page.totalBytes);
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

  // -- pairing ---------------------------------------------------------------
  // Both files, whole, then the subset rule. Nothing here is derived from the streamed
  // page: see PAIR_AUTO_BYTES. A pair already in the table is not read again, which is
  // also what keeps this effect from looping on its own writes.
  useEffect(() => {
    if (!pairId || needsConsent || pairs[pairId] || !manifest || !compareManifest) return;
    let cancelled = false;
    Promise.all([
      loadJsonlAll<unknown>(manifest.stream.url),
      loadJsonlAll<unknown>(compareManifest.stream.url),
    ])
      .then(([rawA, rawB]) => {
        if (cancelled) return;
        const a = rawA.map((raw, index) => normalizeRecord(raw, index));
        const b = rawB.map((raw, index) => normalizeRecord(raw, index));
        setPairs((state) => ({ ...state, [pairId]: { result: pairCorpora(a, b) } }));
        setSelectedRecordId("");
      })
      .catch((error) => {
        if (cancelled) return;
        setPairs((state) => ({
          ...state,
          [pairId]: { error: describeLoadError(error, source) },
        }));
      });
    return () => {
      cancelled = true;
    };
  }, [pairId, needsConsent, pairs, manifest, compareManifest, source]);

  // Escape closes the drawer. Subscribing to the document is what an effect is for, and
  // a drawer that can only be closed by finding its button again is a drawer that gets
  // left open.
  useEffect(() => {
    if (!pickerOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPickerOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [pickerOpen]);

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
      if (page.totalBytes) setFileBytes(page.totalBytes);
    } catch (error) {
      setLoadError(describeLoadError(error, source));
    } finally {
      setLoading(false);
    }
  }

  function chooseCorpus(id: string) {
    if (pickerTarget === "b") {
      setCompareId(id === selectedDatasetId ? "" : id);
    } else {
      setSelectedDatasetId(id);
      if (id === compareId) setCompareId("");
    }
    setPickerOpen(false);
  }

  function toggleCompare(on: boolean) {
    setCompareOn(on);
    if (!on) {
      setCompareId("");
      setPickerTarget("a");
      return;
    }
    // Nothing is chosen for the reader: which corpus B should be is the whole question,
    // and a guessed second corpus would start a full two-file read on its own.
    setPickerTarget("b");
    setPickerOpen(true);
  }

  const comparing = compareOn;
  const paired = comparing && pair.result?.ok ? pair.result : undefined;

  // In a pair, the records on screen are the aligned ones, whole file already read; on
  // its own, the corpus is whatever has been streamed so far.
  const pairedARecords = useMemo(
    () => (paired ? paired.rows.flatMap((row) => (row.a ? [row.a] : [])) : []),
    [paired],
  );
  const readRecords = paired ? pairedARecords : records;

  const categories = useMemo(() => {
    const declared = Object.keys(manifest?.stats.categories || {});
    const seen = new Set(readRecords.map((record) => record.category));
    return Array.from(new Set([...declared, ...seen])).sort();
  }, [manifest?.stats.categories, readRecords]);
  const splits = Object.keys(manifest?.stats.splits || {});
  // Named from the records themselves: a mixture groups by `source`, a synthdoc
  // corpus by the constitution trait its scenario came from. Calling both
  // "sources" would misdescribe half the collection.
  const groupingNames = GROUPING_NAMES[readRecords[0]?.category_field || ""] || [
    "category",
    "categories",
  ];
  const grouping = groupingNames[categories.length === 1 ? 0 : 1];
  const made = useMemo(
    () => composition(manifest?.stats.categories, manifest?.stats.categories_source),
    [manifest?.stats.categories, manifest?.stats.categories_source],
  );

  const viewRows: ViewRow[] = useMemo(() => {
    const rows: ViewRow[] = paired
      ? paired.rows.map((row, index) => ({ id: `pair-${index}`, a: row.a, b: row.b }))
      : records.map((record) => ({ id: record.id, a: record }));
    return rows.filter((row) => {
      const record = row.a || row.b;
      if (!record) return false;
      return (
        (category === "all" || record.category === category) &&
        (split === "all" || record.split === split)
      );
    });
  }, [paired, records, category, split]);

  const filtered = category !== "all" || split !== "all";
  const selectedRow =
    viewRows.find((row) => row.id === selectedRecordId) || viewRows[0];
  const selectedIndex = viewRows.findIndex((row) => row.id === selectedRow?.id);
  const selectedRecord = selectedRow?.a || selectedRow?.b;

  if (!selectedDataset || !manifest) {
    return <div className="empty-state">No indexed dialogue dataset found.</div>;
  }

  // The listing states a size only when a tree call was needed; otherwise the
  // first byte-range response says how big the file is.
  const totalBytes = stream?.total_bytes || fileBytes;
  const readBytes = Math.min(cursor.offset, totalBytes || cursor.offset);
  // Only ever stated when a published statistic says so. A corpus with no
  // stats sidecar reports what has been read and nothing more.
  const knownCount = manifest.record_count || 0;
  const unloaded = knownCount ? Math.max(0, knownCount - records.length) : 0;
  const reasoningRecords = readRecords.filter((record) => record.has_reasoning).length;
  const emptyThinkRecords = readRecords.filter((record) => record.empty_think).length;

  return (
    <div className="dataset-workspace">
      {/* The picker replaces a 43-option <select>. A dropdown gives one line of
          text at a time, which is the wrong instrument for choosing between
          corpora that differ by a blend ratio: "Qwen3 6 27b synthdocv2 mixture
          15 85" and "...20 80" are adjacent, near-identical strings, and the
          number that distinguishes them is the whole point. Shown as a list,
          the ladder is visible — and once a corpus is chosen, folded away. */}
      <div className={pickerOpen ? "dataset-columns" : "dataset-columns is-collapsed"}>
        {pickerOpen && (
          <aside className="dataset-rail" id="dataset-rail">
            <CorpusPicker
              corpora={datasets}
              selectedId={selectedDataset.id}
              compareId={compareId}
              target={comparing ? pickerTarget : undefined}
              onTargetChange={setPickerTarget}
              onSelect={chooseCorpus}
              onClose={() => setPickerOpen(false)}
            />
          </aside>
        )}

        <div className="dataset-main">
          <div className="dataset-toolbar">
            <button
              type="button"
              className={pickerOpen ? "rail-toggle is-open" : "rail-toggle"}
              onClick={() => setPickerOpen((open) => !open)}
              aria-expanded={pickerOpen}
              aria-controls="dataset-rail"
            >
              {pickerOpen ? <X size={14} /> : <Search size={14} />}
              {pickerOpen ? "Hide corpus list" : "Find a corpus"}
            </button>
            <label className="compare-toggle">
              <input
                type="checkbox"
                checked={comparing}
                onChange={(event) => toggleCompare(event.target.checked)}
              />
              Compare two corpora
            </label>
            {comparing && (
              <div className="run-legend">
                <span className="legend-chip">
                  <i className="swatch swatch-a" />A — {selectedDataset.title}
                </span>
                <span className="legend-chip">
                  <i className="swatch swatch-b" />
                  {compareDataset ? `B — ${compareDataset.title}` : "B — not chosen yet"}
                  <button
                    type="button"
                    className="chip-clear"
                    aria-label="Stop comparing"
                    onClick={() => toggleCompare(false)}
                  >
                    <X size={12} />
                  </button>
                </span>
              </div>
            )}
          </div>

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

          {/* Everything the pair can be: waiting on a size the reader has not
              agreed to spend, failing to load, refused by the subset rule, or
              lined up. Each says which, in the numbers it was decided on. */}
          {comparing && needsConsent && (
            <div className="pairing-note">
              <p>
                Lining these up reads the two files in full — {formatBytes(pairBytes)}{" "}
                together. Only whole files can answer whether one corpus&apos;s prompts are a
                subset of the other&apos;s.
              </p>
              <button
                type="button"
                className="load-rollouts"
                onClick={() => setForcedPair(pairId)}
              >
                Read {formatBytes(pairBytes)} and pair
              </button>
            </div>
          )}
          {comparing && pairLoading && (
            <div className="pairing-note">
              <LoaderCircle size={13} className="spin" /> Reading both corpora in full…
            </div>
          )}
          {comparing && !compareDataset && (
            <div className="pairing-note">
              <p>Pick a second corpus in the list to read it beside this one.</p>
            </div>
          )}
          {comparing && pair.error && (
            <div className="pairing-error" role="alert">
              <TriangleAlert size={15} />
              <div>{pair.error}</div>
            </div>
          )}
          {comparing && pair.result && !pair.result.ok && (
            <div className="pairing-error" role="alert">
              <TriangleAlert size={15} />
              <div>
                <strong>These two corpora cannot be shown side by side.</strong>
                <p>{pair.result.reason}</p>
                <p className="pairing-counts">
                  <span><strong>{pair.result.shared.toLocaleString()}</strong> prompts in both</span>
                  <span><strong>{pair.result.onlyA.toLocaleString()}</strong> only in A</span>
                  <span><strong>{pair.result.onlyB.toLocaleString()}</strong> only in B</span>
                </p>
              </div>
            </div>
          )}
          {paired && (
            <p className="pairing-ok">
              Aligned on <strong>{paired.rows.length.toLocaleString()}</strong> shared prompts
              {paired.direction === "equal"
                ? " — both corpora publish exactly the same prompt set."
                : paired.direction === "a-in-b"
                  ? " — every prompt in A also appears in B."
                  : " — every prompt in B also appears in A."}
              {paired.hidden > 0 && (
                <em>
                  {" "}
                  {paired.hidden.toLocaleString()} records of the larger corpus have no
                  counterpart and are not shown.
                </em>
              )}
            </p>
          )}

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
          ) : paired ? (
            // A pair has both files whole, so there is nothing unread to warn about.
            <>{paired.rows.length.toLocaleString()} paired · {viewRows.length} shown</>
          ) : (
            <>
              {records.length} read · {viewRows.length} shown
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
          <Brain size={13} /> Of the {readRecords.length} records read{paired ? " in A" : ""},{" "}
          <strong>{reasoningRecords}</strong> carry a reasoning trace and{" "}
          <strong>{emptyThinkRecords}</strong> have an empty <code>&lt;think&gt;</code>{" "}
          block, where reasoning is rendered but not supervised.
        </p>
      )}

      <div className={paired ? "dataset-browser is-compare" : "dataset-browser"}>
        <aside className="record-index">
          <div className="pane-heading"><Database size={14} /> Records</div>
          <div className="record-list">
            {viewRows.map((row, index) => {
              const record = row.a || row.b;
              if (!record) return null;
              const isActive = row.id === selectedRow?.id;
              return (
                <button
                  className={isActive ? "record-button is-active" : "record-button"}
                  type="button"
                  onClick={() => setSelectedRecordId(row.id)}
                  key={row.id}
                >
                  <span>
                    <code>{paired ? `#${index + 1}` : record.id}</code>
                    {record.has_reasoning && (
                      <small className="think-flag" title="Carries a reasoning trace">think</small>
                    )}
                    {paired && !(row.a && row.b) && (
                      <small className="think-flag" title="Present in only one corpus">
                        {row.a ? "A only" : "B only"}
                      </small>
                    )}
                  </span>
                  <strong>{label(record.category)}</strong>
                  <small>{record.messages.length} messages</small>
                </button>
              );
            })}
            {/* A filter that matches nothing in the fetched page is not the
                same as a corpus that contains nothing. Say which one it is. */}
            {viewRows.length === 0 && !loading && (
              <p className="record-empty">
                {cursor.done || paired
                  ? "No record in this corpus matches the current filters."
                  : `No match in the ${records.length} records read so far — keep loading to search further into the file.`}
              </p>
            )}
            {!cursor.done && !paired && (
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
            <span>{paired ? "Same prompt, both corpora" : "Conversation preview"}</span>
            <div className="record-nav">
              <button
                type="button"
                aria-label="Previous record"
                disabled={selectedIndex <= 0}
                onClick={() => setSelectedRecordId(viewRows[selectedIndex - 1]?.id)}
              ><ChevronLeft size={15} /></button>
              <code>{selectedIndex + 1} / {viewRows.length}</code>
              <button
                type="button"
                aria-label="Next record"
                disabled={selectedIndex < 0 || selectedIndex >= viewRows.length - 1}
                onClick={() => setSelectedRecordId(viewRows[selectedIndex + 1]?.id)}
              ><ChevronRight size={15} /></button>
            </div>
          </div>
          {selectedRow && paired ? (
            // Two panes, one prompt: the same shape the eval explorer uses for A/B
            // rollouts, so the two comparisons read the same way.
            <div className="transcript-panes is-compare">
              {(["a", "b"] as const).map((side) => {
                const record = selectedRow[side];
                const corpus = side === "a" ? selectedDataset : compareDataset;
                return (
                  <div className="transcript-pane" key={side}>
                    <div className={`pane-title pane-${side}`}>
                      <i className={`swatch swatch-${side}`} />
                      {side.toUpperCase()} — {corpus?.title}
                    </div>
                    {record ? (
                      <DialogueTranscript messages={record.messages} compact />
                    ) : (
                      <div className="pane-missing">
                        This prompt is not in this corpus.
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : selectedRecord ? (
            <DialogueTranscript messages={selectedRecord.messages} />
          ) : (
            <div className="empty-state">
              {loading ? "Reading the corpus from the Hub…" : "No records match the current filters."}
            </div>
          )}
        </section>

        <aside className="record-metadata">
          <div className="pane-heading">Record metadata</div>
          {selectedRow && (
            <>
              {(paired ? (["a", "b"] as const) : (["a"] as const)).map((side) => {
                const record = selectedRow[side];
                if (!record) return null;
                return (
                  <div className="record-meta-block" key={side}>
                    <div className="record-identity">
                      <span>{paired ? `Record ${side.toUpperCase()}` : "Record"}</span>
                      <code>{record.id}</code>
                    </div>
                    <dl>
                      {Object.entries(record.metadata).map(([key, value]) => (
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
                  </div>
                );
              })}
            </>
          )}
        </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
