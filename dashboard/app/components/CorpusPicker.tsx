"use client";

import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { Composition, composition, formatShare } from "@/lib/composition";
import type { DatasetManifest } from "@/lib/content";
import { corpusMatches } from "@/lib/trainingData";
import { MockBadge } from "./MockDataBanner";

export type PickerCorpus = {
  id: string;
  title: string;
  summary?: string;
  status?: string;
  tags?: string[];
  dataset?: DatasetManifest;
  mock?: boolean;
};

/** Everything the search box can match: repo id, title, summary, kind, tags, sources, rows file. */
function searchFields(corpus: PickerCorpus): Array<string | undefined> {
  return [
    corpus.id,
    corpus.title,
    corpus.summary,
    corpus.status,
    ...(corpus.tags || []),
    ...Object.keys(corpus.dataset?.stats.categories || {}),
    corpus.dataset?.source_file.split("/").pop(),
  ];
}

function formatBytes(bytes: number) {
  if (!bytes) return "";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

/**
 * The share bar: constitution-grounded data against everything else.
 *
 * Two categories, not ten. A ten-source mixture has ten `by_source` entries, but
 * colouring all ten would mean generating hues past the point where they can be
 * told apart - and it would bury the one contrast the experiment is about. The
 * full ten are in the table underneath, with exact counts, so nothing is hidden;
 * this is the headline. The 2px gap is the secondary encoding, so the split
 * survives being read without colour.
 */
function ShareBar({ share, thin = false }: { share: number; thin?: boolean }) {
  const percent = Math.max(0, Math.min(1, share)) * 100;
  return (
    <div className={thin ? "share-bar thin" : "share-bar"} aria-hidden="true">
      <span className="share-constitution" style={{ width: `${percent}%` }} />
      <span className="share-general" style={{ width: `${100 - percent}%` }} />
    </div>
  );
}

type Group = { key: string; heading: string; blurb: string; items: PickerCorpus[] };

/**
 * Grouped by what the published statistics say, never by what a slug looks
 * like. A corpus lands in "controls" because its measured constitution share is
 * zero, not because "tulu" appears in its name.
 */
function groupCorpora(corpora: PickerCorpus[], compositions: Map<string, Composition | null>) {
  const groups: Group[] = [
    {
      key: "mixtures",
      heading: "Constitution mixtures",
      blurb: "Instruction data blended with constitution-grounded synthetic data.",
      items: [],
    },
    {
      key: "controls",
      heading: "No constitution data",
      blurb: "Published composition contains none of the constitution sources.",
      items: [],
    },
    {
      // Not "generated corpora": this group is defined by a missing statistics
      // file, not by what the data is. It holds raw synthdoc exports AND
      // mixtures that were published without a `mixture_stats.json` - calling
      // all of them generated corpora would assert something about the second
      // kind that is plainly false.
      key: "unpublished",
      heading: "Composition not published",
      blurb: "No statistics file on the Hub, so the blend cannot be stated. Records read as usual.",
      items: [],
    },
  ];
  for (const corpus of corpora) {
    const share = compositions.get(corpus.id)?.constitutionShare ?? null;
    const key = share === null ? "unpublished" : share > 0 ? "mixtures" : "controls";
    groups.find((group) => group.key === key)?.items.push(corpus);
  }
  for (const group of groups) {
    // Biggest intervention first: the sweep is the story, and 20/80 above 10/90
    // reads as the ladder it is.
    group.items.sort((a, b) => {
      const sa = compositions.get(a.id)?.constitutionShare ?? -1;
      const sb = compositions.get(b.id)?.constitutionShare ?? -1;
      return sb - sa || a.title.localeCompare(b.title);
    });
  }
  return groups.filter((group) => group.items.length);
}

export function CorpusPicker({
  corpora,
  selectedId,
  onSelect,
  compareId = "",
  target,
  onTargetChange,
  onClose,
}: {
  corpora: PickerCorpus[];
  selectedId: string;
  onSelect: (id: string) => void;
  /** The second corpus of a side-by-side pair, marked B in the list. */
  compareId?: string;
  /** Which side the next click fills. Absent when nothing is being compared. */
  target?: "a" | "b";
  onTargetChange?: (target: "a" | "b") => void;
  onClose?: () => void;
}) {
  const [query, setQuery] = useState("");
  const comparing = Boolean(target && onTargetChange);

  const compositions = useMemo(() => {
    const map = new Map<string, Composition | null>();
    for (const corpus of corpora) {
      map.set(
        corpus.id,
        composition(corpus.dataset?.stats.categories, corpus.dataset?.stats.categories_source),
      );
    }
    return map;
  }, [corpora]);

  // Order-free, separator-agnostic term match over every field a reader might
  // remember a corpus by (see corpusMatches). Still not a record search.
  const matches = useMemo(
    () => corpora.filter((corpus) => corpusMatches(searchFields(corpus), query)),
    [corpora, query],
  );

  const groups = useMemo(() => groupCorpora(matches, compositions), [matches, compositions]);

  return (
    <div className="corpus-picker">
      {/* Which corpus a click lands on has to be visible BEFORE the click: with two
          slots and one list, an unlabelled picker would silently replace whichever
          corpus the reader was already reading. */}
      {(comparing || onClose) && (
        <div className="picker-head">
          {comparing && (
            <div className="picker-target" role="group" aria-label="Assign the next choice to">
              {(["a", "b"] as const).map((side) => (
                <button
                  type="button"
                  key={side}
                  className={target === side ? "is-active" : ""}
                  aria-pressed={target === side}
                  onClick={() => onTargetChange?.(side)}
                >
                  <i className={`swatch swatch-${side}`} />
                  Choose {side.toUpperCase()}
                </button>
              ))}
            </div>
          )}
          {onClose && (
            <button type="button" className="picker-close" onClick={onClose} aria-label="Hide the corpus list">
              <X size={14} />
            </button>
          )}
        </div>
      )}
      <label className="corpus-search">
        <Search size={14} />
        <input
          type="search"
          value={query}
          placeholder="Filter by name, date, tag, source or file"
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Filter corpora by name, date, tag, source or file"
        />
      </label>

      {matches.length === 0 && (
        <p className="record-empty">
          No corpus matches &ldquo;{query}&rdquo;. Every word must appear somewhere in a
          corpus&apos;s repo name, date, tags, sources or rows file (in any order, any
          separator) — this does not search the records themselves.
        </p>
      )}

      <div className="corpus-groups">
        {groups.map((group) => (
          <section key={group.key}>
            <header>
              <span className="eyebrow">{group.heading}</span>
              <small>{group.blurb}</small>
            </header>
            {group.items.map((corpus) => {
              const made = compositions.get(corpus.id);
              const share = made?.constitutionShare ?? null;
              const count = corpus.dataset?.record_count || 0;
              const bytes = corpus.dataset?.stream?.total_bytes || 0;
              const side = corpus.id === selectedId ? "a" : corpus.id === compareId ? "b" : "";
              return (
                <button
                  type="button"
                  key={corpus.id}
                  className={side ? `corpus-row is-active is-${side}` : "corpus-row"}
                  onClick={() => onSelect(corpus.id)}
                  aria-current={Boolean(side)}
                >
                  <strong>
                    {/* The letter, not colour alone: which corpus is A and which is B is
                        the axis of the whole comparison. */}
                    {comparing && side && (
                      <i className={`side-marker swatch swatch-${side}`} aria-hidden="true" />
                    )}
                    {corpus.title}
                    {corpus.mock && <MockBadge />}
                  </strong>
                  {share !== null && (
                    <span className="corpus-share">
                      <ShareBar share={share} thin />
                      <em>{formatShare(share)} constitution</em>
                    </span>
                  )}
                  <small>
                    {count ? `${count.toLocaleString()} records` : "record count unpublished"}
                    {bytes ? ` · ${formatBytes(bytes)}` : ""}
                  </small>
                </button>
              );
            })}
          </section>
        ))}
      </div>
    </div>
  );
}

/** The selected corpus's full breakdown: bar, legend, and every source. */
export function CompositionPanel({ made }: { made: Composition }) {
  const share = made.constitutionShare;
  return (
    <div className="composition-panel">
      {share !== null ? (
        <>
          <div className="composition-headline">
            <strong>{formatShare(share)}</strong>
            <span>
              constitution-grounded — {made.constitutionCount.toLocaleString()} of{" "}
              {made.total.toLocaleString()} records
            </span>
          </div>
          <ShareBar share={share} />
          {/* Identity is never colour alone: the legend names both halves. */}
          <div className="composition-legend">
            <span><i className="swatch constitution" /> Constitution-grounded</span>
            <span><i className="swatch general" /> General instruction data</span>
          </div>
        </>
      ) : (
        <p className="composition-note">
          Grouped by {made.rows.length} categories rather than by mixture source, so there is no
          constitution share to state for it. The categories below are as published.
        </p>
      )}

      {/* The table is not a fallback for the bar, it is the record. Two colours
          summarise; these are the numbers, all of them, as published. */}
      <table className="composition-table">
        <thead>
          <tr><th scope="col">Source</th><th scope="col">Records</th><th scope="col">Share</th></tr>
        </thead>
        <tbody>
          {made.rows.map((row) => (
            <tr key={row.name} className={row.constitution ? "is-constitution" : undefined}>
              <th scope="row">{row.name}</th>
              <td>{row.count.toLocaleString()}</td>
              <td>{formatShare(row.share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
