"use client";

// Reader for a LESS-style targeted data selection: every row of a scored pool,
// ordered by influence on a set of target behaviours.
//
// Three payloads, none of them baked into the build:
//   1. scores.jsonl        (~2 MB)  the ranking itself, fetched whole on mount
//   2. pool_offsets.json   (~75 KB) id -> [byte offset, length] into the pool
//   3. one conversation    (~8 KB)  range-fetched from the pool only when selected
//
// The third is the point of the offset sidecar: the scored pool is a 24 MB JSONL
// and pulling it to show one conversation would be absurd, but the Hub serves
// `accept-ranges: bytes`, so an exact range makes per-row reading cheap.
//
// The ordering defaults to `max` over the per-subtask vector because that is what
// the LESS paper specifies, but max is lossy in a way that matters here - it hides
// whether a row was chosen for one behaviour or all of them - so the whole vector
// is always on screen and every component is a sort option.

import { useCallback, useEffect, useMemo, useState } from "react";
import { LoaderCircle, CloudOff } from "lucide-react";
import { describeLoadError, loadJsonDoc, loadJsonlAll, loadJsonlRow } from "@/lib/lazy";

export type SelectionSource = {
  /** HF `resolve` base for the repo holding scores and offsets. */
  base: string;
  scoresFile: string;
  offsetsFile: string;
};

type ScoreRow = {
  rank: number;
  less_id: string;
  score_max: number;
  score_mean: number;
  score_min: number;
  per_subtask: Record<string, number>;
  argmax_subtask: string;
  trait_id: string | null;
  in_warmup: boolean;
};

type Offsets = {
  repo: string;
  file: string;
  offsets: Record<string, [number, number]>;
};

type PoolRow = {
  messages: { role: string; content: string; reasoning_content?: string }[];
  metadata: { trait_name?: string };
};

const SUBTASK_COLOR = ["var(--cyan)", "var(--amber)", "var(--violet)"];
const fmt = (v: number) => (v >= 0 ? "+" : "") + v.toExponential(3);
const pretty = (s: string) => s.replace(/_/g, " ");

export function SelectionRankingViewer({ source }: { source: SelectionSource }) {
  const [rows, setRows] = useState<ScoreRow[] | null>(null);
  const [offsets, setOffsets] = useState<Offsets | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [order, setOrder] = useState("score_max");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  // Keyed by row id rather than cleared on selection change: a bare `setConvo(null)`
  // in the effect body is a synchronous setState that cascades a render, and keying
  // makes a stale conversation impossible to show without that.
  const [convo, setConvo] = useState<{ id: string; row: PoolRow } | null>(null);
  const [convoError, setConvoError] = useState<{ id: string; message: string } | null>(null);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [s, o] = await Promise.all([
          loadJsonlAll<ScoreRow>(`${source.base}/${source.scoresFile}`),
          loadJsonDoc<Offsets>(`${source.base}/${source.offsetsFile}`),
        ]);
        if (!live) return;
        setRows(s);
        setOffsets(o);
      } catch (e) {
        if (live) setError(describeLoadError(e, "Hugging Face"));
      }
    })();
    return () => {
      live = false;
    };
  }, [source]);

  const subtasks = useMemo(
    () => (rows?.length ? Object.keys(rows[0].per_subtask) : []),
    [rows],
  );

  const keyOf = useCallback(
    (r: ScoreRow) =>
      order.startsWith("sub:") ? r.per_subtask[order.slice(4)] : (r[order as keyof ScoreRow] as number),
    [order],
  );

  const view = useMemo(() => {
    if (!rows) return [];
    const q = query.trim().toLowerCase();
    const filtered = q
      ? rows.filter(
          (r) => r.less_id.toLowerCase().includes(q) || (r.trait_id || "").toLowerCase().includes(q),
        )
      : rows.slice();
    return filtered.sort((a, b) => keyOf(b) - keyOf(a));
  }, [rows, query, keyOf]);

  const vmax = useMemo(
    () => Math.max(1e-12, ...(rows || []).flatMap((r) => Object.values(r.per_subtask).map(Math.abs))),
    [rows],
  );

  // Selecting a row range-fetches exactly that conversation out of the 24 MB pool.
  useEffect(() => {
    if (!selected || !offsets) return;
    const span = offsets.offsets[selected];
    if (!span) return; // reported during render; see `missingFromIndex`
    let live = true;
    const id = selected;
    const url = `https://huggingface.co/datasets/${offsets.repo}/resolve/main/${offsets.file}`;
    loadJsonlRow<PoolRow>(url, span[0], span[1])
      .then((row) => live && setConvo({ id, row }))
      .catch((e) => live && setConvoError({ id, message: describeLoadError(e, "the scored pool") }));
    return () => {
      live = false;
    };
  }, [selected, offsets]);

  const jump = (index: number) => {
    const target = view[Math.max(0, Math.min(view.length - 1, index))];
    if (!target) return;
    setSelected(target.less_id);
    document
      .getElementById(`sel-row-${target.less_id}`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
  };

  if (error) {
    return (
      <div className="empty-state transcript-error">
        <CloudOff size={16} />
        <span>{error}</span>
      </div>
    );
  }
  if (!rows || !offsets) {
    return (
      <div className="empty-state">
        <LoaderCircle size={16} className="spin" />
        <span>Loading ranking from Hugging Face&hellip;</span>
      </div>
    );
  }

  const mid = Math.floor(view.length / 2);
  const negCount = rows.filter((r) => Math.min(...Object.values(r.per_subtask)) < 0).length;

  const bar = (value: number, i: number) => {
    const width = Math.min(50, (Math.abs(value) / vmax) * 50);
    const style = value >= 0 ? { left: "50%", width: `${width}%` } : { left: `${50 - width}%`, width: `${width}%` };
    return (
      <span className="sel-cell" key={i} title={`${pretty(subtasks[i])}: ${fmt(value)}`}>
        <span className="sel-zero" />
        <span
          className="sel-fill"
          style={{ ...style, background: value >= 0 ? SUBTASK_COLOR[i % 3] : "var(--red)" }}
        />
      </span>
    );
  };

  const current = view.find((r) => r.less_id === selected) || null;
  const currentIndex = current ? view.indexOf(current) : -1;
  // Derived, not stored: only the payload for the row currently selected counts.
  const shownConvo = convo && convo.id === selected ? convo.row : null;
  const shownError = convoError && convoError.id === selected ? convoError.message : null;
  const missingFromIndex = Boolean(selected && !offsets.offsets[selected]);

  return (
    <div className="sel-wrap">
      <div className="sel-stats">
        {[
          ["rows scored", rows.length.toLocaleString()],
          ["subtasks", String(subtasks.length)],
          ["best", rows[0]?.score_max.toExponential(2) ?? "-"],
          ["worst", rows[rows.length - 1]?.score_max.toExponential(2) ?? "-"],
          ["negative on some subtask", `${negCount} (${((100 * negCount) / rows.length).toFixed(1)}%)`],
        ].map(([k, v]) => (
          <div className="sel-stat" key={k}>
            <span className="sel-lbl">{k}</span>
            <b>{v}</b>
          </div>
        ))}
      </div>

      <div className="sel-controls">
        <span className="sel-lbl">Jump</span>
        <button type="button" onClick={() => jump(0)}>Top 10</button>
        <button type="button" onClick={() => jump(mid)}>Middle 10</button>
        <button type="button" onClick={() => jump(view.length - 1)}>Bottom 10</button>
        <span className="sel-spacer" />
        <label className="sel-lbl" htmlFor="sel-order">Sort</label>
        <select id="sel-order" value={order} onChange={(e) => setOrder(e.target.value)}>
          <option value="score_max">max (paper default)</option>
          <option value="score_mean">mean</option>
          <option value="score_min">min</option>
          {subtasks.map((s) => (
            <option key={s} value={`sub:${s}`}>by {pretty(s)}</option>
          ))}
        </select>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="filter id or trait"
          aria-label="Filter rows"
        />
      </div>

      <div className="sel-legend">
        {subtasks.map((s, i) => (
          <span key={s}>
            <i className="sel-sw" style={{ background: SUBTASK_COLOR[i % 3] }} />
            {pretty(s)}
          </span>
        ))}
        <span><i className="sel-sw" style={{ background: "var(--red)" }} />negative influence</span>
      </div>

      <div className="sel-main">
        <div className="sel-list" role="listbox" aria-label="Ranked rows" tabIndex={-1}>
          {view.map((r, i) => {
            const band =
              i < 10 ? " band-top" : i >= view.length - 10 && view.length > 20 ? " band-bot"
                : i >= mid - 5 && i < mid + 5 ? " band-mid" : "";
            const k = keyOf(r);
            return (
              <div
                key={r.less_id}
                id={`sel-row-${r.less_id}`}
                className={`sel-row${band}${selected === r.less_id ? " is-selected" : ""}`}
                role="option"
                aria-selected={selected === r.less_id}
                tabIndex={0}
                onClick={() => setSelected(r.less_id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSelected(r.less_id);
                  }
                }}
              >
                <span className="sel-rk">{i}</span>
                <span className="sel-id" title={r.less_id}>
                  {r.less_id}
                  {r.in_warmup && <i className="sel-warm" title="in the warmup split" />}
                </span>
                <span className="sel-trait" style={{ color: SUBTASK_COLOR[subtasks.indexOf(r.argmax_subtask) % 3] }}>
                  {r.trait_id}
                </span>
                <span className="sel-vec">{subtasks.map((s, j) => bar(r.per_subtask[s], j))}</span>
                <span className={`sel-score${k < 0 ? " is-neg" : ""}`}>{fmt(k)}</span>
              </div>
            );
          })}
        </div>

        <aside className="sel-detail">
          {!current ? (
            <div className="empty-state"><span>Select a row to read its conversation.</span></div>
          ) : (
            <>
              <div className="sel-block">
                <span className="sel-lbl">
                  Rank {currentIndex} of {view.length}
                  {current.in_warmup ? " · warmup split" : ""}
                </span>
                <div className="sel-mono">{current.less_id}</div>
              </div>
              <div className="sel-block">
                <span className="sel-lbl">Influence per subtask</span>
                {subtasks.map((s, i) => (
                  <div className="sel-srow" key={s}>
                    <span style={{ color: SUBTASK_COLOR[i % 3] }}>{pretty(s)}</span>
                    <span className="sel-vec">{bar(current.per_subtask[s], i)}</span>
                    <span className={current.per_subtask[s] < 0 ? "is-neg" : ""}>
                      {fmt(current.per_subtask[s])}
                    </span>
                  </div>
                ))}
                <div className="sel-srow sel-agg">
                  <span>max / mean / min</span>
                  <span />
                  <span>{`${fmt(current.score_max)}  ${fmt(current.score_mean)}  ${fmt(current.score_min)}`}</span>
                </div>
              </div>
              <div className="sel-block">
                <span className="sel-lbl">Trait {current.trait_id}</span>
                <div className="sel-note">{shownConvo?.metadata?.trait_name || ""}</div>
              </div>
              <div className="sel-block">
                <span className="sel-lbl">Conversation</span>
                {missingFromIndex ? (
                  <div className="empty-state transcript-error">
                    <CloudOff size={16} />
                    <span>This row is not in the pool offset index.</span>
                  </div>
                ) : shownError ? (
                  <div className="empty-state transcript-error">
                    <CloudOff size={16} />
                    <span>{shownError}</span>
                  </div>
                ) : !shownConvo ? (
                  <div className="empty-state">
                    <LoaderCircle size={16} className="spin" />
                    <span>Fetching this conversation&hellip;</span>
                  </div>
                ) : (
                  <Conversation row={shownConvo} />
                )}
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

function Conversation({ row }: { row: PoolRow }) {
  const system = row.messages.find((m) => m.role === "system");
  const user = row.messages.find((m) => m.role === "user");
  const assistant = row.messages.find((m) => m.role === "assistant");
  const turns: [string, string, string, boolean][] = [
    ["sys", "System prompt", system?.content || "", false],
    ["", "User", user?.content || "", true],
    // Labelled explicitly: the reasoning trace is supervised in training, so it is
    // part of what the gradients were taken over, not commentary alongside them.
    ["think", "Assistant reasoning — supervised in training", assistant?.reasoning_content || "", true],
    ["", "Assistant response", assistant?.content || "", true],
  ];
  return (
    <>
      {turns
        .filter(([, , text]) => text)
        .map(([cls, label, text, open]) => (
          <details className={`sel-turn ${cls}`} key={label} open={open}>
            <summary>
              <span>{label}</span>
              <span className="sel-cnt">{text.length.toLocaleString()} chars</span>
            </summary>
            <div className="sel-body">{text}</div>
          </details>
        ))}
    </>
  );
}
