"use client";

// ABOUTME: Interactive explorer for the org's eval-run repos on HF: pick an eval type and
// ABOUTME: run (or two, to compare), then read results side by side and aligned rollouts.

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, LoaderCircle } from "lucide-react";
import { describeLoadError } from "@/lib/lazy";
import { loadJsonlAll } from "@/lib/lazy";
import {
  EvalRun, JudgeVerdicts, Json, JsonlSpec, TreeItem, VerdictSpec, adapterFor, flattenMetrics,
  listEvalRuns, listRolloutFiles, loadJudgeVerdicts, loadResults, loadText, medianScore,
  parseStepTranscript, repoDate, repoUrl, resolveUrl,
} from "@/lib/evalRuns";
import { DialogueTranscript } from "./DialogueTranscript";
import { MarkdownRenderer } from "./MarkdownRenderer";

const JSONL_AUTO_LIMIT = 6_000_000; // above this, rollout rows load on click, not on mount

type ResultsState = { data?: Json; error?: string };
type VerdictState = { data?: JudgeVerdicts; error?: string };
type RolloutState = {
  units?: Map<string, TreeItem[]>; // tree adapters
  rows?: Record<string, Json>; // jsonl adapters
  pendingBytes?: number; // jsonl above the auto limit, waiting for a click
  error?: string;
  loading?: boolean;
};

function fmt(value: number): string {
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function firstField(row: Json, names: string[]): string {
  for (const name of names) {
    const v = row[name];
    if (typeof v === "string" && v.trim()) return v;
  }
  return "";
}

function runTitle(run: EvalRun): string {
  const date = repoDate(run.repo);
  return [run.model, run.mode, date].filter(Boolean).join(" · ");
}

export function EvalRunExplorer({ org }: { org?: string }) {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [runsError, setRunsError] = useState("");
  const [evalName, setEvalName] = useState("");
  const [repoA, setRepoA] = useState("");
  const [repoB, setRepoB] = useState("");
  const [compare, setCompare] = useState(false);
  const [tab, setTab] = useState<"results" | "rollouts">("results");
  const [results, setResults] = useState<Record<string, ResultsState>>({});
  const [rollouts, setRollouts] = useState<Record<string, RolloutState>>({});
  const [unit, setUnit] = useState("");
  const [texts, setTexts] = useState<Record<string, { text?: string; error?: string }>>({});
  const [verdicts, setVerdicts] = useState<Record<string, VerdictState>>({});

  // -- discovery ------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    listEvalRuns(org)
      .then((found) => {
        if (cancelled) return;
        setRuns(found);
        const first = found[0];
        if (first) {
          setEvalName((current) => current || first.evalName);
          setRepoA((current) => current || first.repo);
        }
      })
      .catch((error) => {
        if (!cancelled) setRunsError(describeLoadError(error, "Hugging Face"));
      });
    return () => { cancelled = true; };
  }, [org]);

  const evalTypes = useMemo(
    () => [...new Set((runs || []).map((r) => r.evalName))].sort(),
    [runs],
  );
  const ofType = useMemo(
    () => (runs || []).filter((r) => r.evalName === evalName),
    [runs, evalName],
  );
  const adapter = useMemo(() => adapterFor(evalName), [evalName]);
  const active = useMemo(
    () => [repoA, compare ? repoB : ""].filter(Boolean),
    [repoA, repoB, compare],
  );

  function pickEval(next: string) {
    setEvalName(next);
    const first = (runs || []).find((r) => r.evalName === next);
    setRepoA(first ? first.repo : "");
    setRepoB("");
    setUnit("");
  }

  // -- results --------------------------------------------------------------
  useEffect(() => {
    for (const repo of active) {
      if (results[repo]) continue;
      loadResults(repo)
        .then((data) => setResults((s) => ({ ...s, [repo]: { data } })))
        .catch((error) =>
          setResults((s) => ({ ...s, [repo]: { error: describeLoadError(error, `Hugging Face (${repo})`) } })));
    }
  }, [active, results]);

  // -- rollout indexes ------------------------------------------------------
  // A repo with no entry in `rollouts` is by definition still loading (the views
  // derive their spinner from that). No cancel flag and no dedupe ref: the network
  // is deduped by `cached()` in evalRuns, the state write is keyed and idempotent,
  // and a cancel flag here silently dropped results whenever ANY state change
  // re-ran the effect — the permanent "Indexing rollouts…" bug.
  useEffect(() => {
    if (tab !== "rollouts") return;
    for (const repo of active) {
      if (rollouts[repo]) continue;
      const spec = adapter.rollouts;
      (async () => {
        const files = await listRolloutFiles(repo);
        if (spec.kind === "tree") return { units: spec.group(files) } as RolloutState;
        const entry = files.find((f) => f.path === spec.file);
        if (!entry) throw new Error(`rollouts/${spec.file} not found in ${repo}`);
        if (entry.size > JSONL_AUTO_LIMIT) return { pendingBytes: entry.size } as RolloutState;
        return { rows: await loadRows(repo, spec) } as RolloutState;
      })()
        .then((state) => setRollouts((s) => ({ ...s, [repo]: state })))
        .catch((error) =>
          setRollouts((s) => ({ ...s, [repo]: { error: describeLoadError(error, `Hugging Face (${repo})`) } })));
    }
  }, [tab, active, rollouts, adapter]);

  async function loadRows(repo: string, spec: JsonlSpec): Promise<Record<string, Json>> {
    const files = await listRolloutFiles(repo);
    const entry = files.find((f) => f.path === spec.file);
    if (!entry) throw new Error(`${spec.file} not found in ${repo}`);
    const rows = await loadJsonlAll<Json>(resolveUrl(repo, entry.full));
    const keyed: Record<string, Json> = {};
    for (const row of rows) {
      const key = spec.keyFields.map((k) => row[k]).find((v) => v !== undefined);
      if (key !== undefined) keyed[String(key)] = row;
    }
    return keyed;
  }

  function forceLoadRows(repo: string) {
    const spec = adapter.rollouts;
    if (spec.kind !== "jsonl") return;
    setRollouts((s) => ({ ...s, [repo]: { loading: true } }));
    loadRows(repo, spec)
      .then((rows) => setRollouts((s) => ({ ...s, [repo]: { rows } })))
      .catch((error) =>
        setRollouts((s) => ({ ...s, [repo]: { error: describeLoadError(error, `Hugging Face (${repo})`) } })));
  }

  const unitKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const repo of active) {
      const state = rollouts[repo];
      if (state?.units) for (const k of state.units.keys()) keys.add(k);
      if (state?.rows) for (const k of Object.keys(state.rows)) keys.add(k);
    }
    return [...keys].sort();
  }, [active, rollouts]);

  // Derived, never synced in an effect: a stale selection (eval switch, new keys)
  // falls back to the first unit without a cascading render.
  const effectiveUnit = unitKeys.includes(unit) ? unit : unitKeys[0] || "";

  // -- unit texts (tree adapters fetch each file of the selected unit) ------
  useEffect(() => {
    if (tab !== "rollouts" || !effectiveUnit) return;
    for (const repo of active) {
      for (const item of rollouts[repo]?.units?.get(effectiveUnit) || []) {
        const key = `${repo}:${item.path}`;
        if (texts[key]) continue;
        loadText(resolveUrl(repo, item.path))
          .then((text) => setTexts((s) => ({ ...s, [key]: { text } })))
          .catch((error) =>
            setTexts((s) => ({ ...s, [key]: { error: describeLoadError(error, `Hugging Face (${repo})`) } })));
      }
    }
  }, [tab, effectiveUnit, active, rollouts, texts]);

  // -- judge verdicts (evals whose adapter declares them) ---------------------
  // Same shape as the results effect: keyed idempotent writes, network deduped by
  // `cached()`, no cancel flag (see the rollout-index effect for why).
  useEffect(() => {
    if (tab !== "rollouts" || !adapter.verdicts) return;
    for (const repo of active) {
      if (verdicts[repo]) continue;
      loadJudgeVerdicts(repo)
        .then((data) => setVerdicts((s) => ({ ...s, [repo]: { data } })))
        .catch((error) =>
          setVerdicts((s) => ({ ...s, [repo]: { error: describeLoadError(error, `Hugging Face (${repo})`) } })));
    }
  }, [tab, active, verdicts, adapter]);

  // -- render ---------------------------------------------------------------
  if (runsError) return <div className="empty-state">{runsError}</div>;
  if (!runs) {
    return (
      <div className="empty-state">
        <LoaderCircle size={13} className="spin" /> Listing eval runs on Hugging Face…
      </div>
    );
  }
  if (!runs.length) {
    return (
      <div className="empty-state">
        No eval-run repos found. A run appears here once its HF repo is public and its
        card carries the <code>eval-run</code> tag — stamped automatically by
        <code> run_eval.py</code> pushes since 2026-08-24.
      </div>
    );
  }

  const runA = ofType.find((r) => r.repo === repoA);
  const runB = ofType.find((r) => r.repo === repoB);
  const showCompare = compare && Boolean(runB);

  return (
    <div className="eval-explorer">
      <div className="explorer-controls">
        <div className="controls-row">
          <div className="control-group">
            <label htmlFor="ee-eval">Eval</label>
            <select id="ee-eval" value={evalName} onChange={(e) => pickEval(e.target.value)}>
              {evalTypes.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <label className="compare-toggle">
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => {
                setCompare(e.target.checked);
                if (e.target.checked && !repoB) {
                  const other = ofType.find((r) => r.repo !== repoA);
                  if (other) setRepoB(other.repo);
                }
              }}
            />
            Compare
          </label>
          <div className="view-tabs" role="tablist">
            {(["results", "rollouts"] as const).map((t) => (
              <button
                key={t} role="tab" aria-selected={tab === t}
                className={tab === t ? "is-active" : ""} onClick={() => setTab(t)}
              >
                {t === "results" ? "Results" : "Rollouts"}
              </button>
            ))}
          </div>
        </div>
        <div className="controls-row">
          <div className="control-group grow">
            <label htmlFor="ee-run-a">{showCompare ? "Run A" : "Run"}</label>
            <select id="ee-run-a" value={repoA} onChange={(e) => setRepoA(e.target.value)}>
              {ofType.map((r) => <option key={r.repo} value={r.repo}>{runTitle(r)}</option>)}
            </select>
          </div>
          {compare && (
            <div className="control-group grow">
              <label htmlFor="ee-run-b">Run B</label>
              <select id="ee-run-b" value={repoB} onChange={(e) => setRepoB(e.target.value)}>
                {ofType.filter((r) => r.repo !== repoA).map((r) => (
                  <option key={r.repo} value={r.repo}>{runTitle(r)}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {(runA || runB) && (
        <div className="run-legend">
          {runA && (
            <span className="legend-chip">
              <i className="swatch swatch-a" />{showCompare ? "A — " : ""}{runTitle(runA)}
              <a href={repoUrl(runA.repo)} target="_blank" rel="noreferrer" aria-label="Open run A on Hugging Face">
                <ArrowUpRight size={13} />
              </a>
            </span>
          )}
          {showCompare && runB && (
            <span className="legend-chip">
              <i className="swatch swatch-b" />B — {runTitle(runB)}
              <a href={repoUrl(runB.repo)} target="_blank" rel="noreferrer" aria-label="Open run B on Hugging Face">
                <ArrowUpRight size={13} />
              </a>
            </span>
          )}
        </div>
      )}

      {tab === "results" ? (
        <ResultsView
          a={results[repoA]} b={showCompare ? results[repoB] : undefined}
          featured={adapter.featured} compare={showCompare}
        />
      ) : (
        <RolloutsView
          active={active} rollouts={rollouts} unit={effectiveUnit} unitKeys={unitKeys}
          setUnit={setUnit} texts={texts} adapter={adapter} forceLoadRows={forceLoadRows}
          verdicts={verdicts}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function ResultsView({ a, b, featured, compare }: {
  a?: ResultsState; b?: ResultsState; featured: string[]; compare: boolean;
}) {
  if (!a) return <div className="empty-state"><LoaderCircle size={13} className="spin" /> Loading results…</div>;
  if (a.error) return <div className="empty-state">{a.error}</div>;
  if (compare && b?.error) return <div className="empty-state">{b.error}</div>;
  if (compare && !b?.data) {
    return <div className="empty-state"><LoaderCircle size={13} className="spin" /> Loading run B results…</div>;
  }

  const flatA = flattenMetrics(a.data || {});
  const flatB = compare ? flattenMetrics(b?.data || {}) : {};
  const ordered = [
    ...featured.filter((k) => k in flatA || k in flatB),
    ...Object.keys({ ...flatA, ...flatB }).filter((k) => !featured.includes(k)).sort(),
  ];
  if (!ordered.length) return <div className="empty-state">results/results.json holds no numeric fields.</div>;

  // Bar charts only where an axis can honestly be shared: metrics cluster by scale.
  // Percent-named keys chart together; values living in [0, 1] chart together; anything
  // else (counts, dollars, negatives, mixed scales) stays in the table below.
  const values = (k: string) =>
    [flatA[k], flatB[k]].filter((v): v is number => typeof v === "number");
  const pctKeys = ordered.filter(
    (k) => /(_pct|_percent)$/.test(k) && values(k).every((v) => v >= 0 && v <= 100),
  );
  const unitKeys = ordered.filter(
    (k) => !pctKeys.includes(k) && values(k).length > 0
      && values(k).every((v) => v >= 0 && v <= 1) && values(k).some((v) => !Number.isInteger(v)),
  );
  const tableKeys = ordered.filter((k) => !pctKeys.includes(k) && !unitKeys.includes(k));

  return (
    <div>
      <MetricBarChart title="Percent metrics" keys={pctKeys} flatA={flatA} flatB={flatB} compare={compare} />
      <MetricBarChart title="0–1 scale metrics" keys={unitKeys} flatA={flatA} flatB={flatB} compare={compare} />
      {tableKeys.length > 0 && (
        <div className="table-scroll">
          <table className="data-table metric-compare-table">
            <thead>
              <tr>
                <th>Other metrics</th>
                <th>{compare ? "A" : "Value"}</th>
                {compare && <th>B</th>}
                {compare && <th>Δ (B − A)</th>}
              </tr>
            </thead>
            <tbody>
              {tableKeys.map((key) => {
                const va = flatA[key];
                const vb = flatB[key];
                const both = typeof va === "number" && typeof vb === "number";
                return (
                  <tr key={key} className={featured.includes(key) ? "is-featured" : ""}>
                    <td><code>{key}</code></td>
                    <td>{typeof va === "number" ? fmt(va) : "—"}</td>
                    {compare && <td>{typeof vb === "number" ? fmt(vb) : "—"}</td>}
                    {compare && <td>{both ? fmt(vb - va) : "—"}</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="chart-note">
        Each chart shares one axis only within its own scale; unshareable metrics stay
        tabular. Read from <code>results/results.json</code> in each run&apos;s repo.
      </p>
    </div>
  );
}

function MetricBarChart({ title, keys, flatA, flatB, compare }: {
  title: string; keys: string[];
  flatA: Record<string, number>; flatB: Record<string, number>; compare: boolean;
}) {
  if (!keys.length) return null;
  const max = Math.max(...keys.flatMap((k) => [flatA[k] ?? 0, flatB[k] ?? 0])) || 1;
  return (
    <div className="metric-chart">
      <div className="metric-chart-title"><span>{title}</span><span>axis 0 – {fmt(max)}</span></div>
      {keys.map((key) => {
        const va = flatA[key];
        const vb = flatB[key];
        return (
          <div className="mchart-row" key={key} role="img"
            aria-label={compare ? `${key}: A ${fmt(va ?? 0)}, B ${fmt(vb ?? 0)}` : `${key}: ${fmt(va ?? 0)}`}>
            <code className="mchart-label">{key}</code>
            <div className="mchart-bars">
              <div className="mchart-bar-line">
                <div className="mc-track">
                  {typeof va === "number" && <i className="mc-bar mc-a" style={{ width: `${(va / max) * 100}%` }} />}
                </div>
                <span className="mchart-val">{typeof va === "number" ? fmt(va) : "—"}</span>
              </div>
              {compare && (
                <div className="mchart-bar-line">
                  <div className="mc-track">
                    {typeof vb === "number" && <i className="mc-bar mc-b" style={{ width: `${(vb / max) * 100}%` }} />}
                  </div>
                  <span className="mchart-val">{typeof vb === "number" ? fmt(vb) : "—"}</span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------

function RolloutsView({ active, rollouts, unit, unitKeys, setUnit, texts, adapter, forceLoadRows, verdicts }: {
  active: string[];
  rollouts: Record<string, RolloutState>;
  unit: string;
  unitKeys: string[];
  setUnit: (u: string) => void;
  texts: Record<string, { text?: string; error?: string }>;
  adapter: ReturnType<typeof adapterFor>;
  forceLoadRows: (repo: string) => void;
  verdicts: Record<string, VerdictState>;
}) {
  // One pass selection shared by both panes, so A and B always show the same pass.
  const [passLabel, setPassLabel] = useState("");
  const pending = active.filter((repo) => rollouts[repo]?.pendingBytes);
  const loading = active.filter((repo) => rollouts[repo]?.loading || !rollouts[repo]);
  const errors = active.map((repo) => rollouts[repo]?.error).filter(Boolean);

  if (errors.length) return <div className="empty-state">{errors[0]}</div>;
  if (pending.length) {
    const repo = pending[0];
    const mb = ((rollouts[repo]?.pendingBytes || 0) / 1e6).toFixed(1);
    return (
      <div className="empty-state">
        <p>This run&apos;s rollout file is {mb} MB.</p>
        <button className="load-rollouts" onClick={() => forceLoadRows(repo)}>Load {mb} MB of rollouts</button>
      </div>
    );
  }
  if (loading.length) {
    return <div className="empty-state"><LoaderCircle size={13} className="spin" /> Indexing rollouts…</div>;
  }
  if (!unitKeys.length) return <div className="empty-state">No rollouts found under rollouts/ in this run.</div>;

  const index = unitKeys.indexOf(unit);
  return (
    <div>
      <div className="unit-nav">
        <button disabled={index <= 0} onClick={() => setUnit(unitKeys[index - 1])}>‹ Prev</button>
        <select value={unit} onChange={(e) => setUnit(e.target.value)} aria-label="Rollout unit">
          {unitKeys.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
        <button disabled={index >= unitKeys.length - 1} onClick={() => setUnit(unitKeys[index + 1])}>Next ›</button>
        <span className="unit-count">{index + 1} / {unitKeys.length}</span>
      </div>
      <div className={`rollout-panes ${active.length > 1 ? "is-compare" : ""}`}>
        {active.map((repo, i) => (
          <RolloutPane
            key={repo} repo={repo} label={active.length > 1 ? (i === 0 ? "A" : "B") : ""}
            state={rollouts[repo]} unit={unit} texts={texts} adapter={adapter}
            passLabel={passLabel} setPassLabel={setPassLabel} verdicts={verdicts[repo]}
          />
        ))}
      </div>
    </div>
  );
}

function RolloutPane({ repo, label, state, unit, texts, adapter, passLabel, setPassLabel, verdicts }: {
  repo: string; label: string; state?: RolloutState; unit: string;
  texts: Record<string, { text?: string; error?: string }>;
  adapter: ReturnType<typeof adapterFor>;
  passLabel: string; setPassLabel: (l: string) => void;
  verdicts?: VerdictState;
}) {
  const items = state?.units?.get(unit);
  const row = state?.rows?.[unit];
  // The shared pass selection falls back to this pane's first item when the
  // selected pass does not exist here (e.g. run B has one pass fewer).
  const activeItem =
    items && (items.find((i) => i.label === passLabel) || items[0]);
  return (
    <div className="rollout-pane">
      {label && <div className={`pane-title pane-${label.toLowerCase()}`}><i className={`swatch swatch-${label.toLowerCase()}`} />{label} — {repo.split("/")[1]}</div>}
      {!items && !row && <div className="pane-missing">Not present in this run.</div>}
      {items && items.length > 1 && (
        <div className="pass-tabs" role="tablist">
          {items.map((item) => (
            <button
              key={item.label} role="tab" aria-selected={item === activeItem}
              className={item === activeItem ? "is-active" : ""}
              onClick={() => setPassLabel(item.label)}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
      {activeItem && <PaneDocument repo={repo} item={activeItem} texts={texts} />}
      {activeItem && adapter.verdicts && (
        <JudgePanel spec={adapter.verdicts} state={verdicts} unit={unit} itemLabel={activeItem.label} />
      )}
      {row && <JsonlRow row={row} spec={adapter.rollouts as JsonlSpec} />}
    </div>
  );
}

// Every judge's grade and rationale for the rollout on screen, plus the median the
// eval actually scores on. A pass the judge never saw (dropped, or an unjudged cell)
// says so rather than showing a neighbour's verdict.
function JudgePanel({ spec, state, unit, itemLabel }: {
  spec: VerdictSpec; state?: VerdictState; unit: string; itemLabel: string;
}) {
  if (!state) {
    return <div className="judge-panel"><div className="pane-missing"><LoaderCircle size={13} className="spin" /> Loading judge verdicts…</div></div>;
  }
  if (state.error) return <div className="judge-panel"><div className="pane-missing">{state.error}</div></div>;
  const table = state.data?.judges || {};
  const judges = Object.keys(table).sort();
  if (!judges.length) return null; // this run published no scores_<judge>.json
  const key = spec.keysFor(unit, itemLabel, state.data?.keptPasses ?? null)
    .find((k) => judges.some((j) => k in table[j]));
  const rows = judges.map((judge) => ({ judge, verdict: key ? table[judge][key] : undefined }));
  const median = medianScore(rows.map((r) => r.verdict?.score ?? null));
  const tone = median === null ? "" : median >= spec.violationAt ? "is-violation" : "is-clean";
  const scoreText = (v?: { score: number | null }) =>
    v === undefined ? "—" : v.score === null ? "N/A" : fmt(v.score);
  return (
    <section className="judge-panel" aria-label="Judge verdicts">
      <div className="judge-head">
        <span className="judge-title">Judges</span>
        <span className={`judge-median ${tone}`}>
          {median === null
            ? "not judged"
            : `median ${fmt(median)} · ${median >= spec.violationAt ? "violation" : "no violation"} (threshold ${spec.violationAt})`}
        </span>
      </div>
      {rows.map(({ judge, verdict }) => (
        <details key={judge} open>
          <summary>
            <code>{judge}</code>
            <b className={`judge-score ${verdict?.score !== null && verdict !== undefined && verdict.score >= spec.violationAt ? "is-violation" : ""}`}>
              {scoreText(verdict)}
            </b>
          </summary>
          {verdict?.reasoning && <p className="judge-reasoning">{verdict.reasoning}</p>}
        </details>
      ))}
    </section>
  );
}

function PaneDocument({ repo, item, texts }: {
  repo: string; item: TreeItem;
  texts: Record<string, { text?: string; error?: string }>;
}) {
  const doc = texts[`${repo}:${item.path}`];
  if (!doc) return <div className="pane-missing"><LoaderCircle size={13} className="spin" /> Loading…</div>;
  if (doc.error) return <div className="pane-missing">{doc.error}</div>;
  if (doc.text === undefined) return null;
  if (item.render === "markdown") {
    return <div className="markdown-body"><MarkdownRenderer markdown={doc.text} /></div>;
  }
  const dialogue = item.render === "text" ? parseStepTranscript(doc.text) : null;
  if (dialogue) return <DialogueTranscript messages={dialogue} compact />;
  return <pre className="rollout-pre">{item.render === "json" ? prettyJson(doc.text) : doc.text}</pre>;
}

function prettyJson(text: string): string {
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

function JsonlRow({ row, spec }: { row: Json; spec: JsonlSpec }) {
  const prompt = firstField(row, spec.prompt);
  const reasoning = firstField(row, spec.reasoning);
  const response = firstField(row, spec.response);
  const used = new Set([...spec.prompt, ...spec.reasoning, ...spec.response, ...spec.keyFields]);
  const rest = Object.fromEntries(Object.entries(row).filter(([k]) => !used.has(k)));
  return (
    <div className="jsonl-row">
      {prompt && <><h4>Prompt</h4><pre className="rollout-pre">{prompt}</pre></>}
      {reasoning && (
        <details><summary>Reasoning trace</summary><pre className="rollout-pre">{reasoning}</pre></details>
      )}
      {response && <><h4>Response</h4><pre className="rollout-pre">{response}</pre></>}
      {Object.keys(rest).length > 0 && (
        <details><summary>Other fields</summary><pre className="rollout-pre">{JSON.stringify(rest, null, 2)}</pre></details>
      )}
      {!prompt && !response && (
        <pre className="rollout-pre">{JSON.stringify(row, null, 2)}</pre>
      )}
    </div>
  );
}
