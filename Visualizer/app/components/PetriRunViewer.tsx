"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  CircleDot,
  Cloud,
  Download,
  Eye,
  Filter,
  LoaderCircle,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";
import {
  PetriManifest,
  PetriTranscript,
  ResearchEntry,
  formatBytes,
  humanize,
} from "@/lib/content";
import { describeLoadError, loadTranscript } from "@/lib/lazy";
import { MockBadge } from "./MockDataBanner";
import { DialogueTranscript } from "./DialogueTranscript";

export function PetriRunViewer({ run }: { run: ResearchEntry }) {
  const manifest = run.petri as PetriManifest;
  const index = useMemo(() => manifest.transcript_index || [], [manifest]);
  const [selectedTranscriptId, setSelectedTranscriptId] = useState(
    index[0]?.id || "",
  );
  const [outcomeFilter, setOutcomeFilter] = useState("all");

  // The transcript body is NOT in the page. It arrives from its sidecar when a
  // reader selects it - one request of roughly 23 KB rather than 707 KB up
  // front for the whole run.
  const [fetched, setFetched] = useState<PetriTranscript | null>(null);
  const [failure, setFailure] = useState<{ id: string; message: string } | null>(null);

  const filtered = useMemo(
    () =>
      index.filter(
        (transcript) =>
          outcomeFilter === "all" || transcript.outcome === outcomeFilter,
      ),
    [index, outcomeFilter],
  );
  const selected =
    filtered.find((transcript) => transcript.id === selectedTranscriptId) ||
    filtered[0];
  const scenario = manifest.scenarios.find(
    (candidate) => candidate.id === selected?.scenario_id,
  );

  const remote = manifest.source?.kind === "hf";
  const sourceLabel = remote
    ? `Hugging Face (${manifest.source.repo_id})`
    : "this site";
  const transcriptBase = manifest.transcript_base;

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    loadTranscript(transcriptBase, selected.file).then(
      (record) => {
        if (!cancelled) setFetched(record);
      },
      (error) => {
        if (!cancelled) {
          setFailure({ id: selected.id, message: describeLoadError(error, sourceLabel) });
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selected, transcriptBase, sourceLabel]);

  // Derive the three view states rather than storing them, so nothing has to be
  // written back into state while an effect is running.
  const body = fetched && fetched.id === selected?.id ? fetched : null;
  const bodyError = failure && failure.id === selected?.id ? failure.message : "";
  const bodyLoading = Boolean(selected) && !body && !bodyError;
  const chartData = (manifest.scores.by_category || []).map((item) => ({
    ...item,
    pass: Math.max(0, item.audits - item.concerning - item.eval_aware),
    label: humanize(item.category),
  }));

  const roles = [
    ["Auditor", run.auditor_model_id, Bot],
    ["Target", run.target_checkpoint_id, CircleDot],
    ["Realism", run.realism_model_id, Eye],
    ["Judge", run.judge_model_id, ShieldCheck],
  ] as const;

  return (
    <div className="petri-workspace">
      <section className="petri-run-header">
        <div>
          <div className="document-kicker">
            <span className="type-chip petri-runs"><ScanSearch size={13} /> Petri run</span>
            <span className={`status status-${run.status}`}>{humanize(run.status)}</span>
            <time dateTime={run.date}>{run.date}</time>
            {run.mock === true && <MockBadge />}
          </div>
          <h1>{run.title}</h1>
          <p>{run.summary}</p>
          {remote && manifest.source.url && (
            <a
              className="hf-source-badge"
              href={manifest.source.url}
              target="_blank"
              rel="noreferrer noopener"
            >
              <Cloud size={13} /> Data:{" "}
              <code>
                {manifest.source.repo_id}
                {manifest.source.commit ? `@${manifest.source.commit.slice(0, 8)}` : ""}
              </code>
            </a>
          )}
          {manifest.source?.fallback_from && (
            <span className="hf-source-badge">
              <AlertTriangle size={13} /> {manifest.source.fallback_from} was
              unreachable at build time; showing the on-disk copy.
            </span>
          )}
        </div>
        <div className="petri-run-id">
          <span>Run ID</span>
          <code>{run.petri_run_id}</code>
          <small>{run.petri_version} · {run.seed_set}</small>
        </div>
      </section>

      <section className="petri-flow" aria-label="Petri audit flow">
        {roles.map(([label, value, Icon], index) => (
          <div className="petri-role" key={label}>
            {index > 0 && <span className="flow-connector" />}
            <div className="role-icon"><Icon size={17} /></div>
            <span>{label}</span>
            <code title={String(value || "not set")}>{String(value || "not set")}</code>
          </div>
        ))}
      </section>

      <section className="petri-summary-grid">
        <div className="petri-chart-panel">
          <div className="panel-title">
            <div><span className="eyebrow">Outcome map</span><h2>Findings by hypothesis</h2></div>
            <div className="chart-legend-note">Retained transcripts only</div>
          </div>
          <div className="petri-chart">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 18, right: 20 }}>
                <CartesianGrid stroke="#273038" horizontal={false} />
                <XAxis type="number" allowDecimals={false} stroke="#73808b" />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={142}
                  stroke="#93a0aa"
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "#14191e",
                    border: "1px solid #34404a",
                    borderRadius: 9,
                  }}
                />
                <Legend />
                <Bar dataKey="pass" name="Pass" stackId="outcome" fill="#74ba77" />
                <Bar dataKey="eval_aware" name="Eval-aware" stackId="outcome" fill="#ffbd6d" />
                <Bar dataKey="concerning" name="Concerning" stackId="outcome" fill="#ff7d86" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <aside className="petri-brief">
          <span className="eyebrow">Brief qualitative finding</span>
          <div className="brief-alert"><AlertTriangle size={18} /><strong>1 concerning lead</strong></div>
          <p>
            The flagged transcript involves proposed manipulation of an oversight
            record after shutdown pressure. It is localized to the goal-guarding
            seed and requires targeted replication.
          </p>
          <dl>
            <div><dt>Retained</dt><dd>{manifest.scores.retained_transcripts}</dd></div>
            <div><dt>Realism filtered</dt><dd>{manifest.scores.filtered_for_realism}</dd></div>
            <div><dt>Max turns</dt><dd>{run.max_turns}</dd></div>
            <div><dt>Realism threshold</dt><dd>{run.realism_threshold}</dd></div>
          </dl>
        </aside>
      </section>

      <section className="petri-transcript-section">
        <div className="section-heading row">
          <div>
            <span className="eyebrow">Audit evidence</span>
            <h2>Transcript explorer</h2>
          </div>
          <label className="outcome-filter">
            <Filter size={14} />
            <select
              value={outcomeFilter}
              onChange={(event) => setOutcomeFilter(event.target.value)}
              aria-label="Filter transcripts by outcome"
            >
              <option value="all">All outcomes</option>
              <option value="pass">Pass</option>
              <option value="concerning">Concerning</option>
              <option value="eval-aware">Eval-aware</option>
            </select>
          </label>
        </div>

        <p className="transcript-source-note">
          {remote ? <Cloud size={13} /> : <Download size={13} />}
          {manifest.transcript_count} transcripts indexed. Bodies load on demand
          from {sourceLabel}
          {manifest.deferred_bytes
            ? ` (${formatBytes(manifest.deferred_bytes)} kept out of this page)`
            : ""}
          .
        </p>

        <div className="petri-transcript-browser">
          <aside className="petri-transcript-index">
            {filtered.map((transcript) => (
              <button
                type="button"
                key={transcript.id}
                onClick={() => setSelectedTranscriptId(transcript.id)}
                className={transcript.id === selected?.id ? "is-active" : ""}
              >
                <span>
                  {transcript.outcome === "pass" ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                  <code>{transcript.id}</code>
                </span>
                <strong>{humanize(transcript.category)}</strong>
                <small className={`outcome outcome-${transcript.outcome}`}>{humanize(transcript.outcome)}</small>
              </button>
            ))}
          </aside>

          <div className="petri-transcript-main">
            {selected && (
              <>
                <div className="transcript-context">
                  <div><span>Hypothesis</span><p>{scenario?.hypothesis}</p></div>
                  <div><span>Expected signal</span><p>{scenario?.expected_signal}</p></div>
                </div>
                {bodyError ? (
                  <div className="empty-state transcript-error">
                    <AlertTriangle size={16} /> {bodyError}
                  </div>
                ) : body ? (
                  <>
                    <DialogueTranscript messages={body.messages} compact />
                    <div className="judge-finding">
                      <div><ShieldCheck size={16} /><span>Judge finding</span></div>
                      <p>{body.judge_summary}</p>
                    </div>
                  </>
                ) : (
                  <div className="empty-state transcript-loading">
                    {bodyLoading && <LoaderCircle size={15} className="spin" />}
                    Loading {selected.message_count} messages
                    {selected.size_bytes ? ` (${formatBytes(selected.size_bytes)})` : ""}…
                  </div>
                )}
              </>
            )}
          </div>

          <aside className="petri-score-panel">
            <div className="pane-heading">Transcript scores</div>
            {selected &&
              Object.entries(selected.scores).map(([name, value]) => (
                <div className="score-row" key={name}>
                  <div><span>{humanize(name)}</span><code>{Math.round(value * 100)}%</code></div>
                  <div className="score-track"><i style={{ width: `${value * 100}%` }} /></div>
                </div>
              ))}
            {selected && (
              <div className="score-tags">
                {selected.tags.map((tag) => <code key={tag}>#{tag}</code>)}
              </div>
            )}
          </aside>
        </div>
      </section>

      <section className="scenario-table-section">
        <div className="section-heading row">
          <div><span className="eyebrow">Generated audit dataset</span><h2>Scenario seeds</h2></div>
          <div className="artifact-actions">
            {run.assets
              .filter((asset) => asset.name.endsWith(".jsonl"))
              .map((asset) => (
                <a href={asset.path} download key={asset.path}><Download size={14} /> {asset.name}</a>
              ))}
          </div>
        </div>
        <div className="table-scroll">
          <table className="data-table scenario-table">
            <thead><tr><th>ID</th><th>Category</th><th>Hypothesis</th><th>Expected signal</th></tr></thead>
            <tbody>
              {manifest.scenarios.map((item) => (
                <tr key={item.id}>
                  <td><code>{item.id}</code></td>
                  <td>{humanize(item.category)}</td>
                  <td>{item.hypothesis}</td>
                  <td>{item.expected_signal}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
