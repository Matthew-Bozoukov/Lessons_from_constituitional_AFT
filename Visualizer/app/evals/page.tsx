import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { MetricExplorer } from "../components/MetricExplorer";
import {
  ResearchEntry,
  allMock,
  anyMock,
  entriesOfType,
  formatMetric,
  humanize,
} from "@/lib/content";
import { MockDataBanner } from "../components/MockDataBanner";

export const metadata: Metadata = { title: "Evals" };

/**
 * Metrics reported by more than one run, and therefore worth a shared column.
 *
 * This table used to hardcode three metric names belonging to a single eval
 * suite. Every run measuring anything else rendered as a row of dashes - and
 * once the corpus became real, that was every run, because each instrument
 * measures its own thing. A column that only one run can fill is not a
 * comparison; it is a dash factory that implies the others failed to report.
 */
function sharedMetricsFor(evals: ResearchEntry[], limit = 4) {
  const frequency = new Map<string, number>();
  for (const entry of evals) {
    for (const name of Object.keys(entry.metrics)) {
      frequency.set(name, (frequency.get(name) || 0) + 1);
    }
  }
  return [...frequency.entries()]
    .filter(([, count]) => count > 1)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([name]) => name);
}

/** A column only earns its place if some run actually fills it. */
function anyHas(evals: ResearchEntry[], field: keyof ResearchEntry) {
  return evals.some((entry) => entry[field] !== undefined && entry[field] !== "");
}

export default function EvalsPage() {
  const evals = entriesOfType("evals");
  const sharedMetrics = sharedMetricsFor(evals);
  const showStage = anyHas(evals, "training_stage");
  const showSeed = anyHas(evals, "seed");
  return (
    <main className="page-container inner-page">
      {anyMock(evals) && (
        <MockDataBanner scope={allMock(evals) ? "all" : "some"} />
      )}
      <header className="page-heading">
        <div>
          <span className="eyebrow">Compatible evidence groups</span>
          <h1>Evaluation results</h1>
          <p>
            Structured metrics remain optional. When present, they unlock safe
            stage comparisons without hiding suite, dataset, version, or seed.
          </p>
        </div>
      </header>

      <MetricExplorer entries={evals} />

      <section className="data-section">
        <div className="section-heading row">
          <div>
            <span className="eyebrow">Run-level evidence</span>
            <h2>Evaluation index</h2>
          </div>
          <span className="table-note">{evals.length} structured records</span>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                {showStage && <th>Stage</th>}
                {showSeed && <th>Seed</th>}
                {sharedMetrics.map((name) => <th key={name}>{humanize(name)}</th>)}
                <th>Reported</th>
                <th>Status</th><th />
              </tr>
            </thead>
            <tbody>
              {evals.map((entry) => {
                // Everything this run measured that has no shared column. Listed
                // per run rather than spread across columns, because these are
                // different instruments and the numbers do not line up.
                const own = Object.entries(entry.metrics).filter(
                  ([name]) => !sharedMetrics.includes(name),
                );
                return (
                  <tr key={entry.id}>
                    <td><strong>{entry.run_id || entry.title}</strong><small>{entry.checkpoint_id}</small></td>
                    {showStage && <td><code>{entry.training_stage || "—"}</code></td>}
                    {showSeed && <td>{entry.seed ?? "—"}</td>}
                    {sharedMetrics.map((name) => (
                      <td key={name}>
                        {entry.metrics[name] ? formatMetric(entry.metrics[name]) : "—"}
                      </td>
                    ))}
                    <td>
                      {own.length === 0 ? (
                        "—"
                      ) : (
                        <ul className="metric-chips">
                          {own.map(([name, metric]) => (
                            <li key={name}>
                              <span>{humanize(name)}</span>
                              <strong>{formatMetric(metric)}</strong>
                            </li>
                          ))}
                        </ul>
                      )}
                    </td>
                    <td><span className={`status status-${entry.status}`}>{humanize(entry.status)}</span></td>
                    <td><Link aria-label={`Open ${entry.title}`} href={`/entry/${entry.slug}`}><ArrowUpRight size={16} /></Link></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

