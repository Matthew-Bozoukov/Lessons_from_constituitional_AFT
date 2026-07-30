import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { MetricExplorer } from "../components/MetricExplorer";
import { entriesOfType, formatMetric, humanize } from "@/lib/content";

export const metadata: Metadata = { title: "Evals" };

export default function EvalsPage() {
  const evals = entriesOfType("evals");
  return (
    <main className="page-container inner-page">
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
                <th>Run</th><th>Stage</th><th>Seed</th><th>Misalignment</th>
                <th>Adherence</th><th>Capability</th><th>Status</th><th />
              </tr>
            </thead>
            <tbody>
              {evals.map((entry) => (
                <tr key={entry.id}>
                  <td><strong>{entry.run_id || entry.title}</strong><small>{entry.checkpoint_id}</small></td>
                  <td><code>{entry.training_stage || "unknown"}</code></td>
                  <td>{entry.seed ?? "—"}</td>
                  <td>{entry.metrics.agentic_misalignment_rate ? formatMetric(entry.metrics.agentic_misalignment_rate) : "—"}</td>
                  <td>{entry.metrics.constitution_adherence ? formatMetric(entry.metrics.constitution_adherence) : "—"}</td>
                  <td>{entry.metrics.capability_retention ? formatMetric(entry.metrics.capability_retention) : "—"}</td>
                  <td><span className={`status status-${entry.status}`}>{humanize(entry.status)}</span></td>
                  <td><Link aria-label={`Open ${entry.title}`} href={`/entry/${entry.slug}`}><ArrowUpRight size={16} /></Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

