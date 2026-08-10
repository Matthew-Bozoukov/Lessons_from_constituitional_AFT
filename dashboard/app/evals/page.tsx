import type { Metadata } from "next";
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { MetricExplorer } from "../components/MetricExplorer";
import { CorpusMix, KindChip } from "../components/CorpusMix";
import {
  ResearchEntry,
  allMock,
  anyMock,
  byKindThenDate,
  entriesOfType,
  entryKind,
  evalFamily,
  formatMetric,
  humanize,
} from "@/lib/content";
import { MockDataBanner } from "../components/MockDataBanner";

export const metadata: Metadata = { title: "Evals" };

/** Bounds of an interval are not a measurement on their own. */
const NOT_A_COLUMN = /^(ci_lower|ci_upper|.*_ci_(lower|upper)|n|total|count)$/i;

/**
 * Metrics worth a shared column, meaning most rows can actually fill them.
 *
 * The previous rule was "reported by more than one run", which on a real corpus
 * of six unrelated instruments promoted `n`, `truncation_rate`,
 * `accuracy_parsed_only` and `ci_lower` — each empty in 26 or 27 of 30 rows.
 * Half the table (121 of 240 cells) was an em-dash, which reads as "these runs
 * failed to report" rather than "these runs measure different things".
 *
 * A column now has to be filled by at least `minShare` of the runs, and an
 * interval bound can never be one: `ci_lower` in isolation, with no point
 * estimate and no upper bound beside it, is not a number anyone can read.
 */
function sharedMetricsFor(evals: ResearchEntry[], limit = 3, minShare = 0.34) {
  const frequency = new Map<string, number>();
  for (const entry of evals) {
    for (const name of Object.keys(entry.metrics)) {
      if (NOT_A_COLUMN.test(name)) continue;
      frequency.set(name, (frequency.get(name) || 0) + 1);
    }
  }
  const floor = Math.max(2, Math.ceil(evals.length * minShare));
  return [...frequency.entries()]
    .filter(([, count]) => count >= floor)
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

  // Grouped by instrument. A flat list put an MMLU accuracy directly above a
  // Petri flag count, inviting a comparison between numbers that share no
  // scale, no denominator and no question.
  const families = new Map<string, ResearchEntry[]>();
  for (const entry of [...evals].sort(byKindThenDate)) {
    families.set(evalFamily(entry), [...(families.get(evalFamily(entry)) || []), entry]);
  }
  const ordered = [...families.entries()].sort(
    (a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]),
  );

  return (
    <main className="page-container inner-page">
      {anyMock(evals) && (
        <MockDataBanner scope={allMock(evals) ? "all" : "some"} />
      )}
      <header className="page-heading">
        <div>
          <span className="eyebrow">Run-level evidence</span>
          <h1>Evaluation results</h1>
          <p>
            Every evaluation run against a checkpoint in this project, grouped by
            the instrument it used. Runs from different instruments are not
            comparable to each other — an MMLU accuracy and a red-teaming rate
            share no scale — so each group is read on its own terms.
          </p>
          <CorpusMix entries={evals} noun="runs" />
        </div>
      </header>

      <MetricExplorer entries={evals} />

      {ordered.map(([family, rows]) => (
        <section className="data-section" key={family}>
          <div className="section-heading row">
            <div>
              <span className="eyebrow">{family}</span>
              <h2>
                {rows.length} {rows.length === 1 ? "run" : "runs"}
              </h2>
            </div>
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
                {rows.map((entry) => {
                  // Everything this run measured that has no shared column.
                  // Listed per run rather than spread across columns, because
                  // these are different instruments and the numbers do not
                  // line up.
                  const own = Object.entries(entry.metrics).filter(
                    ([name]) => !sharedMetrics.includes(name),
                  );
                  const kind = entryKind(entry);
                  return (
                    <tr key={entry.id} className={kind === "stub" ? "is-stub" : undefined}>
                      <td>
                        <strong>{entry.run_id || entry.title}</strong>
                        <KindChip kind={kind} />
                        <small>{entry.checkpoint_id}</small>
                      </td>
                      {showStage && <td><code>{entry.training_stage || "—"}</code></td>}
                      {showSeed && <td>{entry.seed ?? "—"}</td>}
                      {sharedMetrics.map((name) => (
                        <td key={name}>
                          {entry.metrics[name] ? formatMetric(entry.metrics[name]) : "—"}
                        </td>
                      ))}
                      <td>
                        {/* "not reported" rather than an em-dash: a stub has no
                            numbers because nobody pulled them, not because the
                            run failed to produce any. */}
                        {own.length === 0 ? (
                          <span className="cell-none">not reported</span>
                        ) : (
                          <ul className="metric-chips">
                            {own.slice(0, 4).map(([name, metric]) => (
                              <li key={name}>
                                <span>{humanize(name)}</span>
                                <strong>{formatMetric(metric)}</strong>
                              </li>
                            ))}
                            {own.length > 4 && (
                              <li className="more">+{own.length - 4} more</li>
                            )}
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
      ))}
    </main>
  );
}
