import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CloudOff } from "lucide-react";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { MetricTiles } from "../components/MetricTiles";
import { PetriRunViewer } from "../components/PetriRunViewer";
import { entriesOfType } from "@/lib/content";

export const metadata: Metadata = { title: "Petri audits" };

export default function PetriPage() {
  const runs = entriesOfType("petri-runs");
  const run = runs[0];
  // Only a genuinely empty collection is a 404. A run whose payload could not
  // be resolved still has a title, metrics and a research note worth rendering -
  // an unreachable dataset must degrade, not disappear.
  if (!run) notFound();

  return (
    <main className="page-container inner-page petri-page">
      {run.petri ? (
        <PetriRunViewer run={run} />
      ) : (
        <section className="petri-run-header">
          <div>
            <h1>{run.title}</h1>
            <p>{run.summary}</p>
            <div className="empty-state transcript-error">
              <CloudOff size={16} />
              <span>
                Audit evidence for this run is unavailable
                {run.hf ? ` from ${run.hf.repo_id}` : ""}.
                {run.hf?.message ? ` ${run.hf.message}.` : ""} Run-level metrics
                and the research note below are unaffected.
              </span>
            </div>
          </div>
        </section>
      )}
      <section className="petri-metrics-section">
        <span className="eyebrow">Run-level metrics</span>
        <MetricTiles metrics={run.metrics} />
      </section>
      <section className="petri-notes">
        <div className="section-heading">
          <span className="eyebrow">Research note</span>
          <h2>Interpretation and follow-up</h2>
        </div>
        <MarkdownRenderer markdown={run.body} />
      </section>
    </main>
  );
}
