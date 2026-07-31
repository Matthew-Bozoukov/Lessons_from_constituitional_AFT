import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CloudOff } from "lucide-react";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { MetricTiles } from "../components/MetricTiles";
import { PetriRunViewer } from "../components/PetriRunViewer";
import { entriesOfType } from "@/lib/content";
import { MockDataBanner } from "../components/MockDataBanner";
import { entryBody } from "@/lib/body";

export const metadata: Metadata = { title: "Petri audits" };

export default async function PetriPage() {
  const runs = entriesOfType("petri-runs");
  // Prefer a real run explicitly. This page shows one audit, and which one it
  // picks must not depend on index sort order - otherwise a fixture could
  // become the flagship result the day one happens to sort first. A fixture is
  // shown only when there is no real run to show, and then it is labelled.
  const run = runs.find((candidate) => candidate.mock !== true) || runs[0];
  // Only a genuinely empty collection is a 404. A run whose payload could not
  // be resolved still has a title, metrics and a research note worth rendering -
  // an unreachable dataset must degrade, not disappear.
  if (!run) notFound();

  return (
    <main className="page-container inner-page petri-page">
      {/* This page renders one run, so the warning tracks that run - not the
          collection. Warning about a real run because a fixture exists
          elsewhere would discredit the evidence it is meant to protect. */}
      {run.mock && <MockDataBanner />}
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
        <MarkdownRenderer markdown={await entryBody(run.slug)} />
      </section>
    </main>
  );
}
