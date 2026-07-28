import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { MetricTiles } from "../components/MetricTiles";
import { PetriRunViewer } from "../components/PetriRunViewer";
import { entriesOfType } from "@/lib/content";

export const metadata: Metadata = { title: "Petri audits" };

export default function PetriPage() {
  const runs = entriesOfType("petri-runs");
  const run = runs[0];
  if (!run?.petri) notFound();

  return (
    <main className="page-container inner-page petri-page">
      <PetriRunViewer run={run} />
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

