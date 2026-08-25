import type { Metadata } from "next";
import Link from "next/link";
import { TrainingDataExplorer } from "../components/TrainingDataExplorer";

export const metadata: Metadata = { title: "Synthetic datasets" };

export default function DatasetsPage() {
  return (
    <main className="page-container inner-page datasets-page">
      <header className="page-heading compact-heading">
        <div>
          <span className="eyebrow">Training-data inspection</span>
          <h1>Synthetic datasets</h1>
          <p>
            Every public training corpus in the org, discovered live from Hugging Face
            by its card tags — the synth runs, the mixtures built from them, and the
            property-ablation arms. Each blends ordinary instruction data with some
            share of constitution-grounded synthetic conversations; that share is what
            the experiments vary, so it leads every entry below. Records are read
            straight from the published JSONL a page at a time, so what you see is the
            file, not a copy of it that can drift.{" "}
            <Link href="/glossary">Unfamiliar terms are in the glossary</Link>.
          </p>
        </div>
      </header>

      <section className="data-section">
        <TrainingDataExplorer />
      </section>
    </main>
  );
}
