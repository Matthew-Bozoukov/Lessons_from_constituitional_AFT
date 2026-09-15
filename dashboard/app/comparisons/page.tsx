// ABOUTME: Live HF comparison surface linking dataset properties to trained-model outcomes.
// ABOUTME: Research facts remain in versioned public artifacts, never baked into the app.
import type { Metadata } from "next";
import { ComparisonExplorer } from "../components/ComparisonExplorer";
export const metadata: Metadata = { title: "Dataset → model comparisons" };
export default function ComparisonsPage() {
  return <main className="page-container inner-page comparison-page">
    <header className="page-heading"><div><span className="eyebrow">Training data → model behavior</span><h1>Dataset comparisons</h1>
      <p>Compare dataset properties with the outcomes of models trained on them. Published evidence is discovered live from Hugging Face.</p></div></header>
    <ComparisonExplorer />
  </main>;
}
