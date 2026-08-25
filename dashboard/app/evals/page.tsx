import type { Metadata } from "next";
import { EvalRunExplorer } from "../components/EvalRunExplorer";

export const metadata: Metadata = { title: "Evals" };

export default function EvalsPage() {
  return (
    <main className="page-container inner-page">
      <header className="page-heading">
        <div>
          <span className="eyebrow">Run-level evidence</span>
          <h1>Evaluation results</h1>
          <p>
            Every public eval-run repo in the org, discovered live from Hugging Face
            by its Hub tags. Pick one eval at a time, open a run&apos;s results and
            rollouts — or compare two runs side by side on the same metrics and the
            same scenarios. Runs from different instruments share no scale, so the
            explorer never mixes them.
          </p>
        </div>
      </header>

      <section className="data-section">
        <EvalRunExplorer />
      </section>
    </main>
  );
}
