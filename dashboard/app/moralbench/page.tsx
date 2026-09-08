import type { Metadata } from "next";
import { MoralBenchExplorer } from "../components/MoralBenchExplorer";

export const metadata: Metadata = { title: "MoralBench" };

export default function MoralBenchPage() {
  return (
    <main className="page-container inner-page">
      <header className="page-heading">
        <div>
          <span className="eyebrow">Declarative values probe</span>
          <h1>Moral foundations profile</h1>
          <p>
            MoralBench asks a checkpoint what it thinks is relevant to right and wrong,
            with no scenario and no stakes, and returns a position in a six-foundation
            taxonomy that predates our constitution. Every other misalignment eval here
            is behavioural and returns a scalar; this one returns a shape, which is what
            makes it useful for saying <em>how</em> two checkpoints differ rather than
            just how much.
          </p>
          <p>
            Pick any runs published to Hugging Face and read them side by side, overall
            or broken out by foundation. The interesting quantity is the difference
            between arms on identical items — not any one arm&apos;s level, and not a
            ranking: a higher score means closer agreement with the human norming
            sample, which is a description, not a verdict.
          </p>
        </div>
      </header>

      <section className="data-section">
        <MoralBenchExplorer />
      </section>
    </main>
  );
}
