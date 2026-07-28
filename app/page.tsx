import Link from "next/link";
import {
  ArrowRight,
  Beaker,
  BrainCircuit,
  CircleDollarSign,
  FlaskConical,
  Gauge,
  ShieldCheck,
} from "lucide-react";
import { EntryCard } from "./components/EntryCard";
import { MetricExplorer } from "./components/MetricExplorer";
import { entries, entriesOfType, modelsInCorpus } from "@/lib/content";

export default function Home() {
  const evals = entriesOfType("evals");
  const logs = entriesOfType("logs");
  const findings = entriesOfType("findings");
  const recent = entries.slice(0, 4);
  const totalCost = evals.reduce(
    (sum, entry) => sum + (entry.metrics.cost_usd?.value || 0),
    0,
  );
  const completed = entries.filter((entry) => entry.status === "complete").length;

  return (
    <main>
      <section className="hero">
        <div className="hero-grid" aria-hidden="true" />
        <div className="page-container hero-inner">
          <div className="hero-copy">
            <div className="hero-eyebrow">
              <span className="pulse-dot" />
              Local research corpus · demonstration data
            </div>
            <h1>
              Synthetic Finetuning
              <span>for Constitution</span>
            </h1>
            <p>
              A living record of training interventions, behavioral evaluations,
              generalization checks, and the evidence between them.
            </p>
            <div className="hero-actions">
              <Link href="/evals" className="button primary">
                Explore evals <ArrowRight size={16} />
              </Link>
              <Link href="/logs" className="button secondary">
                Open experiment logs
              </Link>
            </div>
          </div>

          <div className="hero-signal">
            <div className="signal-header">
              <span>Program surface</span>
              <code>demo/v1</code>
            </div>
            <div className="signal-lineage">
              {["base", "sft", "bounded-dpo"].map((stage, index) => (
                <div className="lineage-stage" key={stage}>
                  <span className={`stage-node stage-${index}`} />
                  <div>
                    <small>Stage {String(index + 1).padStart(2, "0")}</small>
                    <strong>{stage}</strong>
                  </div>
                </div>
              ))}
            </div>
            <div className="signal-footer">
              <span><ShieldCheck size={15} /> OOD checks</span>
              <span><Gauge size={15} /> Capability guardrails</span>
            </div>
          </div>
        </div>
      </section>

      <div className="page-container overview-content">
        <section className="stat-grid" aria-label="Corpus summary">
          <div className="stat-card cyan">
            <Beaker size={20} />
            <div><strong>{evals.length}</strong><span>Eval results</span></div>
          </div>
          <div className="stat-card violet">
            <FlaskConical size={20} />
            <div><strong>{logs.length}</strong><span>Experiment logs</span></div>
          </div>
          <div className="stat-card lime">
            <BrainCircuit size={20} />
            <div><strong>{modelsInCorpus().length}</strong><span>Model families</span></div>
          </div>
          <div className="stat-card amber">
            <CircleDollarSign size={20} />
            <div><strong>${totalCost.toFixed(0)}</strong><span>Recorded eval cost</span></div>
          </div>
          <div className="stat-card neutral">
            <ShieldCheck size={20} />
            <div><strong>{completed}/{entries.length}</strong><span>Complete records</span></div>
          </div>
        </section>

        <section className="program-grid">
          <div className="section-heading">
            <span className="eyebrow">Research program</span>
            <h2>Evidence, not just outcomes</h2>
            <p>
              Each result stays connected to its intervention, data recipe,
              compatible eval group, and known vulnerabilities.
            </p>
          </div>
          <div className="thread-grid">
            {[
              ["01", "Replications", "Reproduce published SDF and model-spec interventions across open checkpoints."],
              ["02", "Extensions", "Test reasons-rich LoRA SFT and bounded preference optimization branches."],
              ["03", "Generalization", "Separate knowledge recall from behavior under held-out distribution shifts."],
              ["04", "Vulnerability checks", "Track synthetic artifacts, adaptive pressure, eval awareness, and regressions."],
            ].map(([number, title, description]) => (
              <div className="thread-card" key={number}>
                <code>{number}</code>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="explorer-section">
          <MetricExplorer entries={evals} />
        </section>

        <section className="recent-section">
          <div className="section-heading row">
            <div>
              <span className="eyebrow">Latest from the corpus</span>
              <h2>Recent research records</h2>
            </div>
            <Link href="/findings" className="text-link">
              View findings <ArrowRight size={15} />
            </Link>
          </div>
          <div className="entry-grid">
            {recent.map((entry) => <EntryCard entry={entry} key={entry.id} />)}
          </div>
        </section>

        {findings[0] && (
          <section className="finding-callout">
            <span className="eyebrow">Current provisional finding</span>
            <h2>{findings[0].title}</h2>
            <p>{findings[0].summary}</p>
            <Link href={`/entry/${findings[0].slug}`}>
              Read evidence and counterevidence <ArrowRight size={16} />
            </Link>
          </section>
        )}
      </div>
    </main>
  );
}

