import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Glossary · SFC Research Log",
  description:
    "Plain-language definitions for the terms used across this research log.",
};

/**
 * Every term a reader hits on the front page or a metric tile without being told what it means.
 * The log was previously unreadable to anyone outside the project: the hero said "not
 * attributable to MSM or AFT" and a headline tile read "MSM ATTRIBUTABLE EFFECTS: 0 contrasts",
 * with nothing anywhere defining MSM, AFT, or a contrast.
 */
const SECTIONS: {
  heading: string;
  blurb: string;
  terms: { term: string; short: string; body: string }[];
}[] = [
  {
    heading: "The research question",
    blurb:
      "What this project is trying to find out, and the vocabulary the question is posed in.",
    terms: [
      {
        term: "Difficult advice",
        short: "The training data at the centre of the project",
        body:
          "Synthetic conversations where a user faces an ethically ambiguous situation and the assistant reasons openly about its values, then declines to help with a norm violation. The hypothesis is that training on these transfers to unrelated situations.",
      },
      {
        term: "Agentic misalignment",
        short: "The behaviour being measured",
        body:
          "A model taking a harmful autonomous action when given the opportunity — for example blackmailing an employee or leaking confidential data to avoid being shut down. Measured with honeypot scenarios the model was never trained on.",
      },
      {
        term: "Out-of-distribution (OOD)",
        short: "Different from anything in training",
        body:
          "A situation unlike the training data. The whole point of the experiment: training on advice conversations and then measuring behaviour in agentic scenarios that share no surface features with them.",
      },
      {
        term: "Constitution",
        short: "The written spec the training data is grounded in",
        body:
          "A document stating the values a model should hold. Synthetic training data is generated from it, so the constitution is the intended difference between two otherwise-matched datasets.",
      },
    ],
  },
  {
    heading: "Training",
    blurb: "How the checkpoints in this log were produced.",
    terms: [
      {
        term: "SFT",
        short: "Supervised fine-tuning",
        body:
          "Training a model on example conversations by imitation — the standard way the checkpoints here are produced. Loss is computed only on the assistant's tokens, never the prompt.",
      },
      {
        term: "LoRA / adapter",
        short: "A small trainable patch on a frozen model",
        body:
          "Low-Rank Adaptation. Instead of updating all 27 billion weights, a small set of extra weights is trained and applied on top. Cheap to train, cheap to store, and swappable at serving time.",
      },
      {
        term: "Checkpoint",
        short: "One specific saved model state",
        body:
          "A model at a specific point in training, identified by a hash. Results are always tied to a checkpoint, because two runs of the same recipe are not the same model.",
      },
      {
        term: "Mixture / ratio (e.g. 20/80)",
        short: "How much of each data source went in",
        body:
          "Training data is blended: e.g. 20% difficult-advice documents and 80% general instruction data, so any behaviour change is attributable to the share of the intervention data.",
      },
      {
        term: "Thinking mode",
        short: "Whether the model reasons before answering",
        body:
          "Qwen3.6 can emit a hidden reasoning trace before its answer. Whether a checkpoint was trained with real traces is recorded on the adapter and pinned at serving time — mixing the two invalidates a comparison.",
      },
    ],
  },
  {
    heading: "Evaluation",
    blurb: "How claims in this log are measured, and how strong the evidence is.",
    terms: [
      {
        term: "pass@1",
        short: "Solved on the first and only try",
        body:
          "The share of benchmark problems solved with a single attempt, no retries or reranking. The denominator is every selected instance — including ones the model failed to attempt, which is what stops the number being flattered.",
      },
      {
        term: "SWE-bench Verified",
        short: "A real-world coding benchmark",
        body:
          "500 real GitHub issues. The model must produce a patch that makes the repository's actual test suite pass. Graded by running the tests, not by a judge model.",
      },
      {
        term: "Paired comparison",
        short: "Same problems, two models",
        body:
          "Running two checkpoints over the identical instance set so the difference between them is not confounded by which problems each happened to get.",
      },
      {
        term: "McNemar test",
        short: "Is the difference real or noise?",
        body:
          "The statistical test for paired yes/no outcomes. It looks only at instances where the two models disagree. A p-value above 0.05 means the observed gap is consistent with chance.",
      },
      {
        term: "Confidence interval (CI)",
        short: "The range the true value plausibly lies in",
        body:
          "A 95% CI of 40–52% means the underlying rate is plausibly anywhere in that band. When two models' intervals overlap heavily, their scores are not meaningfully different.",
      },
      {
        term: "Negative result",
        short: "We looked and found nothing",
        body:
          "A finding that an expected effect did not appear. These are recorded here as first-class results — an intervention that does not work is worth exactly as much as one that does, and much cheaper to learn early.",
      },
      {
        term: "Contrast",
        short: "One specific before/after comparison",
        body:
          "A single matched pair being tested, e.g. one checkpoint against its control on one metric. \"0 of 15 contrasts survive correction\" means no comparison stayed significant once multiple testing was accounted for.",
      },
    ],
  },
  {
    heading: "Auditing",
    blurb: "Tools that probe for behaviour rather than scoring a benchmark.",
    terms: [
      {
        term: "Petri audit",
        short: "Automated red-teaming",
        body:
          "An auditor model puts the target model through many adversarial scenarios; a judge model scores the transcripts for concerning behaviour. Produces flagged transcripts, not an accuracy number.",
      },
      {
        term: "Flag",
        short: "One transcript marked as concerning",
        body:
          "A single scenario the judge marked for review. Flags are candidate findings, not confirmed ones — they must survive validation, and most do not.",
      },
      {
        term: "False positive rate",
        short: "How many flags were not real",
        body:
          "The share of flagged transcripts that did not hold up on review. A 57% rate means more than half the automated flags were not genuine issues, which is why flags are never reported as findings directly.",
      },
      {
        term: "MSM / AFT",
        short: "Training stages from the earlier vulnerabilities workstream",
        body:
          "Model-Spec Midtraining and Alignment Fine-Tuning — two stages in a separate investigation into whether training on a model spec introduces new vulnerabilities. Terms appear on older entries dated July 2026.",
      },
    ],
  },
  {
    heading: "Reading this log",
    blurb: "What the labels on an entry mean.",
    terms: [
      {
        term: "MOCK",
        short: "Fabricated interface fixture — not a measurement",
        body:
          "A placeholder used to exercise the interface. It is never a real result and supports no claim. Always shown with a warning banner.",
      },
      {
        term: "STUB",
        short: "The artifact exists; the write-up does not",
        body:
          "Generated automatically from a published Hugging Face dataset so the artifact is discoverable and linked. It carries no result or interpretation. A human replaces the body and drops the label.",
      },
      {
        term: "Provenance",
        short: "Where a number came from",
        body:
          "The commit, checkpoint, dataset revision and config that produced a result — recorded so it can be reproduced or challenged. A number without provenance is not a result.",
      },
    ],
  },
];

export default function GlossaryPage() {
  return (
    <main>
      <section className="compact-hero">
        <div className="compact-hero-grid" aria-hidden="true" />
        <div className="page-container compact-hero-inner">
          <div>
            <div className="hero-eyebrow">
              <span className="pulse-dot" />
              Start here
            </div>
            <h1>Glossary</h1>
            <p>
              Plain-language definitions for every term this log uses without
              explaining. If an entry or a metric tile is opaque, the word is
              probably defined here.
            </p>
            <div className="hero-actions">
              <Link href="/" className="button secondary">
                Back to overview
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="page-container" style={{ paddingBottom: "4rem" }}>
        {SECTIONS.map((section) => (
          <div key={section.heading} style={{ marginTop: "2.5rem" }}>
            <span className="eyebrow">{section.heading}</span>
            <p style={{ color: "var(--text-secondary, #6b7280)", marginTop: ".25rem" }}>
              {section.blurb}
            </p>
            <dl
              style={{
                display: "grid",
                gap: "1rem",
                gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
                marginTop: "1rem",
              }}
            >
              {section.terms.map((t) => (
                <div
                  key={t.term}
                  style={{
                    border: "1px solid var(--border, #e5e7eb)",
                    borderRadius: "12px",
                    padding: "1rem 1.1rem",
                  }}
                >
                  <dt style={{ fontWeight: 650, fontSize: "1.02rem" }}>{t.term}</dt>
                  <div
                    style={{
                      fontSize: ".84rem",
                      textTransform: "uppercase",
                      letterSpacing: ".04em",
                      color: "var(--text-muted, #9ca3af)",
                      margin: ".2rem 0 .5rem",
                    }}
                  >
                    {t.short}
                  </div>
                  <dd style={{ margin: 0, lineHeight: 1.55 }}>{t.body}</dd>
                </div>
              ))}
            </dl>
          </div>
        ))}
      </section>
    </main>
  );
}
