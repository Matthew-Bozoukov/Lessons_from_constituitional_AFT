import Link from "next/link";
import {
  ArrowRight,
  Beaker,
  BrainCircuit,
  Database,
  FileText,
  FlaskConical,
  ScanSearch,
} from "lucide-react";
import { MetricTiles } from "./components/MetricTiles";
import { MockBadge, MockDataBanner } from "./components/MockDataBanner";
import { entries, entriesOfType, humanize } from "@/lib/content";

const surfaces = [
  {
    href: "/logs",
    title: "Experiment logs",
    description: "Training notes, runtime, cost, failures, and linked artifacts.",
    icon: FlaskConical,
    accent: "cyan",
  },
  {
    href: "/evals",
    title: "Evaluation results",
    description: "Compatible metrics, seeded runs, and checkpoint comparisons.",
    icon: Beaker,
    accent: "violet",
  },
  {
    href: "/datasets",
    title: "Synthetic datasets",
    description: "Inspect JSONL training records as readable conversations.",
    icon: Database,
    accent: "cyan",
  },
  {
    href: "/petri",
    title: "Petri audits",
    description: "Scenarios, transcripts, judge scores, and qualitative findings.",
    icon: ScanSearch,
    accent: "amber",
  },
  {
    href: "/models",
    title: "Model lineages",
    description: "Generated dossiers across base, SFT, DPO, and future stages.",
    icon: BrainCircuit,
    accent: "lime",
  },
  {
    href: "/findings",
    title: "Findings",
    description: "Claims with supporting evidence, caveats, and counterevidence.",
    icon: FileText,
    accent: "lime",
  },
];

export default function Home() {
  const evals = entriesOfType("evals");
  const logs = entriesOfType("logs");
  const findings = entriesOfType("findings");
  const datasets = entriesOfType("datasets");
  const petriRuns = entriesOfType("petri-runs");
  const latestFinding = findings[0];
  const latestCompleteEval =
    evals.find((entry) => entry.training_stage === "sft" && entry.status === "complete") ||
    evals.find((entry) => entry.status === "complete");
  // Stubs are generated placeholders (scripts/hf-discover.mjs) that record an artifact exists
  // and link it, with a machine-derived title and no result. They belong in the corpus, but a
  // reader landing here must see written research first — otherwise the front door is a wall of
  // "Lmsys qwen3 6 27b lora t2 9000 synthdoc 1000 r64".
  const isStub = (entry: { status?: string }) => entry?.status === "stub";
  const recent = [...entries.filter((e) => !isStub(e)), ...entries.filter(isStub)].slice(0, 6);
  // The featured finding and the headline eval tiles are the most dangerous
  // place a fixture can hide: a reader sees a number before any context.
  const featuredMock = [latestFinding, latestCompleteEval, ...recent].some(
    (entry) => entry?.mock === true,
  );
  const mockCount = entries.filter((entry) => entry.mock === true).length;

  return (
    <main>
      {featuredMock && (
        <div className="page-container mock-banner-wrap">
          <MockDataBanner scope="some" />
        </div>
      )}
      <section className="compact-hero">
        <div className="compact-hero-grid" aria-hidden="true" />
        <div className="page-container compact-hero-inner">
          <div>
            <div className="hero-eyebrow">
              <span className="pulse-dot" />
              Local research corpus
            </div>
            <h1>Synthetic Finetuning <span>for Constitution</span></h1>
            <p>
              <strong>The question:</strong> if you fine-tune a model on synthetic
              &ldquo;difficult advice&rdquo; conversations — where the assistant reasons about
              its values and declines to help with a norm violation — does it behave better in
              situations it was never trained on, like being given the chance to blackmail or
              leak data?
            </p>
            <p>
              This log holds the evidence: the training datasets, the fine-tuned checkpoints,
              the evaluations run against them, and the findings that survived scrutiny. New to
              the project? <Link href="/glossary">Start with the glossary</Link>.
            </p>
            <div className="hero-actions">
              <Link href="/evals" className="button primary">
                View results <ArrowRight size={16} />
              </Link>
              <Link href="/datasets" className="button secondary">
                Inspect datasets
              </Link>
            </div>
          </div>
          <div className="corpus-glance">
            <span className="eyebrow">Corpus at a glance</span>
            <div>
              <Link href="/evals"><strong>{evals.length}</strong><span>Evals</span></Link>
              <Link href="/logs"><strong>{logs.length}</strong><span>Logs</span></Link>
              <Link href="/datasets"><strong>{datasets.length}</strong><span>Datasets</span></Link>
              <Link href="/petri"><strong>{petriRuns.length}</strong><span>Petri runs</span></Link>
            </div>
            {/* State the real/fixture split in the counts themselves, rather
                than a blanket reassurance that fixtures are "clearly marked" -
                the counts above include them either way. */}
            <small>
              {mockCount === 0
                ? "Every record is a real research result."
                : `${entries.length - mockCount} of ${entries.length} records are real research results; ${mockCount} are interface fixtures, marked MOCK.`}
            </small>
          </div>
        </div>
      </section>

      <div className="page-container dashboard-home">
        <section className="current-state">
          <div className="state-heading">
            <span className="eyebrow">Current state</span>
            <h2>What the corpus says now</h2>
          </div>
          {latestFinding && (
            <Link className="latest-finding-card" href={`/entry/${latestFinding.slug}`}>
              <div>
                <span className={`status status-${latestFinding.status}`}>
                  {humanize(latestFinding.status)}
                </span>
                <small>Latest finding · {latestFinding.date}</small>
                {latestFinding.mock && <MockBadge />}
              </div>
              <h3>{latestFinding.title}</h3>
              <p>{latestFinding.summary}</p>
              <span className="text-link">Read evidence and caveats <ArrowRight size={14} /></span>
            </Link>
          )}
          {latestCompleteEval && (
            <div className="latest-metrics-card">
              <div className="latest-metrics-heading">
                <div>
                  <span className="eyebrow">Latest compatible signal</span>
                  <h3>{latestCompleteEval.checkpoint_id}</h3>
                  {latestCompleteEval.mock && <MockBadge />}
                </div>
                <Link href={`/entry/${latestCompleteEval.slug}`}>Open run <ArrowRight size={14} /></Link>
              </div>
              <MetricTiles metrics={latestCompleteEval.metrics} limit={3} />
              {/* Only the fields this run actually declares. Real runs mostly
                  do not carry a suite/version/seed triple, and rendering the
                  labels regardless produced a bare "seed" with nothing after
                  it, which reads as missing data rather than as inapplicable. */}
              <div className="compatibility-line">
                {latestCompleteEval.eval_suite && (
                  <code>{latestCompleteEval.eval_suite}</code>
                )}
                {latestCompleteEval.eval_version && (
                  <span>{latestCompleteEval.eval_version}</span>
                )}
                {latestCompleteEval.dataset_version && (
                  <span>{latestCompleteEval.dataset_version}</span>
                )}
                {latestCompleteEval.seed !== undefined && (
                  <span>seed {latestCompleteEval.seed}</span>
                )}
                {latestCompleteEval.git_commit && (
                  <span>@{String(latestCompleteEval.git_commit).slice(0, 7)}</span>
                )}
              </div>
            </div>
          )}
        </section>

        <section className="surface-section">
          <div className="section-heading row">
            <div><span className="eyebrow">Research surfaces</span><h2>Go straight to the evidence</h2></div>
          </div>
          <div className="surface-grid">
            {surfaces.map(({ href, title, description, icon: Icon, accent }) => (
              <Link className={`surface-card ${accent}`} href={href} key={href}>
                <span className="surface-icon"><Icon size={18} /></span>
                <div><h3>{title}</h3><p>{description}</p></div>
                <ArrowRight size={16} />
              </Link>
            ))}
          </div>
        </section>

        <section className="activity-section">
          <div className="section-heading row">
            <div><span className="eyebrow">Recent activity</span><h2>Latest records</h2></div>
            <span className="table-note">{entries.length} indexed records</span>
          </div>
          <div className="activity-list">
            {recent.map((entry) => (
              <Link href={`/entry/${entry.slug}`} key={entry.id}>
                <time dateTime={entry.date}>{entry.date}</time>
                <span className={`type-chip ${entry.type}`}>{humanize(entry.type)}</span>
                <strong>{entry.title}</strong>
                <span className={`status status-${entry.status}`}>{humanize(entry.status)}</span>
                <ArrowRight size={15} />
              </Link>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

