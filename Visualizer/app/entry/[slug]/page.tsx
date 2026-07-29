import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, Download, FileText } from "lucide-react";
import { MarkdownRenderer } from "@/app/components/MarkdownRenderer";
import { MetricTiles } from "@/app/components/MetricTiles";
import {
  entries,
  entryBySlug,
  formatBytes,
  humanize,
} from "@/lib/content";

export function generateStaticParams() {
  return entries.map((entry) => ({ slug: entry.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const entry = entryBySlug(slug);
  return entry
    ? { title: entry.title, description: entry.summary }
    : { title: "Entry not found" };
}

export default async function EntryPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const entry = entryBySlug(slug);
  if (!entry) notFound();
  const collectionHref =
    entry.type === "petri-runs" ? "/petri" : `/${entry.type}`;

  const metadata = [
    ["Model", entry.model_id],
    ["Checkpoint", entry.checkpoint_id],
    ["Parent", entry.parent_checkpoint_id],
    ["Training stage", entry.training_stage],
    ["Method", entry.training_method],
    ["Run", entry.run_id],
    ["Seed", entry.seed],
    ["Eval suite", entry.eval_suite],
    ["Eval version", entry.eval_version],
    ["Dataset", entry.dataset_version],
    ["Git commit", entry.git_commit],
  ].filter(([, value]) => value !== undefined && value !== "");

  return (
    <main className="page-container document-page">
      <Link href={collectionHref} className="back-link">
        <ArrowLeft size={15} /> Back to {humanize(entry.type)}
      </Link>

      <header className="document-header">
        <div className="document-title">
          <div className="document-kicker">
            <span className={`type-chip ${entry.type}`}>{humanize(entry.type)}</span>
            <span className={`status status-${entry.status}`}>{humanize(entry.status)}</span>
            <time dateTime={entry.date}>{entry.date}</time>
          </div>
          <h1>{entry.title}</h1>
          {entry.summary && <p>{entry.summary}</p>}
          <div className="tag-row">
            {entry.tags.map((tag) => <span key={tag}>#{tag}</span>)}
          </div>
        </div>
      </header>

      {Object.keys(entry.metrics).length > 0 && (
        <section className="document-metrics">
          <span className="eyebrow">Structured metrics</span>
          <MetricTiles metrics={entry.metrics} />
        </section>
      )}

      <div className="document-layout">
        <article>
          <MarkdownRenderer markdown={entry.body} />
        </article>
        <aside className="metadata-sidebar">
          <div className="sidebar-panel">
            <div className="sidebar-heading"><FileText size={15} /> Provenance</div>
            <dl>
              {metadata.map(([label, value]) => (
                <div key={String(label)}>
                  <dt>{label}</dt>
                  <dd><code>{String(value)}</code></dd>
                </div>
              ))}
              <div>
                <dt>Source</dt>
                <dd><code>{entry.source_path}</code></dd>
              </div>
            </dl>
          </div>

          {entry.assets.length > 0 && (
            <div className="sidebar-panel">
              <div className="sidebar-heading"><Download size={15} /> Artifacts</div>
              <div className="artifact-list">
                {entry.assets.map((asset) => (
                  <a href={asset.path} download key={asset.path}>
                    <span><strong>{asset.name}</strong><small>{asset.kind} · {formatBytes(asset.size_bytes)}</small></span>
                    <Download size={14} />
                  </a>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
