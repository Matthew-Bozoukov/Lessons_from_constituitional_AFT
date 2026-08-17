import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SelectionRankingViewer, type SelectionSource } from "../components/SelectionRankingViewer";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { entriesOfType } from "@/lib/content";
import { entryBody } from "@/lib/body";

export const metadata: Metadata = { title: "Data selection" };

/**
 * A selection run is a dataset entry that declares a `selection` block: where its
 * ranking lives on the Hub and which files hold it. Keying off the declaration
 * rather than a hardcoded slug means a second selection run appears here by
 * publishing its content entry, with no code change.
 */
type SelectionDecl = {
  scores?: string;
  offsets?: string;
  targets?: string[];
  method?: string;
};

export default async function SelectionPage() {
  const runs = entriesOfType("datasets").filter((entry) => entry.selection);
  const run = runs.find((candidate) => candidate.mock !== true) || runs[0];
  if (!run) notFound();

  const decl = run.selection as SelectionDecl;
  const repo = (run.hf_source as { repo_id?: string } | undefined)?.repo_id;
  if (!repo) notFound();

  const source: SelectionSource = {
    base: `https://huggingface.co/datasets/${repo}/resolve/main`,
    scoresFile: decl.scores || "scores/scores.jsonl",
    offsetsFile: decl.offsets || "rankings/pool_offsets.json",
  };
  const body = await entryBody(run.slug);

  return (
    <main className="page-container inner-page selection-page">
      <header className="selection-head">
        <h1>{run.title}</h1>
        <p>{run.summary}</p>
        <p className="selection-meta">
          {decl.method ? <span>{decl.method}</span> : null}
          {decl.targets?.length ? <span>targets: {decl.targets.join(", ")}</span> : null}
          <a href={`https://huggingface.co/datasets/${repo}`} target="_blank" rel="noreferrer">
            {repo}
          </a>
        </p>
      </header>

      <SelectionRankingViewer source={source} />

      {body ? (
        <section className="selection-note">
          <MarkdownRenderer markdown={body} />
        </section>
      ) : null}
    </main>
  );
}
