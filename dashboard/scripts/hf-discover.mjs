// Reconcile the Hugging Face org against `content/`, so drift is visible instead of silent.
//
// The research log is hand-curated: an entry exists only because someone wrote
// `content/<type>/<slug>/index.md`. That is good for narrative quality and terrible for
// coverage — every HF publish needs a matching file, and when nobody writes one the dashboard
// silently under-reports the project. On 2026-08-10 it showed 1 of 79 datasets and 1 of 38
// model families, all of it 12 days stale.
//
// This script does not replace curation. It makes the gap measurable:
//
//   node scripts/hf-discover.mjs              # report drift, exit 1 if anything is unlisted
//   node scripts/hf-discover.mjs --generate   # additionally write stub entries for the gap
//   node scripts/hf-discover.mjs --json       # machine-readable, for CI
//
// Stubs are deliberately marked `status: stub` and carry a visible banner in the body, so a
// generated placeholder can never be mistaken for a written-up result. A human replaces the
// body; the frontmatter is already correct because it comes from the dataset card the
// publishing code was required to write.

import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

const ORG = process.env.HF_ORG || "LASR-Callum";
const ENDPOINT = process.env.HF_ENDPOINT?.replace(/\/+$/, "") || "https://huggingface.co";
const ROOT = path.resolve(import.meta.dirname, "..");
const CONTENT = path.join(ROOT, "content");

function token() {
  return (
    process.env.HF_TOKEN ||
    process.env.HUGGING_FACE_HUB_TOKEN ||
    process.env.HUGGINGFACEHUB_API_TOKEN ||
    ""
  );
}

async function hf(pathname) {
  const t = token();
  const res = await fetch(`${ENDPOINT}${pathname}`, {
    headers: t ? { authorization: `Bearer ${t}` } : {},
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) throw new Error(`${pathname} -> HTTP ${res.status}`);
  return res.json();
}

/** Which surface an HF repo belongs on, from its name. Eval names win over "train". */
function classify(id) {
  const n = id.split("/").pop().toLowerCase();
  if (/petri/.test(n)) return "petri-runs";
  if (/swebench|mmlu|lmsys|psychosis|gpqa|odcv|arena|agentic|misalignment|capability|internaliz|eval|probe|audit|surf/.test(n))
    return "evals";
  if (/train|sft|corpus|synthdoc|mixture|dataset|selfreflect|self-reflection|memself|model-eval-model/.test(n))
    return "datasets";
  return "datasets";
}

/** Existing entries, and every HF repo id they already reference. */
async function readContent() {
  const entries = [];
  const referenced = new Set();
  for (const type of ["logs", "evals", "findings", "datasets", "petri-runs"]) {
    const dir = path.join(CONTENT, type);
    let slugs = [];
    try {
      slugs = await fs.readdir(dir);
    } catch {
      continue;
    }
    for (const slug of slugs) {
      const file = path.join(dir, slug, "index.md");
      let text;
      try {
        text = await fs.readFile(file, "utf8");
      } catch {
        continue;
      }
      const repo = /repo_id:\s*([^\s#]+)/.exec(text)?.[1];
      if (repo) referenced.add(repo.trim());
      entries.push({ type, slug, file, repo: repo?.trim() ?? null });
    }
  }
  return { entries, referenced };
}

/**
 * How a published card declares itself a fabricated interface fixture.
 *
 * The fixture repos say it in their `experiment` field, in caps, as the first
 * thing in the card: "MOCK DATA - NOT A TRAINING CORPUS". That is the marker
 * that exists on the Hub, so it is the one read here.
 *
 * This matters because of how the flag was lost. `2026-07-30-visualizer-mock-dialogues`
 * is the dataset browser's fixture, and its entry was regenerated as a stub by
 * this script — which carried the card's summary across but not its status, so
 * eleven hand-written dialogues rendered on /datasets as a real corpus with no
 * badge and no banner. A generator that can create an entry for a fixture has
 * to be able to mark one.
 */
const MOCK_CARD = /\bMOCK DATA\b|\bNOT A (TRAINING CORPUS|RESEARCH RESULT)\b/;

/**
 * First real prose paragraph of a dataset card — our cards lead with a table,
 * so skip it — plus whether the card declares itself a fixture.
 */
async function summaryFor(id) {
  try {
    const res = await fetch(`${ENDPOINT}/datasets/${id}/resolve/main/README.md`, {
      headers: token() ? { authorization: `Bearer ${token()}` } : {},
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) return { summary: null, mock: false };
    const card = await res.text();
    const body = card.replace(/^---[\s\S]*?---\n/, "");
    const mock = MOCK_CARD.test(card);
    for (const para of body.split(/\n\s*\n/)) {
      const p = para.trim();
      if (!p || p.startsWith("|") || p.startsWith("```")) continue;
      if (p.startsWith("#")) continue;
      return { summary: p.replace(/\s+/g, " ").slice(0, 400), mock };
    }
    return { summary: null, mock };
  } catch {
    /* a missing card is not an error; the stub just has no summary */
  }
  return { summary: null, mock: false };
}

function slugFor(id) {
  return id.split("/").pop().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function dateFor(id, lastModified) {
  const m = /(\d{4}-\d{2}-\d{2})/.exec(id.split("/").pop());
  return m ? m[1] : (lastModified || "").slice(0, 10);
}

function yamlEscape(s) {
  return `'${String(s).replace(/'/g, "''")}'`;
}

async function writeStub(repo, type, summary, mock) {
  const slug = slugFor(repo.id);
  const dir = path.join(CONTENT, type, slug);
  const file = path.join(dir, "index.md");
  try {
    await fs.access(file);
    return { slug, written: false };
  } catch {
    /* does not exist — write it */
  }
  const date = dateFor(repo.id, repo.lastModified);
  const title = slug.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/-/g, " ");
  const fm = [
    "---",
    `title: ${yamlEscape(title.charAt(0).toUpperCase() + title.slice(1))}`,
    `date: ${yamlEscape(date)}`,
    summary ? `summary: ${yamlEscape(summary)}` : `summary: ${yamlEscape("Generated stub — no dataset card summary found. Replace with a written summary.")}`,
    "status: stub",
    // Carried from the card, never inferred from the slug: a fixture that
    // renders as a research result is the worst failure this site can have.
    ...(mock ? ["mock: true"] : []),
    "hf_source:",
    `  repo_id: ${repo.id}`,
    `  revision: ${repo.sha || "main"}`,
    "tags:",
    "  - generated-stub",
    "---",
    "",
    "> **Generated stub.** This entry was created by `scripts/hf-discover.mjs --generate` from",
    "> the Hugging Face dataset card. It records that the artifact exists and links it; it is",
    `> **not** a write-up and supports no claim. Replace this body with the result, then drop`,
    "> `status: stub` and the `generated-stub` tag.",
    "",
    `Source: [\`${repo.id}\`](${ENDPOINT}/datasets/${repo.id})`,
    "",
  ].join("\n");
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(file, fm, "utf8");
  return { slug, written: true };
}

async function main() {
  const generate = process.argv.includes("--generate");
  const asJson = process.argv.includes("--json");

  const [datasets, models] = await Promise.all([
    hf(`/api/datasets?author=${ORG}&limit=1000`),
    hf(`/api/models?author=${ORG}&limit=1000`),
  ]);
  const { entries, referenced } = await readContent();

  const unlisted = datasets.filter((d) => !referenced.has(d.id));
  const dangling = [...referenced].filter(
    (r) => !datasets.some((d) => d.id === r) && !models.some((m) => m.id === r),
  );

  const report = {
    org: ORG,
    hf_datasets: datasets.length,
    hf_models: models.length,
    content_entries: entries.length,
    entries_with_hf_source: referenced.size,
    unlisted_datasets: unlisted.length,
    dangling_references: dangling,
    coverage_pct: datasets.length ? Math.round((referenced.size / datasets.length) * 100) : 0,
  };

  if (asJson) {
    console.log(JSON.stringify({ ...report, unlisted: unlisted.map((d) => d.id) }, null, 2));
  } else {
    console.log(`Hugging Face org: ${ORG}`);
    console.log(`  datasets on hub      : ${report.hf_datasets}`);
    console.log(`  models on hub        : ${report.hf_models}`);
    console.log(`  content entries      : ${report.content_entries}`);
    console.log(`  entries linked to hub: ${report.entries_with_hf_source}`);
    console.log(`  COVERAGE             : ${report.coverage_pct}% of datasets are represented`);
    if (dangling.length) {
      console.log(`\n  DANGLING (entry points at a repo that is not on the hub):`);
      for (const d of dangling) console.log(`    ${d}`);
    }
    if (unlisted.length) {
      console.log(`\n  UNLISTED (on the hub, invisible in the dashboard): ${unlisted.length}`);
      const byType = {};
      for (const d of unlisted) (byType[classify(d.id)] ??= []).push(d.id);
      for (const [type, ids] of Object.entries(byType)) {
        console.log(`    ${type} (${ids.length}):`);
        for (const id of ids.slice(0, 12)) console.log(`      ${id}`);
        if (ids.length > 12) console.log(`      ... and ${ids.length - 12} more`);
      }
    }
  }

  if (generate) {
    console.log(`\n  generating stubs for ${unlisted.length} unlisted datasets ...`);
    let n = 0;
    for (const d of unlisted) {
      const type = classify(d.id);
      const { summary, mock } = await summaryFor(d.id);
      const { slug, written } = await writeStub(d, type, summary, mock);
      if (written) {
        n += 1;
        console.log(`    + ${type}/${slug}`);
      }
    }
    console.log(`  wrote ${n} stub entries. Review them, replace the bodies, drop 'status: stub'.`);
  }

  process.exitCode = unlisted.length || dangling.length ? 1 : 0;
}

main().catch((e) => {
  console.error(`hf-discover failed: ${e.message}`);
  process.exitCode = 2;
});
