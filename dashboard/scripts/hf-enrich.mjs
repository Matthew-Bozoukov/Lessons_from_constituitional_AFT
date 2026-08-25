// Pull REAL measured values out of published HF eval bundles into content entries.
//
// A stub records that an artifact exists. It displays nothing, because the dashboard renders
// metric tiles from the `metrics:` frontmatter block and a stub has none. Meanwhile the eval
// bundles on the Hub already contain exactly those numbers - `results.json` from run_eval holds
// the target, the mode, the sample count and the scores.
//
// This script copies those numbers into the entry. It invents nothing: every value written here
// was read from a published file, and an entry whose bundle has no readable results is left
// alone rather than filled with a guess.
//
//   node scripts/hf-enrich.mjs            # report what could be enriched
//   node scripts/hf-enrich.mjs --apply    # rewrite the entries
//
// Enriched entries are tagged `auto-indexed` and say so in the body: the NUMBERS are real and
// measured, but no human has written the interpretation, and a reader must not mistake a
// transcribed metric for an analysed result.

import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

const ENDPOINT = process.env.HF_ENDPOINT?.replace(/\/+$/, "") || "https://huggingface.co";
const ROOT = path.resolve(import.meta.dirname, "..");
const CONTENT = path.join(ROOT, "content");
// Contract layout (results/, metadata/) first; bare names are pre-2026-08-24 repos.
const CANDIDATES = [
  "results/results.json",
  "results/metrics.json",
  "results.json",
  "metrics.json",
  "summary.json",
  "metadata/run_meta.json",
  "run_meta.json",
];

function token() {
  return (
    process.env.HF_TOKEN ||
    process.env.HUGGING_FACE_HUB_TOKEN ||
    process.env.HUGGINGFACEHUB_API_TOKEN ||
    ""
  );
}

async function fetchJson(repo, file) {
  try {
    const res = await fetch(`${ENDPOINT}/datasets/${repo}/resolve/main/${file}`, {
      headers: token() ? { authorization: `Bearer ${token()}` } : {},
      signal: AbortSignal.timeout(25000),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Units are inferred from the name and range, never invented: a rate in [0,1] is a proportion. */
function unitFor(key, value) {
  const k = key.toLowerCase();
  if (/_ci_|ci_lower|ci_upper/.test(k)) return "bound";
  if (/rate|accuracy|mean_score|share|proportion|pct|percent/.test(k)) {
    return value >= 0 && value <= 1 ? "proportion" : "percent";
  }
  if (/^n$|^n_|count|instances|turns|samples|records/.test(k)) return "count";
  if (/mean|avg|score/.test(k)) return "score";
  if (/p_value|pvalue/.test(k)) return "p";
  return "value";
}

const LOWER_IS_BETTER =
  /delusion|collusion|refus|harm|violation|leak|blackmail|error|fail|p_value|overflow/i;

/** Flatten one level of nesting so `scores.mean` becomes `scores_mean`. */
function numericMetrics(doc) {
  const out = {};
  const take = (k, v) => {
    if (typeof v !== "number" || !Number.isFinite(v)) return;
    const key = k.replace(/[^a-z0-9]+/gi, "_").replace(/^_|_$/g, "").toLowerCase();
    if (!key || key in out) return;
    out[key] = { value: Math.round(v * 10000) / 10000, unit: unitFor(key, v) };
    if (LOWER_IS_BETTER.test(key)) out[key].lower_is_better = true;
  };
  for (const [k, v] of Object.entries(doc ?? {})) {
    if (typeof v === "number") take(k, v);
    else if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v)) {
        if (typeof v2 === "number") take(`${k}_${k2}`, v2);
      }
    }
  }
  return out;
}

function yamlStr(s) {
  return `'${String(s).replace(/'/g, "''")}'`;
}

/** Readable eval name from the repo slug. Unknown families keep their slug. */
const FAMILIES = [
  [/swebench[-_]?verified/i, "SWE-bench Verified"],
  [/swebench/i, "SWE-bench"],
  [/mmlu/i, "MMLU"],
  [/gpqa/i, "GPQA Diamond"],
  [/lmsys/i, "LMSYS chat win-rate"],
  [/arena[-_]?hard/i, "Arena-Hard"],
  [/psychosis/i, "Psychosis red-teaming"],
  [/odcv/i, "ODCV-Bench"],
  [/agentic[-_]?misalignment/i, "Agentic misalignment honeypots"],
  [/model[-_]?eval[-_]?model/i, "Model-evaluates-model"],
  [/petri/i, "Petri audit"],
  [/surf/i, "SURF audit"],
];

function prettyTitle(slug, target) {
  const bare = slug.replace(/^\d{4}-\d{2}-\d{2}-/, "");
  const fam = FAMILIES.find(([re]) => re.test(bare))?.[1];
  const model = target ? target.split("/").pop() : null;
  if (fam && model) return `${fam} — ${model}`;
  if (fam) return `${fam} — ${bare}`;
  return bare.replace(/-/g, " ").replace(/^./, (c) => c.toUpperCase());
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

/**
 * A factual one-liner built only from measured values. Deliberately descriptive, never
 * evaluative: it says what was measured, not whether it is good.
 */
function factualSummary(slug, metrics, target, mode) {
  const m = (k) => metrics[k]?.value;
  const model = target ? target.split("/").pop() : "the target model";
  const suffix = mode ? ` in ${mode} mode` : "";
  const bare = slug.replace(/^\d{4}-\d{2}-\d{2}-/, "");

  if (/mmlu|gpqa/i.test(bare) && m("mean") != null) {
    const n = m("n");
    const c = m("n_correct");
    const ci =
      m("ci_lower") != null ? ` (95% CI ${pct(m("ci_lower"))}–${pct(m("ci_upper"))})` : "";
    return `Accuracy ${pct(m("mean"))}${ci}${n ? ` over ${n} questions` : ""}${
      c != null ? `, ${c} correct` : ""
    }, for ${model}${suffix}.`;
  }
  if (/psychosis/i.test(bare) && m("referral_rate") != null) {
    return `Delusion red-teaming over ${m("n_graded_turns") ?? "n"} graded turns for ${model}${suffix}: referral rate ${pct(
      m("referral_rate"),
    )}, delusion-confirmation mean ${m("delusion_confirmation_mean")}, pushback mean ${m(
      "pushback_mean",
    )}.`;
  }
  const first = Object.entries(metrics).slice(0, 3);
  if (first.length) {
    const parts = first.map(([k, v]) => `${k.replace(/_/g, " ")} ${v.value}`);
    return `Measured values for ${model}${suffix}: ${parts.join(", ")}.`;
  }
  return `Published evaluation bundle for ${model}${suffix}.`;
}

function renderMetrics(metrics) {
  const lines = ["metrics:"];
  for (const [k, m] of Object.entries(metrics).slice(0, 8)) {
    lines.push(`  ${k}:`);
    lines.push(`    value: ${m.value}`);
    lines.push(`    unit: ${m.unit}`);
    if (m.lower_is_better) lines.push("    lower_is_better: true");
  }
  return lines.join("\n");
}

async function enrich(file, apply) {
  const text = await fs.readFile(file, "utf8");
  const repo = /repo_id:\s*([^\s#]+)/.exec(text)?.[1]?.trim();
  if (!repo) return { skipped: "no hf_source" };
  // Re-processable: a stub, or something this script itself produced. Never a human write-up.
  const isStub = /^status:\s*stub/m.test(text);
  const isAuto = /^\s+-\s+auto-indexed\s*$/m.test(text);
  if (!isStub && !isAuto) return { skipped: "already written by a human" };

  let doc = null;
  let src = null;
  for (const f of CANDIDATES) {
    doc = await fetchJson(repo, f);
    if (doc) {
      src = f;
      break;
    }
  }
  if (!doc) return { skipped: "no results file on the hub" };

  const metrics = numericMetrics(doc);
  if (!Object.keys(metrics).length) return { skipped: `${src} has no numeric fields` };

  const target = typeof doc.target === "string" ? doc.target : null;
  const mode = typeof doc.mode === "string" ? doc.mode : null;

  if (!apply) return { enriched: Object.keys(metrics).length, src, target };

  // Rewrite frontmatter: real metrics in, stub marker out, readable title/summary derived
  // from the measured values rather than from the repo slug.
  const slug = path.basename(path.dirname(file));
  const parts = text.split(/^---$/m);
  let fm = parts[1] ?? "";
  fm = fm.replace(/^status:\s*stub\s*$/m, "status: complete");
  fm = fm.replace(/^title:.*$/m, `title: ${yamlStr(prettyTitle(slug, target))}`);
  fm = fm.replace(
    /^summary:.*(?:\n(?:[ \t]+.*|>-?)$)*/m,
    `summary: ${yamlStr(factualSummary(slug, metrics, target, mode))}`,
  );
  fm = fm.replace(/^tags:\n(?:\s+-\s+.*\n)*/m, "tags:\n  - auto-indexed\n");
  if (!/^tags:/m.test(fm)) fm += "\ntags:\n  - auto-indexed\n";
  if (target && !/^models:/m.test(fm)) fm += `models:\n  - ${target}\n`;
  if (target && !/^target_model_id:/m.test(fm)) fm += `target_model_id: ${target}\n`;
  fm = fm.replace(/^metrics:\n(?:\s{2,}.*\n)*/m, "");
  fm += `${renderMetrics(metrics)}\n`;

  const rows = Object.entries(metrics)
    .slice(0, 8)
    .map(([k, m]) => `| \`${k}\` | ${m.value} | ${m.unit} |`)
    .join("\n");

  const body = [
    "",
    "> **Auto-indexed from the published bundle.** Every number below was read from",
    `> \`${src}\` in the linked Hugging Face dataset — none of it is estimated or filled in.`,
    "> No human has written the interpretation yet, so treat this as measured values, not as an",
    "> analysed result.",
    "",
    "## Measured values",
    "",
    "| metric | value | unit |",
    "| --- | --- | --- |",
    rows,
    "",
    target ? `**Target:** \`${target}\`${mode ? ` · mode \`${mode}\`` : ""}` : "",
    "",
    `Source: [\`${repo}\`](${ENDPOINT}/datasets/${repo})`,
    "",
  ].join("\n");

  await fs.writeFile(file, `---${fm}---\n${body}`, "utf8");
  return { enriched: Object.keys(metrics).length, src, target, written: true };
}

async function main() {
  const apply = process.argv.includes("--apply");
  const files = [];
  for (const type of ["evals", "datasets", "petri-runs", "logs", "findings"]) {
    let slugs = [];
    try {
      slugs = await fs.readdir(path.join(CONTENT, type));
    } catch {
      continue;
    }
    for (const s of slugs) files.push(path.join(CONTENT, type, s, "index.md"));
  }

  let ok = 0;
  const skips = {};
  for (const f of files) {
    let r;
    try {
      r = await enrich(f, apply);
    } catch (e) {
      r = { skipped: e.message };
    }
    if (r.enriched) {
      ok += 1;
      console.log(
        `  ${apply ? "+" : "·"} ${path.relative(CONTENT, f).replace(/\\/g, "/")}  ` +
          `${r.enriched} metrics from ${r.src}`,
      );
    } else if (r.skipped) {
      skips[r.skipped] = (skips[r.skipped] || 0) + 1;
    }
  }
  console.log(`\n  ${apply ? "enriched" : "enrichable"}: ${ok} entries`);
  for (const [k, v] of Object.entries(skips).sort((a, b) => b[1] - a[1])) {
    console.log(`  skipped (${v}): ${k}`);
  }
  if (!apply) console.log("\n  (dry run — pass --apply to write)");
}

main().catch((e) => {
  console.error(`hf-enrich failed: ${e.message}`);
  process.exitCode = 2;
});
