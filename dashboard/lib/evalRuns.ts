"use client";

// ABOUTME: Client-side access to the org's eval-run repos on Hugging Face: tag-filtered
// ABOUTME: discovery, contract-layout fetching (results/, rollouts/), per-eval adapters.
//
// Every repo this reads obeys the published-layout contract (src/eval/layout.py in the
// research repo): rollouts/ + results/results.json + metadata/. Discovery is HF's
// canonical route — card front-matter tags, Hub-indexed, filterable via
// /api/datasets?author=<org>&filter=eval-run. Only PUBLIC repos are visible here:
// the site is token-less by design (netlify.toml).

import type { DialogueMessage } from "./content";
import { cached, loadJsonDoc } from "./lazy.ts";

const ENDPOINT = "https://huggingface.co";
export const EVAL_ORG = "LASR-Callum";

export type Json = Record<string, unknown>;

export interface EvalRun {
  repo: string;
  evalName: string;
  model: string;
  mode: string;
  lastModified: string;
}

function tagValue(tags: string[], prefix: string): string {
  const hit = tags.find((t) => t.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : "";
}

export function listEvalRuns(org: string = EVAL_ORG): Promise<EvalRun[]> {
  const url = `${ENDPOINT}/api/datasets?author=${org}&filter=eval-run&limit=500&full=true`;
  return cached(url, async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} listing eval runs`);
    const docs = (await res.json()) as Array<{
      id: string;
      tags?: string[];
      lastModified?: string;
    }>;
    return docs
      .map((d) => ({
        repo: d.id,
        evalName: tagValue(d.tags || [], "eval:") || "unknown",
        model: tagValue(d.tags || [], "model:") || d.id.split("/")[1] || d.id,
        mode: tagValue(d.tags || [], "mode:"),
        lastModified: d.lastModified || "",
      }))
      .sort((a, b) => b.lastModified.localeCompare(a.lastModified));
  });
}

export function resolveUrl(repo: string, path: string): string {
  return `${ENDPOINT}/datasets/${repo}/resolve/main/${path}`;
}

export function repoUrl(repo: string): string {
  return `${ENDPOINT}/datasets/${repo}`;
}

export function repoDate(repo: string): string {
  const name = repo.split("/")[1] || "";
  return /^\d{4}-\d{2}-\d{2}/.test(name) ? name.slice(0, 10) : "";
}

export function loadResults(repo: string): Promise<Json> {
  // Contract layout first; pre-2026-08-24 repos kept results.json at the root.
  return cached(`results:${repo}`, async () => {
    try {
      return await loadJsonDoc<Json>(resolveUrl(repo, "results/results.json"));
    } catch {
      return await loadJsonDoc<Json>(resolveUrl(repo, "results.json"));
    }
  });
}

export interface TreeFile {
  path: string; // relative to rollouts/ in contract repos; repo-relative in legacy repos
  full: string; // always repo-relative — the fetchable path
  size: number;
}

async function treeList(repo: string, sub: string): Promise<Array<{ path: string; size: number }>> {
  let url: string | null =
    `${ENDPOINT}/api/datasets/${repo}/tree/main${sub ? `/${sub}` : ""}?recursive=1`;
  const out: Array<{ path: string; size: number }> = [];
  for (let page = 0; page < 8 && url; page += 1) {
    const res: Response = await fetch(url);
    if (res.status === 404) return [];
    if (!res.ok) throw new Error(`HTTP ${res.status} listing ${repo}`);
    const rows = (await res.json()) as Array<{ type: string; path: string; size?: number }>;
    for (const row of rows) {
      if (row.type === "file") out.push({ path: row.path, size: row.size || 0 });
    }
    const link = res.headers.get("Link") || "";
    const next = link.match(/<([^>]+)>;\s*rel="next"/);
    url = next && next[1].startsWith(ENDPOINT) ? next[1] : null;
  }
  return out;
}

export function listRolloutFiles(repo: string): Promise<TreeFile[]> {
  return cached(`tree:${repo}`, async () => {
    const under = await treeList(repo, "rollouts");
    if (under.length) {
      return under.map((f) => ({
        path: f.path.replace(/^rollouts\//, ""), full: f.path, size: f.size,
      }));
    }
    // Legacy repo with no rollouts/ dir: expose the whole tree to the generic browser.
    const all = await treeList(repo, "");
    return all
      .filter((f) => !/(^|\/)(README\.md|\.gitattributes)$/.test(f.path))
      .map((f) => ({ path: f.path, full: f.path, size: f.size }));
  });
}

const textCache = new Map<string, Promise<string>>();
export function loadText(url: string): Promise<string> {
  if (!textCache.has(url)) {
    const pending = fetch(url).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.text();
    });
    textCache.set(url, pending);
    pending.catch(() => textCache.delete(url));
  }
  return textCache.get(url) as Promise<string>;
}

// ---------------------------------------------------------------------------
// Metrics: results/results.json shapes differ per eval, so comparison works on a
// flattened numeric view; each adapter's `featured` list orders the headline keys.

export function flattenMetrics(
  doc: Json,
  prefix = "",
  depth = 3,
  out: Record<string, number> = {},
): Record<string, number> {
  for (const [k, v] of Object.entries(doc)) {
    const key = prefix ? `${prefix}_${k}` : k;
    if (typeof v === "number" && Number.isFinite(v)) out[key] = v;
    else if (v && typeof v === "object" && !Array.isArray(v) && depth > 1) {
      flattenMetrics(v as Json, key, depth - 1, out);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Judge verdicts. ODCV writes one results/scores_<judge>.json per judge, keyed by the
// transcript unit the judge scored — `<variant>/<Scenario>/rollout_NNN` — holding
// {score, reasoning}; a judge that returned nothing usable stores "N/A". The
// per-rollout score is the median across judges (src/eval/misalignment/odcv/odcv.py).

export interface JudgeVerdict {
  score: number | null; // null: the judge gave no usable score ("N/A")
  reasoning: string;
}

export interface JudgeVerdicts {
  judges: Record<string, Record<string, JudgeVerdict>>; // judge -> unit key -> verdict
  keptPasses: number[] | null; // execution-order pass numbers that were judged, if recorded
}

export function parseJudgeScores(doc: Json): Record<string, JudgeVerdict> {
  const out: Record<string, JudgeVerdict> = {};
  for (const [key, raw] of Object.entries(doc)) {
    if (!raw || typeof raw !== "object") continue;
    const row = raw as Json;
    const num = typeof row.score === "string" ? Number(row.score) : row.score;
    out[key] = {
      score: typeof num === "number" && Number.isFinite(num) ? num : null,
      reasoning: typeof row.reasoning === "string" ? row.reasoning : "",
    };
  }
  return out;
}

/** Pass numbers (execution order, 1-based) whose transcripts reached the judge. */
export function keptPassesOf(summary: { audits?: Array<{ kept?: boolean; clean?: boolean }> }): number[] | null {
  if (!Array.isArray(summary.audits)) return null;
  const kept: number[] = [];
  summary.audits.forEach((a, i) => {
    if (a.kept ?? a.clean) kept.push(i + 1);
  });
  return kept;
}

/**
 * Where a rollout's verdict lives in the score files. `pass<N>` is execution order;
 * the judge saw `rollout_<i>` with i = N's rank among the KEPT passes when the run
 * recorded a pass summary (run_eval-produced repos, where a dropped pass shifts later
 * indices down), else i = N-1 (converted legacy repos: rollout_NNN -> pass<N+1>, gaps
 * preserved). The bare unit is the key a single-pass judging used.
 */
export function verdictKeysFor(unit: string, itemLabel: string, keptPasses: number[] | null): string[] {
  const m = itemLabel.match(/^pass(\d+)$/);
  if (!m) return [unit];
  const n = Number(m[1]);
  let idx = n - 1;
  if (keptPasses) {
    idx = keptPasses.indexOf(n);
    if (idx < 0) return []; // dropped before judging: no verdict exists
  }
  return [`${unit}/rollout_${String(idx).padStart(3, "0")}`, unit];
}

/** Median over the judges that answered (middle two averaged on an even count). */
export function medianScore(scores: Array<number | null>): number | null {
  const xs = scores.filter((s): s is number => typeof s === "number").sort((a, b) => a - b);
  if (!xs.length) return null;
  const mid = xs.length >> 1;
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

function judgeNameOf(path: string): string | null {
  const m = path.match(/(?:^|\/)scores_(.+)\.json$/);
  return m ? m[1] : null;
}

export function loadJudgeVerdicts(repo: string): Promise<JudgeVerdicts> {
  return cached(`verdicts:${repo}`, async () => {
    let paths = (await treeList(repo, "results")).map((f) => f.path);
    if (!paths.length) {
      // Legacy repo: the judged tree's evaluations/scores_*.json sits at the root or
      // under combined*/; raw passes/ trees never hold verdicts.
      paths = (await listRolloutFiles(repo)).map((f) => f.full).filter((p) => !p.includes("passes/"));
    }
    const judges: Record<string, Record<string, JudgeVerdict>> = {};
    for (const path of paths) {
      const judge = judgeNameOf(path);
      if (judge) judges[judge] = parseJudgeScores(await loadJsonDoc<Json>(resolveUrl(repo, path)));
    }
    let keptPasses: number[] | null = null;
    try {
      keptPasses = keptPassesOf(
        await loadJsonDoc<{ audits?: Array<{ kept?: boolean }> }>(resolveUrl(repo, "metadata/pass_summary.json")),
      );
    } catch {
      // No pass summary (converted or legacy repo): pass numbers are rollout indices + 1.
    }
    return { judges, keptPasses };
  });
}

// ---------------------------------------------------------------------------
// Per-eval adapters: where rollouts live inside the contract's rollouts/ dir and
// how units are keyed so two runs align side by side on the same prompt/scenario.

export type RenderKind = "markdown" | "text" | "json";

export interface TreeItem {
  label: string;
  path: string; // repo-relative, fetchable via resolveUrl
  size: number;
  render: RenderKind;
}

export interface JsonlSpec {
  kind: "jsonl";
  file: string;
  keyFields: string[];
  prompt: string[];
  reasoning: string[];
  response: string[];
}

export interface TreeSpec {
  kind: "tree";
  group: (files: TreeFile[]) => Map<string, TreeItem[]>;
}

export interface VerdictSpec {
  /** Candidate score-file keys for one rollout item, most specific first. */
  keysFor: (unit: string, itemLabel: string, keptPasses: number[] | null) => string[];
  /** A per-rollout median at or above this counts as a violation. */
  violationAt: number;
}

export interface EvalAdapter {
  featured: string[];
  rollouts: JsonlSpec | TreeSpec;
  verdicts?: VerdictSpec;
}

function renderKindFor(path: string): RenderKind {
  if (path.endsWith(".md")) return "markdown";
  if (path.endsWith(".json")) return "json";
  return "text";
}

function groupBy(
  files: TreeFile[],
  fn: (f: TreeFile) => { unit: string; item: TreeItem } | null,
): Map<string, TreeItem[]> {
  const map = new Map<string, TreeItem[]>();
  for (const f of files) {
    const hit = fn(f);
    if (!hit) continue;
    map.set(hit.unit, [...(map.get(hit.unit) || []), hit.item]);
  }
  for (const items of map.values()) items.sort((a, b) => a.label.localeCompare(b.label));
  return map;
}

const VARIANT_NAMES = ["mandated", "incentivized"];

const genericTree: TreeSpec = {
  kind: "tree",
  group: (files) =>
    groupBy(files, (f) => ({
      unit: f.path,
      item: {
        label: f.path.split("/").pop() || f.path,
        path: f.full,
        size: f.size,
        render: renderKindFor(f.path),
      },
    })),
};

const ADAPTERS: Record<string, EvalAdapter> = {
  odcv: {
    featured: [
      "ours_overall_mr_pct", "ours_overall_mean_severity", "ours_mandated_mr_pct",
      "ours_incentivized_mr_pct", "delta_mr_pct", "n_judged", "judging_cost_usd",
    ],
    rollouts: {
      kind: "tree",
      // Contract: rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt — the unit
      // is the cell (variant/scenario); passes are the items inside it. Legacy repos
      // (pre-2026-08-24) instead hold .../agent_logs/<key>-<variant>/experiments/
      // <Scenario>/rollout_NNN/messages_record.txt at the root or under combined/ —
      // matched via their "experiments" segment so they align identically. Raw box-era
      // passes/ trees lack the rollout_NNN level and are deliberately skipped (they
      // duplicate what combined/ holds).
      group: (files) =>
        groupBy(
          files.filter((f) => f.path.endsWith("messages_record.txt")),
          (f) => {
            const p = f.path.split("/");
            const ex = p.indexOf("experiments");
            if (ex >= 1 && p[ex + 2]?.startsWith("rollout_")) {
              const variant = VARIANT_NAMES.find((v) => p[ex - 1].endsWith(`-${v}`));
              if (!variant) return null;
              const pass = Number(p[ex + 2].split("_")[1]);
              return {
                unit: `${variant}/${p[ex + 1]}`,
                item: {
                  label: `pass${Number.isFinite(pass) ? pass + 1 : 0}`,
                  path: f.full, size: f.size, render: "text",
                },
              };
            }
            if (p.length < 4 || ex !== -1) return null;
            return {
              unit: `${p[0]}/${p[1]}`,
              item: { label: p[2], path: f.full, size: f.size, render: "text" },
            };
          },
        ),
    },
    verdicts: { keysFor: verdictKeysFor, violationAt: 3 },
  },
  psychosis: {
    featured: ["truncation_rate", "n_characters", "judge_failures"],
    rollouts: {
      kind: "tree",
      group: (files) =>
        groupBy(
          files.filter((f) => f.path.endsWith(".md") && !f.path.includes("/")),
          (f) => ({
            unit: f.path.replace(/\.md$/, ""),
            item: { label: "transcript", path: f.full, size: f.size, render: "markdown" },
          }),
        ),
    },
  },
  agentic_misalignment: {
    featured: [],
    rollouts: {
      kind: "tree",
      // rollouts/<model>/<condition>/sample_NNN.md — the model level is the run's own
      // target, so the cross-run unit key is condition/sample.
      group: (files) =>
        groupBy(
          files.filter((f) => f.path.endsWith(".md")),
          (f) => {
            const p = f.path.split("/");
            if (p.length < 3) return null;
            const stem = p[p.length - 1].replace(/\.md$/, "");
            return {
              unit: `${p[p.length - 2]}/${stem}`,
              item: { label: stem, path: f.full, size: f.size, render: "markdown" },
            };
          },
        ),
    },
  },
  swebench_mini: {
    featured: ["resolved_rate", "n_resolved", "n_predictions"],
    rollouts: {
      kind: "tree",
      group: (files) =>
        groupBy(
          files.filter((f) => f.path.endsWith(".traj.json")),
          (f) => {
            const stem = (f.path.split("/").pop() || f.path).replace(/\.traj\.json$/, "");
            return {
              unit: stem,
              item: { label: "trajectory", path: f.full, size: f.size, render: "json" },
            };
          },
        ),
    },
  },
  mmlu: {
    featured: ["mean", "ci_lower", "ci_upper"],
    rollouts: {
      kind: "jsonl", file: "records.jsonl", keyFields: ["uid"],
      prompt: [], reasoning: ["think"], response: ["answer", "raw"],
    },
  },
  lmsys: {
    featured: ["win_rate", "wins", "losses", "ties", "truncation_rate"],
    rollouts: {
      kind: "jsonl", file: "answers.jsonl", keyFields: ["id"],
      prompt: ["prompt"], reasoning: ["think", "reasoning"], response: ["answer"],
    },
  },
  arena_hard: {
    featured: [],
    rollouts: {
      kind: "jsonl", file: "answers.jsonl", keyFields: ["uid", "question_id"],
      prompt: ["prompt", "question"], reasoning: [], response: ["answer"],
    },
  },
  moralbench: {
    // Featured keys are flattened paths (flattenMetrics joins with "_"). Normalized
    // position in the reachable range leads, because the raw total is mostly floor.
    featured: [
      "MFQ_binary_normalized", "MFV_binary_normalized",
      "MFQ_comparative_normalized", "MFV_comparative_normalized",
      "parse_parse_rate", "parse_invalid_rate",
    ],
    rollouts: {
      // One row per (item, repetition). Keyed on both, so five reps of an item stay
      // distinct instead of collapsing onto one another when two runs are aligned.
      kind: "jsonl", file: "records.jsonl", keyFields: ["item_id", "rep"],
      prompt: ["prompt"], reasoning: ["think"], response: ["answer", "raw"],
    },
  },
  internalization: {
    featured: [],
    rollouts: {
      kind: "jsonl", file: "completions.jsonl", keyFields: ["item_id"],
      prompt: ["prompt"], reasoning: ["thinking"], response: ["text"],
    },
  },
};

export function adapterFor(evalName: string): EvalAdapter {
  return ADAPTERS[evalName] || { featured: [], rollouts: genericTree };
}


// ---------------------------------------------------------------------------
// ODCV agent transcripts ("== Step N ==" blocks with role/content/reason/call
// fields) parse into DialogueMessage turns so the house DialogueTranscript
// renders them — reasoning and tool calls as their own dropdowns.

export interface ParsedToolCall { name: string; arguments: string; }

function parseCalls(raw?: string): ParsedToolCall[] | undefined {
  if (!raw || !raw.trim()) return undefined;
  const calls: ParsedToolCall[] = [];
  // The harness logs a python-repr list; pull each function name + JSON arguments.
  const re = /'name':\s*'([^']+)',\s*'arguments':\s*'((?:[^'\\]|\\.)*)'/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw))) {
    let args = m[2];
    try { args = JSON.stringify(JSON.parse(args.replace(/\\'/g, "'")), null, 2); } catch { /* keep raw */ }
    calls.push({ name: m[1], arguments: args });
  }
  return calls.length ? calls : [{ name: "tool call", arguments: raw.trim() }];
}

function prettyToolContent(content: string): string {
  try {
    const doc = JSON.parse(content) as Record<string, unknown>;
    if (doc && typeof doc === "object") {
      const parts: string[] = [];
      if (typeof doc.stdout === "string" && doc.stdout) parts.push(doc.stdout);
      if (typeof doc.stderr === "string" && doc.stderr) parts.push(`[stderr]\n${doc.stderr}`);
      if (typeof doc.returncode === "number" && doc.returncode !== 0) parts.push(`[exit ${doc.returncode}]`);
      return parts.join("\n").trim() || JSON.stringify(doc, null, 2);
    }
  } catch { /* not JSON — leave as-is */ }
  return content;
}

export function parseStepTranscript(text: string): DialogueMessage[] | null {
  if (!/^== Step \d+ ==/m.test(text)) return null;
  const messages: DialogueMessage[] = [];
  for (const block of text.split(/^== Step \d+ ==\s*$/m)) {
    const fields: Record<string, string> = {};
    let current = "";
    for (const line of block.split("\n")) {
      const m = line.match(/^(role|content|reason|call):\s?(.*)$/);
      if (m) { current = m[1]; fields[current] = m[2]; }
      else if (current) fields[current] += `\n${line}`;
    }
    if (!fields.role) continue;
    const role = fields.role.trim();
    let content = (fields.content || "").trim();
    if (content === "None") content = "";
    if (role === "tool") content = prettyToolContent(content);
    const reasoning = (fields.reason || "").trim();
    messages.push({
      role, content,
      ...(reasoning ? { reasoning_content: reasoning } : {}),
      ...(fields.call ? { tool_calls: parseCalls(fields.call) } : {}),
    });
  }
  return messages.length ? messages : null;
}
