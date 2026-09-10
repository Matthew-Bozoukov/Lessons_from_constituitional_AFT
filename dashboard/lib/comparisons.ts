// ABOUTME: Versioned public dataset-to-model comparisons, discovered by HF card tag.
// ABOUTME: Validate evidence and compare only identical declared evaluation protocols.
import { cached, loadJsonDoc } from "./lazy.ts";
import { EVAL_ORG } from "./evalRuns.ts";

export const COMPARISON_TAG = "dataset-model-comparison";
export const COMPARISON_FILE = "results/dataset_comparison.json";
export type Evidence = { label: string; url: string };
export type TraitValue = { value: string | number | null; basis: "measured" | "design" | "unmeasured"; evidence: Evidence[] };
export type Interval = { low: number; high: number; method: string };
export type Outcome = { value: number; numerator?: number; denominator?: number; interval?: Interval };
export type Arm = {
  id: string; label: string;
  dataset: { repo: string; revision: string; file: string; subset: string; row_count: number };
  model: { repo: string; revision: string; base_revision: string; seed: number | null };
  evaluation: { repo: string; revision: string; protocol: Record<string, unknown>; repeats: number; metrics: Record<string, Outcome> };
  traits: Record<string, TraitValue>;
};
export type Comparison = {
  schema_version: 1; title: string; summary: string; limitations: string[];
  traits: { id: string; label: string; description: string }[];
  metrics: { id: string; label: string; unit: string; lower_is_better?: boolean }[];
  arms: Arm[];
  contrasts: { baseline: string; arm: string; metric: string; delta: number; interval?: Interval }[];
};
export type ComparisonRepo = { repo: string; revision: string; title: string };
const hub = "https://huggingface.co";
const repoPattern = /^[\w.-]+\/[\w.-]+$/;
const revisionPattern = /^[a-f0-9]{40}$/;

function requireValue(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(`Invalid comparison: ${message}`);
}
function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
function text(value: unknown): value is string { return typeof value === "string" && !!value.trim(); }
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value); }
function interval(value: unknown, center: number) {
  if (value === undefined) return;
  requireValue(object(value) && finite(value.low) && finite(value.high) && text(value.method)
    && value.low <= center && center <= value.high, "interval must contain its estimate and name its method");
}
function reference(value: Record<string, unknown>) {
  requireValue(typeof value.repo === "string" && repoPattern.test(value.repo), "invalid HF repo");
  requireValue(typeof value.revision === "string" && revisionPattern.test(value.revision), "immutable revision required");
}
function declarations(value: unknown, name: string): Set<string> {
  requireValue(Array.isArray(value) && value.length > 0, `${name} must be declared`);
  const ids = new Set<string>();
  for (const item of value) {
    requireValue(object(item) && text(item.id) && text(item.label) && !ids.has(item.id), `invalid or duplicate ${name} id`);
    ids.add(item.id);
  }
  return ids;
}
export function protocolKey(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(protocolKey).join(",")}]`;
  if (object(value)) return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${protocolKey(value[k])}`).join(",")}}`;
  return JSON.stringify(value);
}
export function compatible(a: Arm, b: Arm): boolean {
  return protocolKey(a.evaluation.protocol) === protocolKey(b.evaluation.protocol);
}
export function parseComparison(value: unknown): Comparison {
  requireValue(object(value) && value.schema_version === 1, "unsupported schema version");
  requireValue(text(value.title) && text(value.summary), "title and summary required");
  requireValue(Array.isArray(value.limitations) && value.limitations.length > 0 && value.limitations.every(text), "limitations required");
  const traits = declarations(value.traits, "traits"), metrics = declarations(value.metrics, "metrics");
  for (const metric of value.metrics as Record<string, unknown>[]) {
    requireValue(typeof metric.unit === "string" && (metric.lower_is_better === undefined || typeof metric.lower_is_better === "boolean"), "metric units/direction invalid");
  }
  for (const trait of value.traits as Record<string, unknown>[]) requireValue(text(trait.description), "trait description required");
  requireValue(Array.isArray(value.arms) && value.arms.length >= 2, "at least two arms required");
  const ids = new Set<string>();
  for (const arm of value.arms) {
    requireValue(object(arm) && text(arm.id) && text(arm.label) && !ids.has(arm.id), "invalid or duplicate arm");
    ids.add(arm.id);
    requireValue(object(arm.dataset) && object(arm.model) && object(arm.evaluation), "dataset, model and evaluation required");
    for (const ref of [arm.dataset, arm.model, arm.evaluation]) reference(ref);
    requireValue(Number.isInteger(arm.dataset.row_count) && Number(arm.dataset.row_count) > 0
      && text(arm.dataset.subset) && text(arm.dataset.file), "dataset population required");
    requireValue(typeof arm.model.base_revision === "string" && revisionPattern.test(arm.model.base_revision)
      && (arm.model.seed === null || Number.isInteger(arm.model.seed)), "model base pin and seed required");
    requireValue(object(arm.evaluation.protocol) && text(arm.evaluation.protocol.benchmark)
      && text(arm.evaluation.protocol.scenario_set) && Object.keys(arm.evaluation.protocol).length >= 3,
    "explicit benchmark, scenario set and evaluation settings required");
    requireValue(Number.isInteger(arm.evaluation.repeats) && Number(arm.evaluation.repeats)>0, "evaluation repeats required");
    requireValue(object(arm.evaluation.metrics) && object(arm.traits), "metrics and traits required");
    for (const [key, score] of Object.entries(arm.evaluation.metrics)) {
      requireValue(metrics.has(key) && object(score) && finite(score.value), "invalid metric value");
      if (score.numerator !== undefined || score.denominator !== undefined) {
        requireValue(Number.isInteger(score.numerator) && Number.isInteger(score.denominator)
          && Number(score.denominator)>0 && Number(score.numerator)>=0 && Number(score.numerator)<=Number(score.denominator), "invalid counts");
        const metric = (value.metrics as { id: string; unit: string }[]).find(m => m.id === key)!;
        if (metric.unit === "%" || metric.unit === "proportion") requireValue(
          Math.abs(score.value - Number(score.numerator)/Number(score.denominator)*(metric.unit === "%" ? 100 : 1)) < 1e-6,
          "rate differs from exact counts");
      }
      interval(score.interval, score.value);
    }
    for (const [key, trait] of Object.entries(arm.traits)) {
      requireValue(traits.has(key) && object(trait) && ["measured", "design", "unmeasured"].includes(String(trait.basis)), "invalid trait");
      requireValue(trait.basis === "unmeasured" ? trait.value === null : (text(trait.value) || finite(trait.value)), "unknown traits must be null");
      requireValue(Array.isArray(trait.evidence) && (trait.basis === "unmeasured" || trait.evidence.length > 0), "trait evidence required");
      for (const evidence of trait.evidence) requireValue(object(evidence) && text(evidence.label)
        && typeof evidence.url === "string" && /^https:\/\//.test(evidence.url), "invalid evidence link");
    }
  }
  requireValue(Array.isArray(value.contrasts), "contrasts must be an array");
  const parsed = value as unknown as Comparison;
  const contrastIds = new Set<string>();
  for (const contrast of parsed.contrasts) {
    const a = parsed.arms.find(x => x.id === contrast.baseline), b = parsed.arms.find(x => x.id === contrast.arm);
    requireValue(a && b && a !== b && metrics.has(contrast.metric), "invalid contrast references");
    const key = [contrast.baseline, contrast.arm, contrast.metric].join(":");
    requireValue(!contrastIds.has(key), "duplicate contrast"); contrastIds.add(key);
    const x = a.evaluation.metrics[contrast.metric], y = b.evaluation.metrics[contrast.metric];
    requireValue(compatible(a,b) && x && y && finite(contrast.delta) && Math.abs(y.value-x.value-contrast.delta)<1e-6,
      "contrast must match compatible outcomes");
    interval(contrast.interval, contrast.delta);
  }
  return parsed;
}
export function difference(data: Comparison, baseline: Arm, arm: Arm, metric: string) {
  const a=baseline.evaluation.metrics[metric], b=arm.evaluation.metrics[metric];
  if (!compatible(baseline,arm) || !a || !b) return null;
  const exact=data.contrasts.find(c=>c.baseline===baseline.id && c.arm===arm.id && c.metric===metric);
  const reverse=data.contrasts.find(c=>c.arm===baseline.id && c.baseline===arm.id && c.metric===metric);
  return { value:b.value-a.value, interval:exact?.interval ?? (reverse?.interval ? {
    low:-reverse.interval.high, high:-reverse.interval.low, method:reverse.interval.method,
  } : undefined) };
}
export function traitDiffers(data: Comparison, id: string): boolean {
  return new Set(data.arms.map(a=>JSON.stringify(a.traits[id]?.value ?? null))).size>1;
}
export function artifactUrl(repo: string, revision: string, model = false): string {
  return `${hub}/${model ? "" : "datasets/"}${repo}/tree/${revision}`;
}
export async function listComparisons(org = EVAL_ORG): Promise<ComparisonRepo[]> {
  return cached(`comparisons:${org}`, async()=>{
    let url: string | null = `${hub}/api/datasets?author=${encodeURIComponent(org)}&filter=${COMPARISON_TAG}&limit=100&full=true`;
    const rows: ComparisonRepo[] = [];
    while (url) {
      const response: Response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status} listing comparisons`);
      for (const item of await response.json()) {
        if (typeof item.id !== "string" || !repoPattern.test(item.id) || !revisionPattern.test(item.sha ?? "")) throw new Error("Comparison listing lacks immutable revision");
        rows.push({repo:item.id, revision:item.sha, title:item.cardData?.pretty_name || item.id.split("/")[1]});
      }
      const next: string | undefined = response.headers.get("Link")?.match(/<([^>]+)>;\s*rel="next"/)?.[1];
      if (next && new URL(next).origin !== hub) throw new Error("Unexpected comparison pagination host");
      url=next ?? null;
    }
    return rows;
  });
}
export async function loadComparison(repo: ComparisonRepo): Promise<Comparison> {
  return parseComparison(await loadJsonDoc(`${hub}/datasets/${repo.repo}/resolve/${repo.revision}/${COMPARISON_FILE}`));
}
