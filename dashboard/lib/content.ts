import contentIndex from "./generated/content-index.json";

export type Metric = {
  value: number;
  unit?: string;
  lower_is_better?: boolean;
  [key: string]: unknown;
};

export type Asset = {
  name: string;
  path: string;
  size_bytes: number;
  kind: string;
};

export type DialogueMessage = {
  role: string;
  content: string;
  name?: string;
  tool_call_id?: string;
  reasoning_content?: string;
  [key: string]: unknown;
};

export type DialogueRecord = {
  id: string;
  messages?: DialogueMessage[];
  conversation?: DialogueMessage[];
  turns?: DialogueMessage[];
  dialogue?: DialogueMessage[];
  prompt?: string;
  response?: string;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
};

/**
 * Where an entry's bulk payload comes from. `local` means the build sharded
 * files that live in `content/`; `hf` means the shards live in a published
 * Hugging Face dataset and the browser fetches them from the Hub directly.
 */
export type PayloadSource = {
  kind: "local" | "hf";
  repo_id?: string;
  revision?: string;
  commit?: string;
  url?: string;
  /** Set when an HF source was declared but the build fell back to disk. */
  fallback_from?: string;
};

/** Build-time outcome of resolving an entry's `hf_source`. */
export type HfStatus = {
  state: "ok" | "stale" | "unavailable";
  repo_id: string;
  revision: string;
  commit?: string;
  url: string;
  cached?: boolean;
  /** Present when `state` is `unavailable`; safe to show to a reader. */
  message?: string;
};

export type DatasetManifest = {
  source: PayloadSource;
  source_file: string;
  format: string;
  record_count: number;
  chunk_size: number;
  chunks: string[];
  stats: {
    average_turns: number;
    role_counts: Record<string, number>;
    splits: Record<string, number>;
    categories: Record<string, number>;
  };
  deferred_bytes?: number;
};

export type PetriScenario = {
  id: string;
  category: string;
  hypothesis: string;
  seed_instruction: string;
  expected_signal: string;
  source?: string;
};

/**
 * The full transcript, as exported. Lives in a per-transcript sidecar and is
 * fetched only when a reader opens it - never baked into the page.
 */
export type PetriTranscript = DialogueRecord & {
  scenario_id: string;
  category: string;
  outcome: string;
  messages: DialogueMessage[];
  scores: Record<string, number>;
  judge_summary: string;
  tags: string[];
};

/**
 * The baked-in row for one transcript: enough to list, filter and sort the
 * whole run without loading a single message body.
 */
export type PetriTranscriptSummary = {
  id: string;
  /** Sidecar file name, relative to `PetriManifest.transcript_base`. */
  file: string;
  scenario_id: string;
  category: string;
  outcome: string;
  scores: Record<string, number>;
  tags: string[];
  message_count: number;
  size_bytes: number;
};

export type PetriManifest = {
  source: PayloadSource;
  scenarios: PetriScenario[];
  /** Summaries only. Bodies come from `transcript_base` on demand. */
  transcript_index: PetriTranscriptSummary[];
  /** URL prefix the sidecars hang off: a site path, or an HF resolve URL. */
  transcript_base: string;
  transcript_count: number;
  /** Bytes kept out of the initial payload by deferring transcript bodies. */
  deferred_bytes?: number;
  scores: {
    by_category?: Array<{
      category: string;
      audits: number;
      concerning: number;
      eval_aware: number;
      mean_realism: number;
    }>;
    outcomes?: Record<string, number>;
    retained_transcripts?: number;
    filtered_for_realism?: number;
  };
};

export type ResearchEntry = {
  id: string;
  slug: string;
  type: "logs" | "evals" | "findings" | "datasets" | "petri-runs";
  title: string;
  date: string;
  summary: string;
  status: string;
  tags: string[];
  models: string[];
  metrics: Record<string, Metric>;
  body: string;
  source_path: string;
  assets: Asset[];
  model_id?: string;
  checkpoint_id?: string;
  parent_checkpoint_id?: string;
  training_stage?: string;
  training_method?: string;
  run_id?: string;
  seed?: number;
  eval_suite?: string;
  eval_version?: string;
  dataset_version?: string;
  git_commit?: string;
  related?: string[];
  dataset?: DatasetManifest;
  dataset_id?: string;
  training_objective?: string;
  generator_model?: string;
  petri?: PetriManifest;
  petri_run_id?: string;
  petri_version?: string;
  target_model_id?: string;
  target_checkpoint_id?: string;
  auditor_model_id?: string;
  judge_model_id?: string;
  realism_model_id?: string;
  seed_set?: string;
  max_turns?: number;
  realism_filter?: boolean;
  realism_threshold?: number;
  /** Resolution status of this entry's `hf_source`, when it declares one. */
  hf?: HfStatus;
  [key: string]: unknown;
};

export const entries = contentIndex.entries as ResearchEntry[];

/** Build-time Hugging Face status, including any degradation notices. */
export const hfBuildInfo = (contentIndex as { hf?: unknown }).hf as
  | {
      endpoint: string;
      token_present: boolean;
      offline: boolean;
      notices: string[];
    }
  | undefined;

export function entriesOfType(type: ResearchEntry["type"]) {
  return entries.filter((entry) => entry.type === type);
}

export function entryBySlug(slug: string) {
  return entries.find((entry) => entry.slug === slug);
}

export function modelsInCorpus() {
  const models = new Map<string, ResearchEntry[]>();
  for (const entry of entries) {
    for (const model of entry.models) {
      const existing = models.get(model) || [];
      existing.push(entry);
      models.set(model, existing);
    }
  }
  return [...models.entries()]
    .map(([id, modelEntries]) => ({ id, entries: modelEntries }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

export function humanize(value: string) {
  return value
    .replaceAll("-", " ")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatMetric(metric: Metric) {
  const { value, unit } = metric;
  if (unit === "proportion") return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
  if (unit === "USD") return `$${value.toFixed(2)}`;
  if (unit === "minutes") return `${value.toLocaleString()} min`;
  return `${value.toLocaleString()}${unit ? ` ${unit}` : ""}`;
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
