"use client";

// ABOUTME: Typed reader for MoralBench results/results.json: the four blocks, their
// ABOUTME: reachable bounds, per-foundation breakdowns and parse health.

// MoralBench is a values PROFILE, not a leaderboard, and the shape of its results makes
// two mistakes very easy. Both are handled here rather than in the components:
//
//  1. **Raw totals compress differences.** Both binary options score, so the floor is
//     enormous — 60% of the ceiling on MFQ, 74% on MFV. Comparing arms on `total` hides
//     most of the signal, so `normalized` (where the arm sits in the REACHABLE range) is
//     what the UI plots, with the raw total kept beside it.
//  2. **Binary and comparative share no scale.** One is a weighted human-agreement score,
//     the other is accuracy against an answer key. They are never summed, and the
//     comparative half sits at chance for every model published so far — `CHANCE` below
//     is drawn so a reader can see that rather than infer it.
//
// Higher is not "better": the score rewards agreement with the MFQ/MFV human norming
// sample, so a constitution-trained arm that downweights tradition scores LOWER. The UI
// says so; this module just refuses to imply otherwise by ranking.

import { cached, loadJsonDoc } from "./lazy.ts";
import { resolveUrl, type Json } from "./evalRuns";

export const BLOCKS = [
  "MFQ_binary", "MFV_binary", "MFQ_comparative", "MFV_comparative",
] as const;
export type BlockKey = (typeof BLOCKS)[number];

export const BLOCK_LABEL: Record<BlockKey, string> = {
  MFQ_binary: "MFQ-30 · binary",
  MFV_binary: "MFV · binary",
  MFQ_comparative: "MFQ-30 · comparative",
  MFV_comparative: "MFV · comparative",
};

/** Expected score from answering at random, where that is meaningful (comparative only).
 *  MFQ's floor is 1.0 not 0, because `ingroup_2` scores 1 for either answer. */
export const CHANCE: Partial<Record<BlockKey, number>> = {
  MFQ_comparative: 10.5,
  MFV_comparative: 12.0,
};

export const FOUNDATIONS = [
  "care", "fairness", "loyalty", "authority", "sanctity", "liberty",
] as const;
export type Foundation = (typeof FOUNDATIONS)[number];

export const FOUNDATION_LABEL: Record<Foundation, string> = {
  care: "Care / Harm",
  fairness: "Fairness / Cheating",
  loyalty: "Loyalty / Betrayal",
  authority: "Authority / Subversion",
  sanctity: "Sanctity / Degradation",
  liberty: "Liberty / Oppression",
};

/** Moral Foundations Theory is conventionally colour-coded, so hue here is information
 *  rather than decoration — the same foundation keeps its colour across every chart. */
export const FOUNDATION_COLOR: Record<Foundation, string> = {
  care: "var(--red)",
  fairness: "var(--amber)",
  loyalty: "var(--lime)",
  authority: "var(--cyan)",
  sanctity: "var(--violet)",
  liberty: "var(--cyan-dim)",
};

export interface Block {
  total: number;
  nItems: number;
  min: number;
  max: number;
  /** null when the block has no spread (max === min) and a ratio would divide by zero. */
  normalized: number | null;
  /** Present only where duplicate prompts make the per-item ceiling unreachable for a
   *  model that answers identical prompts identically (MFV comparative: 23, not 24). */
  maxDeterministic?: number;
  byFoundation: Partial<Record<Foundation, Block>>;
}

export interface MoralBenchRun {
  repo: string;
  target: string;
  mode: string;
  repetitions: number;
  swapOptions: boolean;
  blocks: Partial<Record<BlockKey, Block>>;
  parseRate: number | null;
  invalidRate: number | null;
  answerBalance: { A: number; B: number } | null;
  totalsByRepetition: Array<{ rep: string; total: number }>;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function readBlock(raw: unknown): Block | null {
  if (!raw || typeof raw !== "object") return null;
  const doc = raw as Json;
  const total = num(doc.total);
  if (total === null) return null;
  const byFoundation: Partial<Record<Foundation, Block>> = {};
  const nested = doc.by_foundation;
  if (nested && typeof nested === "object") {
    for (const f of FOUNDATIONS) {
      const sub = readBlock((nested as Json)[f]);
      if (sub) byFoundation[f] = sub;
    }
  }
  return {
    total,
    nItems: num(doc.n_items) ?? 0,
    min: num(doc.min_possible) ?? 0,
    max: num(doc.max_possible) ?? 0,
    normalized: num(doc.normalized),
    ...(num(doc.max_possible_deterministic) !== null
      ? { maxDeterministic: num(doc.max_possible_deterministic) as number }
      : {}),
    byFoundation,
  };
}

/** Parse one run's results.json. Returns null when the doc is not a MoralBench result,
 *  so a mis-tagged repo is skipped rather than rendered as an empty row. */
export function parseRun(repo: string, doc: Json): MoralBenchRun | null {
  const blocks: Partial<Record<BlockKey, Block>> = {};
  for (const key of BLOCKS) {
    const block = readBlock(doc[key]);
    if (block) blocks[key] = block;
  }
  if (!Object.keys(blocks).length) return null;

  const parse = (doc.parse || {}) as Json;
  const balance = parse.answer_balance as Json | undefined;
  const reps = (doc.totals_by_repetition || {}) as Json;

  return {
    repo,
    target: typeof doc.target === "string" ? doc.target : repo,
    mode: typeof doc.mode === "string" ? doc.mode : "",
    repetitions: num(doc.repetitions) ?? num(doc.n_repetitions) ?? 1,
    swapOptions: doc.swap_options === true,
    blocks,
    parseRate: num(parse.parse_rate),
    invalidRate: num(parse.invalid_rate),
    answerBalance: balance
      ? { A: num(balance.A) ?? 0, B: num(balance.B) ?? 0 }
      : null,
    totalsByRepetition: Object.entries(reps)
      .map(([rep, v]) => ({ rep, total: num(v) ?? 0 }))
      .sort((a, b) => a.rep.localeCompare(b.rep)),
  };
}

export function loadMoralBenchRun(repo: string): Promise<MoralBenchRun | null> {
  return cached(`moralbench:${repo}`, async () => {
    const doc = await loadJsonDoc<Json>(resolveUrl(repo, "results/results.json"));
    return parseRun(repo, doc);
  });
}

/** Fraction of the reachable range an arm reached, 0..1, for bar geometry.
 *  Can fall slightly BELOW zero: an unparsed answer scores 0, which is under every
 *  reachable binary score, so a block with invalid answers can undershoot its own floor.
 *  Clamped for drawing; the caller still shows the true value. */
export function barFraction(block: Block): number {
  if (block.normalized === null) return 0;
  return Math.max(0, Math.min(1, block.normalized));
}

/** True when this block's total is below what random answering would score. Only
 *  meaningful for the comparative halves, where a chance baseline exists. */
export function belowChance(key: BlockKey, block: Block): boolean {
  const chance = CHANCE[key];
  return chance !== undefined && block.total < chance;
}

/** Short label for a run in a legend: model, then mode, so two modes of one model are
 *  distinguishable (they are never comparable — the framework refuses to pair them). */
export function runLabel(run: MoralBenchRun, model: string): string {
  return [model || run.target, run.mode].filter(Boolean).join(" · ");
}
