// ABOUTME: Pairing two training corpora for side-by-side reading: the prompt a record is
// ABOUTME: aligned on, the subset rule that decides whether a pair is legible, and the rows.

// Why a subset rule and not a best-effort join.
//
// The comparison this exists for is "same prompts, different reasoning and responses": an
// ablation arm against the control it was derived from, a Grok-drafted corpus against the
// Sonnet original. Those pairs share their prompt set exactly, and reading them side by
// side is reading the intervention.
//
// Two corpora that merely OVERLAP are a different thing. Aligning them would put a record
// beside whichever record happened to match and skip elsewhere, so a reader scrolling the
// pair would see differences that are an artifact of the matching rather than of the data.
// Refusing, with the counts that say why, is the honest answer, and it is what this
// returns.
//
// Other pairings are imaginable (align on a shared `scenario_id`, on trait, on record
// order). None is implemented: each needs its own argument for why the alignment means
// something, and inventing one silently is how a viewer starts lying.

import type { NormalizedRecord } from "./records";

/** Joins the parts of a key: NUL, which message content never contains. */
const SEP = "\u0000";

/**
 * The prompt a record was written for: every message before the first assistant turn,
 * role and content.
 *
 * The system turn is part of the key. Two corpora whose user turns match but whose system
 * prompts differ were written for different conditions, and calling those the same prompt
 * would present a prompt difference as a response difference — the exact confusion the
 * side-by-side view exists to avoid. Content is trimmed because the chat template's own
 * trailing newlines are rendering, not prompt.
 */
export function promptKey(record: NormalizedRecord): string {
  const messages = record.messages;
  const firstAssistant = messages.findIndex((message) => message.role === "assistant");
  const prompt = firstAssistant < 0 ? messages : messages.slice(0, firstAssistant);
  return prompt
    .map((message) => `${message.role}${SEP}${String(message.content ?? "").trim()}`)
    .join(SEP);
}

/** One aligned line: the same prompt as each corpus publishes it. */
export type PairedRow = { key: string; a?: NormalizedRecord; b?: NormalizedRecord };

export type Pairing =
  | {
      ok: true;
      /** Which way the containment runs, as measured — never assumed from record counts. */
      direction: "equal" | "a-in-b" | "b-in-a";
      rows: PairedRow[];
      /** Distinct prompts present in both. */
      shared: number;
      /** Records of the containing corpus left with no counterpart, so not shown. */
      hidden: number;
    }
  | { ok: false; reason: string; shared: number; onlyA: number; onlyB: number };

type Keyed = { key: string; record: NormalizedRecord };

function keyed(records: NormalizedRecord[]): Keyed[] {
  return records.map((record) => ({ key: promptKey(record), record }));
}

function buckets(rows: Keyed[]): Map<string, NormalizedRecord[]> {
  const map = new Map<string, NormalizedRecord[]>();
  for (const { key, record } of rows) {
    const bucket = map.get(key);
    if (bucket) bucket.push(record);
    else map.set(key, [record]);
  }
  return map;
}

/**
 * Align two corpora on their prompts, or refuse and say why.
 *
 * Succeeds only when one corpus's set of prompts is contained in the other's, equal sets
 * included. The contained corpus is walked in its own order, so the rows read in the order
 * that file publishes them; a prompt occurring more than once in it is paired against
 * successive copies on the other side, and a copy with nothing left to pair against keeps
 * its row with that half empty rather than being dropped.
 */
export function pairCorpora(a: NormalizedRecord[], b: NormalizedRecord[]): Pairing {
  const rowsA = keyed(a);
  const rowsB = keyed(b);
  const byA = buckets(rowsA);
  const byB = buckets(rowsB);

  let shared = 0;
  let onlyA = 0;
  for (const key of byA.keys()) {
    if (byB.has(key)) shared += 1;
    else onlyA += 1;
  }
  let onlyB = 0;
  for (const key of byB.keys()) if (!byA.has(key)) onlyB += 1;

  if (!shared) {
    return {
      ok: false,
      shared,
      onlyA,
      onlyB,
      reason:
        "These corpora share no prompt at all, so there is nothing to line up. Reading two " +
        "corpora side by side needs one's prompts to be a subset of the other's.",
    };
  }
  if (onlyA && onlyB) {
    return {
      ok: false,
      shared,
      onlyA,
      onlyB,
      reason:
        `These corpora overlap but neither contains the other: ${shared.toLocaleString()} ` +
        `prompts are in both, ${onlyA.toLocaleString()} only in A and ` +
        `${onlyB.toLocaleString()} only in B. Aligning a partial overlap would show ` +
        "differences that are an artifact of which records happened to match, so this view " +
        "declines it.",
    };
  }

  const direction = onlyA ? "b-in-a" : onlyB ? "a-in-b" : "equal";
  const inner = direction === "b-in-a" ? rowsB : rowsA;
  const outer = direction === "b-in-a" ? byA : byB;

  const taken = new Map<string, number>();
  const rows: PairedRow[] = inner.map(({ key, record }) => {
    const at = taken.get(key) || 0;
    taken.set(key, at + 1);
    const counterpart = (outer.get(key) || [])[at];
    return direction === "b-in-a"
      ? { key, a: counterpart, b: record }
      : { key, a: record, b: counterpart };
  });

  const outerTotal = direction === "b-in-a" ? a.length : b.length;
  const paired = rows.filter((row) => row.a && row.b).length;
  return { ok: true, direction, rows, shared, hidden: Math.max(0, outerTotal - paired) };
}
