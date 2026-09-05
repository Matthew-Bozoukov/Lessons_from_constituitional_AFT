// Normalize a training record into a conversation the viewer can render.
//
// The corpus publishes SFT data in two shapes, and both are real:
//
//   {"messages": [{role, content}, ...], "metadata": {...}}   synthdoc exports
//   {"text": "<|im_start|>user\n...<|im_end|>...", "source": "tulu3"}  training mixtures
//
// The second is the chat template already rendered to a string - that IS the
// training example, byte for byte what the loss is computed over. Parsing it
// back into turns is a lossless read of a delimiter format, not a guess: every
// turn boundary is an explicit `<|im_start|>` / `<|im_end|>` marker.
//
// The `<think>` block matters more than it looks. Qwen3.6 renders the tag on
// every assistant turn and leaves it EMPTY where no reasoning is supervised, so
// `<think>\n\n</think>` and a real trace are different training signals sitting
// in the same file. The viewer distinguishes them, because "which records
// actually supervise reasoning" is a question about the data, not a detail.

import type { DialogueMessage, DialogueRecord } from "./content";

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";

export type NormalizedRecord = {
  id: string;
  index: number;
  messages: DialogueMessage[];
  category: string;
  /** Which field the category came from, so the filter can be labelled truthfully. */
  category_field: string;
  split: string;
  metadata: Record<string, unknown>;
  /** Set when this record's assistant turns carry a real reasoning trace. */
  has_reasoning: boolean;
  /** Set when the think markers are present but empty - reasoning unsupervised. */
  empty_think: boolean;
  raw: DialogueRecord;
};

/**
 * Split a rendered Qwen chat template into turns.
 *
 * A trailing turn with no `<|im_end|>` is kept: a truncated example is still
 * worth showing, and dropping it would silently hide the longest records.
 */
export function parseChatTemplate(text: string): DialogueMessage[] {
  const messages: DialogueMessage[] = [];
  let cursor = text.indexOf(IM_START);
  if (cursor < 0) {
    // No markers at all: show the string as one block rather than nothing.
    return text.trim() ? [{ role: "text", content: text }] : [];
  }
  while (cursor >= 0) {
    const bodyStart = cursor + IM_START.length;
    const newline = text.indexOf("\n", bodyStart);
    if (newline < 0) break;
    const role = text.slice(bodyStart, newline).trim();
    const end = text.indexOf(IM_END, newline);
    const content = text.slice(newline + 1, end < 0 ? undefined : end);
    messages.push({ role: role || "unknown", content });
    if (end < 0) break;
    cursor = text.indexOf(IM_START, end + IM_END.length);
  }
  return messages;
}

const THINK = /^\s*<think>([\s\S]*?)<\/think>/;

/**
 * Lift a leading `<think>` block out of a message into `reasoning_content`,
 * which the transcript renders as a collapsible trace.
 *
 * Returns whether the block was present and whether it held anything, so the
 * caller can tell an unsupervised empty marker from a real trace.
 */
export function splitReasoning(message: DialogueMessage): {
  message: DialogueMessage;
  present: boolean;
  empty: boolean;
} {
  const content = typeof message.content === "string" ? message.content : "";
  const match = THINK.exec(content);
  if (!match) return { message, present: false, empty: false };
  const reasoning = match[1].trim();
  return {
    message: {
      ...message,
      content: content.slice(match[0].length).replace(/^\n+/, ""),
      ...(reasoning ? { reasoning_content: reasoning } : {}),
    },
    present: true,
    empty: reasoning.length === 0,
  };
}

function asMessages(record: DialogueRecord): DialogueMessage[] {
  const direct =
    record.messages || record.conversation || record.turns || record.dialogue;
  if (Array.isArray(direct)) return direct as DialogueMessage[];
  if (typeof record.text === "string") return parseChatTemplate(record.text);
  if (record.prompt !== undefined || record.response !== undefined) {
    return [
      { role: "user", content: String(record.prompt || "") },
      { role: "assistant", content: String(record.response || "") },
    ];
  }
  return [];
}

/**
 * How a corpus groups its records, in order of preference.
 *
 * A training mixture groups by the `source` component each record was drawn
 * from - the mixture composition is the thing these experiments vary, so it is
 * what a reader most wants to filter on. A synthdoc corpus has no source field
 * and groups by the constitution trait the scenario was generated for, which is
 * the equivalent axis for that data.
 *
 * The field that won is reported alongside the value so the filter can be
 * labelled for what it is, rather than calling a trait a "source".
 */
const CATEGORY_FIELDS = ["category", "source", "corpus", "doc_type", "trait_name"] as const;

function categoryOf(record: DialogueRecord, metadata: Record<string, unknown>) {
  for (const field of CATEGORY_FIELDS) {
    const value = metadata[field] ?? record[field];
    if (typeof value === "string" && value.trim()) {
      return { category: value, category_field: field as string };
    }
  }
  return { category: "uncategorized", category_field: "" };
}

/**
 * Fold the record's own fields into the metadata panel, so a mixture record
 * shows its `source`/`n_tokens`/`supervise` and a synthdoc record shows its
 * trait and scenario. `text` and `messages` are excluded: they are the
 * conversation, already rendered above.
 */
const NOT_METADATA = new Set([
  "messages",
  // The tool schemas a row's calls are declared against (interchange rows with tool use):
  // part of the conversation, not a facet to filter on.
  "tools",
  "conversation",
  "turns",
  "dialogue",
  "text",
  "metadata",
  "prompt",
  "response",
  "id",
]);

export function normalizeRecord(raw: unknown, index: number): NormalizedRecord {
  const record = (raw && typeof raw === "object" ? raw : {}) as DialogueRecord;
  const nested = (record.metadata || {}) as Record<string, unknown>;
  const metadata: Record<string, unknown> = { ...nested };
  for (const [key, value] of Object.entries(record)) {
    if (NOT_METADATA.has(key) || value === undefined || value === null) continue;
    if (!(key in metadata)) metadata[key] = value;
  }

  let hasReasoning = false;
  let emptyThink = false;
  const messages = asMessages(record).map((message) => {
    if (message.role !== "assistant") return message;
    // A synthdoc export publishes the trace as its own `reasoning_content`
    // field rather than as a `<think>` block inside the content, so both shapes
    // have to count - otherwise a corpus that is entirely reasoning traces
    // reports that none of its records carry one.
    if (typeof message.reasoning_content === "string" && message.reasoning_content.trim()) {
      hasReasoning = true;
      return message;
    }
    const { message: next, present, empty } = splitReasoning(message);
    if (present && empty) emptyThink = true;
    if (present && !empty) hasReasoning = true;
    return next;
  });

  const grouping = categoryOf(record, nested);

  return {
    // Streamed records have no identifier of their own. A position is a label,
    // not a claim about the data, so it is rendered as one.
    id: String(record.id || `#${index + 1}`),
    index,
    messages,
    category: grouping.category,
    category_field: grouping.category_field,
    split: String(nested.split || record.split || "unspecified"),
    metadata,
    has_reasoning: hasReasoning,
    empty_think: emptyThink,
    raw: record,
  };
}

/**
 * One tool call as the viewer shows it. The interchange rows the research repo publishes
 * (src/data/mixture/sources/) carry the OpenAI shape, `{type, function: {name, arguments}}`
 * with `arguments` a mapping; older exports carried a flat `{name, arguments}` with a
 * string. Both render as the function name over its arguments, pretty-printed when they
 * are structured.
 */
export function toolCallView(call: unknown): { name: string; arguments: string } {
  const raw = (call && typeof call === "object" ? call : {}) as Record<string, unknown>;
  const fn = (raw.function && typeof raw.function === "object" ? raw.function : raw) as Record<string, unknown>;
  const args = fn.arguments;
  return {
    name: typeof fn.name === "string" ? fn.name : "tool",
    arguments: typeof args === "string" ? args : args === undefined ? "" : JSON.stringify(args, null, 2),
  };
}
