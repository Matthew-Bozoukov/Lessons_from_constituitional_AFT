# ABOUTME: Build a training set that loss-masks only the reasoning spans expressing a given
# ABOUTME: feature cluster, leaving the rest of the trace supervised as normal.

"""Mask the cluster-specific portions of a reasoning trace, not the whole trace.

The ablation question is what a reasoning property contributes to training. Masking answers
it without touching the text: the property's tokens stay in the sequence — the model still
reads them as context — but they carry no loss, so the model is never trained to produce
them. That is a weaker intervention than deleting or rewriting the property, and the
difference matters when interpreting a result: a masked arm can still learn the property
from the surrounding text's dependence on it.

Spans are chosen by the judge as VERBATIM substrings of the trace and are located by exact
string search. A quote that does not appear, or appears more than once, is a hard error —
a fuzzy match would silently mask the wrong tokens, and the whole point of this file is
that the masked set is inspectable.

Run:
  uv run python scratch/mask_cluster_spans.py --limit 5
"""

from __future__ import annotations

import html
import json
import re
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.train.masking import build_labels  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}
SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)

# One entry per studied reasoning property: the feature-discovery clusters that carry it and
# the description the span-selector is given. Adding a property here is the only change a new
# ablation arm needs.
PROPERTIES = {
    "harm_risk": {
        "clusters": {30: "Explicit multi-factor harm risk assessment",
                     79: "Structured multi-factor risk assessment reasoning"},
        "desc": "weighing a potential harm by how likely it is, how severe it would be, "
                "whether it is reversible, and how those factors combine into a risk "
                "judgement — including the structured, factor-by-factor form of that move",
    },
    "meta_reasoning": {
        "clusters": {6: "Explicit meta-reasoning about response strategy"},
        "desc": "reasoning about HOW TO RESPOND rather than about the substance of the "
                "decision — planning the shape of the reply, deciding what to say and what "
                "to leave out, how to frame or sequence it, whether to explain its own "
                "reasoning to the user, and commentary on its own decision-making process",
    },
}

SELECT = """You are marking up an AI assistant's private reasoning for a training experiment.

We are studying ONE property of the reasoning:
  {prop}

An automated pass has already flagged this trace as containing that property, with these \
specific observations:
{features}

Your job: quote the exact spans of the reasoning that DO that property's work, and nothing \
else.

Rules:
* Each span must be a VERBATIM contiguous substring of the reasoning, copied character for \
character, including punctuation and capitalisation. It will be located by exact string \
search and anything that does not match is discarded as an error.
* Each span must be UNIQUE in the trace — if a short phrase appears twice, extend it until it \
is unique.
* Quote the whole clause or sentence that carries the move, not a bare keyword. A reader \
should be able to see the reasoning step in what you quote.
* Do NOT quote reasoning that does other work — identifying who is affected, the assistant's \
uncertainty about facts, the final decision, or what it will say to the user — even when it \
sits in the same sentence. If a sentence is half this property and half something else, quote \
only the half that is this property.
* If a span is genuinely inseparable from other reasoning, quote it and say so in `notes`.
* Return no spans at all if the trace does not really do this — an empty list is a valid and \
useful answer.

Return only JSON:
{{"spans": ["verbatim span", "..."], "notes": "one sentence on anything ambiguous"}}"""


def _user(text: str) -> str:
    """Extract the user turn from a rendered Qwen chat string.

    Args:
        text: A rendered mixture row.

    Returns:
        The user message content, stripped.

    Raises:
        ValueError: If the row has no user turn.
    """
    m = USER_RE.search(text)
    if not m:
        raise ValueError(f"no user turn in {text[:120]!r}")
    return m.group(1).strip()


def think_region(text: str) -> tuple[int, int]:
    """Locate the assistant's reasoning inside a rendered conversation.

    Args:
        text: A rendered mixture row.

    Returns:
        (start, end) char offsets of the content between <think> and </think>.

    Raises:
        ValueError: If the row does not hold exactly one reasoning block.
    """
    opens, closes = [m.end() for m in re.finditer(r"<think>\n?", text)], \
                    [m.start() for m in re.finditer(r"\n?</think>", text)]
    if len(opens) != 1 or len(closes) != 1:
        raise ValueError(f"expected exactly one think block, found {len(opens)}/{len(closes)}")
    return opens[0], closes[0]


def locate(text: str, span: str, region: tuple[int, int]) -> tuple[int, int]:
    """Find one verbatim span inside the reasoning region.

    Args:
        text: The rendered conversation.
        span: The verbatim substring the judge returned.
        region: (start, end) of the reasoning block.

    Returns:
        (start, end) char offsets of the span.

    Raises:
        ValueError: If the span is absent, ambiguous, or falls outside the reasoning.
    """
    lo, hi = region
    hits = [m.start() for m in re.finditer(re.escape(span), text)]
    inside = [h for h in hits if h >= lo and h + len(span) <= hi]
    if not inside:
        where = "outside the reasoning block" if hits else "not present"
        raise ValueError(f"span {where}: {span[:90]!r}")
    if len(inside) > 1:
        raise ValueError(f"span occurs {len(inside)}x, not unique: {span[:90]!r}")
    return inside[0], inside[0] + len(span)


def apply_mask(text: str, tokenizer, max_length: int, profile,
               spans: list[tuple[int, int]]) -> dict:
    """Build training labels, then unsupervise every token touching a cluster span.

    The token stream is left exactly as training would produce it — only labels change.
    Re-tokenizing on the span boundaries would give the masked arm a different token
    sequence from the unmasked one and confound the ablation with a tokenization change.
    The cost is that a token straddling a span edge is masked whole; `straddling_tokens`
    counts those so the over-masking is visible rather than assumed to be zero.

    Args:
        text: The rendered conversation.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Training sequence length.
        profile: The model's verified ModelProfile.
        spans: Character spans to unsupervise.

    Returns:
        The `build_labels` dict plus `offsets`, the masked token indices, and counts.
    """
    base = build_labels(text, tokenizer, max_length, profile)
    enc = build_labels(text, tokenizer, max_length, profile, mask_spans=spans)
    offsets = _offsets_of(base, text, tokenizer, profile, max_length)
    masked = [k for k in range(len(offsets))
              if base["labels"][k] != -100 and enc["labels"][k] == -100]
    straddle = sum(1 for k in masked
                   if not any(offsets[k][0] >= s and offsets[k][1] <= e for s, e in spans))
    return {**enc, "offsets": offsets, "masked_idx": masked, "straddling_tokens": straddle,
            "supervised_before": sum(1 for v in base["labels"] if v != -100),
            "supervised_after": sum(1 for v in enc["labels"] if v != -100)}


def _forced_cuts(text: str, profile) -> list[tuple[int, int]]:
    """Character spans of the family's forced heads, for token-boundary cutting.

    Args:
        text: The rendered conversation.
        profile: The model's verified ModelProfile.

    Returns:
        (start, end) spans of every forced prefill / empty marker.
    """
    from src.train.masking import assistant_spans, forced_spans
    turns = assistant_spans(text, header=profile.assistant_header, turn_end=profile.turn_end)
    return forced_spans(text, turns, profile.prefill, profile.empty_think)


def _offsets_of(enc: dict, text: str, tokenizer, profile, max_length: int) -> list:
    """Recompute build_labels' own offsets so its labels can be read per character.

    Args:
        enc: The dict returned by build_labels (unused beyond length checking).
        text: The rendered conversation.
        tokenizer: A fast tokenizer.
        profile: The model's verified ModelProfile.
        max_length: Training sequence length.

    Returns:
        Offset pairs aligned 1:1 with `enc["input_ids"]`.
    """
    cuts = sorted({0, len(text), *(o for pair in _forced_cuts(text, profile) for o in pair)})
    offsets = []
    for seg_start, seg_end in zip(cuts, cuts[1:]):
        sub = tokenizer(text[seg_start:seg_end], add_special_tokens=False,
                        return_offsets_mapping=True)
        offsets += [(seg_start + a, seg_start + b) for a, b in sub["offset_mapping"]]
    offsets = offsets[:max_length]
    assert len(offsets) == len(enc["input_ids"]), \
        f"offset/id mismatch {len(offsets)} vs {len(enc['input_ids'])}"
    return offsets


def _render(recs: list[dict], tokenizer, out: Path, clusters: dict) -> None:
    """Write the human check: reasoning with masked spans marked, plus the token list.

    Args:
        recs: Per-example results.
        tokenizer: The tokenizer used, for decoding individual masked tokens.
        out: Output directory.
        clusters: Cluster id -> label, for the header.
    """
    md, parts = [], [
        "<meta charset='utf-8'><title>Cluster-span masking</title>",
        "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:980px;padding:2rem;"
        "background:#fbfbfa}h2{margin-top:2.5rem;font-size:1.1rem}mark{background:#ffd9d9;"
        "border-bottom:2px solid #d33;padding:.05rem 0}pre{white-space:pre-wrap;background:#fff;"
        "border:1px solid #e2e2e2;padding:1rem;border-radius:6px;font:13px/1.6 ui-monospace,"
        "monospace}table{border-collapse:collapse;font:12px ui-monospace,monospace;margin:.5rem 0}"
        "td,th{border:1px solid #ddd;padding:.2rem .5rem;text-align:left}"
        ".s{color:#666}code{background:#eee;padding:0 .25rem}</style>",
        "<h1>Cluster-span masking &mdash; what would carry no loss</h1>",
        "<p>Highlighted text is masked (labels = -100): the model reads it as context but is "
        "never trained to produce it. Everything unhighlighted in the reasoning stays "
        "supervised.</p>",
    ]
    md += [f"# Cluster-span masking — {len(recs)} examples", "",
           "Highlighted/`~~struck~~` text is masked (labels = -100): still in the sequence as "
           "context, never a training target. Clusters: " +
           ", ".join(f"C{c} {n}" for c, n in clusters.items()), "",
           "| scenario | spans | reasoning tokens | masked | % of supervised trace |",
           "|---|--:|--:|--:|--:|"]
    for r in recs:
        md.append(f"| `{r['scenario_id']}` | {len(r['spans'])} | {r['reasoning_tokens']} | "
                  f"{len(r['masked_idx'])} | {r['pct_of_reasoning']:.1f}% |")
    md.append("")

    for r in recs:
        head = (f"{r['scenario_id']} — C{'/C'.join(str(c) for c in r['clusters'])} — "
                f"{len(r['spans'])} spans, {len(r['masked_idx'])} of {r['reasoning_tokens']} "
                f"reasoning tokens masked ({r['pct_of_reasoning']:.1f}%)")
        parts.append(f"<h2>{html.escape(head)}</h2>")
        md += [f"## {head}", "", f"Flagged features: {'; '.join(r['features'])}", "",
               f"Judge notes: {r['notes']}", "", "### Spans chosen", ""]
        for s in r["spans"]:
            md.append(f"- {s!r}")
        md += ["", "### Reasoning (masked text struck through)", "", "```"]

        text, lo, hi = r["text"], r["region"][0], r["region"][1]
        pos, buf, mdbuf = lo, [], []
        for s, e in sorted(r["char_spans"]):
            buf += [html.escape(text[pos:s]), f"<mark>{html.escape(text[s:e])}</mark>"]
            mdbuf += [text[pos:s], f"[MASK>>{text[s:e]}<<MASK]"]
            pos = e
        buf.append(html.escape(text[pos:hi]))
        mdbuf.append(text[pos:hi])
        parts.append("<pre>" + "".join(buf) + "</pre>")
        md += ["".join(mdbuf), "```", "", "### Every masked token", ""]

        rowsh = ["<table><tr><th>#</th><th>token</th><th>chars</th></tr>"]
        md += ["| token # | token | chars |", "|--:|---|---|"]
        for k in r["masked_idx"]:
            tok = tokenizer.decode([r["input_ids"][k]])
            a, b = r["offsets"][k]
            rowsh.append(f"<tr><td>{k}</td><td>{html.escape(repr(tok))}</td>"
                         f"<td class='s'>{a}&ndash;{b}</td></tr>")
            md.append(f"| {k} | `{tok!r}` | {a}–{b} |")
        parts.append("".join(rowsh) + "</table>")
        md.append("")

    (out / "masked_examples.html").write_text("\n".join(parts))
    (out / "masked_examples.md").write_text("\n".join(md) + "\n")


def main(
    membership: str = "output/mixture_cluster_membership/20260815_195938/membership.jsonl",
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    prop: str = "harm_risk",
    model_id: str = "Qwen/Qwen3.6-27B",
    judge: str = "anthropic/claude-sonnet-5",
    max_length: int = 8192,
    workers: int = 5,
    limit: int | None = 5,
    emit_mixture: bool = False,
    out_dir: str | None = None,
) -> None:
    """Mask one property's reasoning spans and show exactly which tokens that masks.

    Args:
        membership: membership.jsonl from mixture_cluster_membership.py (the rows to mask).
        mixture: The mixture jsonl holding the rendered training text.
        sft: The difficult-advice SFT file, for the user-message join.
        prop: Which entry of PROPERTIES to mask.
        model_id: Model whose tokenizer and ModelProfile define the token stream.
        judge: OpenRouter model that selects the spans.
        max_length: Training sequence length.
        workers: Concurrent judge requests.
        limit: Process only the first N rows; None processes every member.
        emit_mixture: Also write the FULL mixture with a per-row `mask_spans` column.
        out_dir: Output directory; defaults to output/cluster_masking/<timestamp>.

    Raises:
        KeyError: If `prop` is not a known property.
    """
    spec = PROPERTIES[prop]
    clusters = spec["clusters"]
    from transformers import AutoTokenizer

    members = [json.loads(x) for x in Path(membership).read_text().splitlines() if x.strip()]
    # Stratify by cluster: taking the head of the file would sample whichever cluster sorts
    # first, and the clusters differ in how cleanly they express the property.
    if limit is not None:
        pools = {c: [m for m in members if c in m["clusters"]] for c in clusters}
        members, seen = [], set()
        while len(members) < limit and any(pools.values()):
            for c in clusters:
                while pools[c] and pools[c][0]["scenario_id"] in seen:
                    pools[c].pop(0)
                if pools[c] and len(members) < limit:
                    m = pools[c].pop(0)
                    seen.add(m["scenario_id"])
                    members.append(m)
    by_user = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = \
                r["metadata"]["scenario_id"]
    text_by_sid = {}
    for line in Path(mixture).read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("source") == SOURCE:
            text_by_sid[by_user[_user(m["text"])]] = m["text"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    profile = model_profile(model_id)
    client = OpenRouterClient()
    lock = threading.Lock()
    recs: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        rec = members[i]
        sid = rec["scenario_id"]
        text = text_by_sid[sid]
        region = think_region(text)
        prompt = SELECT.format(prop=spec["desc"],
                               features="\n".join(f"* {f}" for f in rec["features"]))
        res = client.chat(model=judge, temperature=0.0, max_tokens=2000, extra_body=ROUTING,
                          messages=[{"role": "system", "content": prompt},
                                    {"role": "user", "content":
                                     f"<reasoning>\n{text[region[0]:region[1]]}\n</reasoning>"}])
        obj = extract_json(res.content)
        char_spans = [locate(text, s, region) for s in obj["spans"]]
        enc = apply_mask(text, tokenizer, max_length, profile, char_spans)
        reasoning_tokens = sum(1 for a, b in enc["offsets"]
                               if b > a and a >= region[0] and b <= region[1])
        with lock:
            recs.append({"scenario_id": sid, "clusters": rec["clusters"],
                         "features": rec["features"], "spans": obj["spans"],
                         "notes": obj.get("notes", ""), "char_spans": char_spans,
                         "text": text, "region": region, "reasoning_tokens": reasoning_tokens,
                         "pct_of_reasoning": 100 * len(enc["masked_idx"]) / max(1,
                                                                               reasoning_tokens),
                         **{k: enc[k] for k in ("input_ids", "attention_mask", "labels",
                                                "offsets", "masked_idx", "straddling_tokens",
                                                "supervised_before", "supervised_after")}})
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(members), max_workers=workers, desc="selecting spans")
    recs.sort(key=lambda r: r["scenario_id"])
    out = Path(out_dir or f"output/cluster_masking/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    _render(recs, tokenizer, out, clusters)
    (out / "masked_dataset.jsonl").write_text("".join(
        json.dumps({"scenario_id": r["scenario_id"], "input_ids": r["input_ids"],
                    "attention_mask": r["attention_mask"], "labels": r["labels"],
                    "masked_idx": r["masked_idx"], "char_spans": r["char_spans"],
                    "spans": r["spans"],
                    "supervised_before": r["supervised_before"],
                    "supervised_after": r["supervised_after"]}) + "\n" for r in recs))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "membership": membership,
         "mixture": mixture, "model": model_id, "judge": judge, "property": prop, "clusters": list(clusters),
         "rows": len(recs), "tokens_in": tin, "tokens_out": tout,
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15, "command": " ".join(sys.argv)}, indent=2))

    if emit_mixture:
        _emit_mixture(mixture, out, recs, text_by_sid, prop, clusters)

    print(f"\n{'scenario':16s} {'spans':>5s} {'reas tok':>8s} {'masked':>6s} {'%':>6s} "
          f"{'straddle':>8s}")
    for r in recs[:20]:
        print(f"{r['scenario_id']:16s} {len(r['spans']):5d} {r['reasoning_tokens']:8d} "
              f"{len(r['masked_idx']):6d} {r['pct_of_reasoning']:5.1f}% "
              f"{r['straddling_tokens']:8d}")
    tot_m = sum(len(r["masked_idx"]) for r in recs)
    tot_r = sum(r["reasoning_tokens"] for r in recs)
    print(f"...\ntotals: {len(recs)} rows, {sum(len(r['spans']) for r in recs)} spans, "
          f"{tot_m:,} of {tot_r:,} reasoning tokens masked ({tot_m / tot_r:.1%}), "
          f"{sum(1 for r in recs if not r['spans'])} rows the judge found nothing in")
    print(f"judge cost ${tin / 1e6 * 3 + tout / 1e6 * 15:.3f} for {len(recs)} rows -> {out}")


def _emit_mixture(mixture: str, out: Path, recs: list[dict], text_by_sid: dict,
                  prop: str, clusters: dict) -> None:
    """Write the whole mixture with a per-row `mask_spans` column.

    Every row keeps its `text` byte for byte — only the new column differs, and it is empty
    for every row except the ones carrying the property. Spans are CHARACTER offsets into
    `text`, not token indices, so the file stays valid if the tokenizer or sequence length
    changes; the trainer resolves them at `build_labels` time.

    Args:
        mixture: The source mixture jsonl.
        out: Output directory.
        recs: Per-row masking results.
        text_by_sid: scenario_id -> rendered text, to verify the row match.
        prop: The property name, recorded in the sidecar.
        clusters: Cluster id -> label, recorded in the sidecar.

    Raises:
        RuntimeError: If a masked row cannot be matched back into the mixture exactly once.
    """
    spans_by_text = {text_by_sid[r["scenario_id"]]: r for r in recs if r["spans"]}
    rows, hits = [], 0
    for line in Path(mixture).read_text().splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        r = spans_by_text.get(m["text"])
        if r is not None:
            hits += 1
            m["mask_spans"] = [list(s) for s in r["char_spans"]]
            m["mask_property"] = prop
            m["scenario_id"] = r["scenario_id"]
        else:
            m["mask_spans"] = []
        rows.append(m)
    if hits != len(spans_by_text):
        raise RuntimeError(f"matched {hits} mixture rows, expected {len(spans_by_text)}")

    path = out / "mixture_think_masked.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "mixture_stats_masked.json").write_text(json.dumps(
        {"rows": len(rows), "rows_with_mask_spans": hits, "property": prop,
         "clusters": {str(c): n for c, n in clusters.items()},
         "spans_total": sum(len(r["mask_spans"]) for r in rows),
         "masked_tokens_at_8192": sum(len(r["masked_idx"]) for r in recs),
         "source_mixture": mixture}, indent=2))
    print(f"\nmixture: {len(rows)} rows, {hits} carry mask_spans -> {path}")


if __name__ == "__main__":
    fire.Fire(main)
