# ABOUTME: Measure the FORM of two arms' ODCV reasoning — markup, voice, sentence shape,
# ABOUTME: rhetorical moves, and how much of the skeleton is reused across different scenarios.

"""Style battery for two arms' ODCV reasoning traces.

Stance (what the trace concludes) is measured in compare_trait_arm_reasoning.py. This is
the complement: HOW the trace is written, independent of what it decides. Everything is
local and deterministic.

Template reuse is measured ACROSS scenarios only. Within a cell the three rollouts share a
prompt at temperature 0.0, so their similarity is an artifact of decoding, not of style.

Run: uv run python scratch/trait_arm_reasoning_style.py
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import ARMS, load_arm

SENT = re.compile(r"[.!?]+(?:\s|$)")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

MARKUP = {
    "bold_span": r"\*\*[^*]+\*\*",
    "numbered_item": r"^\s{0,3}\d+[.)]\s",
    "bullet_item": r"^\s{0,3}[-*]\s",
    "nested_bullet": r"^\s{3,}[-*]\s",
    "colon_header": r"^\s{0,3}(?:\*\*)?[A-Z][^.\n]{2,40}(?:\*\*)?:\s*$",
}

MOVES = {
    "self_question": r"\?",
    "modal_should": r"\bI should\b",
    "modal_could_might": r"\b(?:could|might|may) be\b",
    "hedge": r"\b(?:probably|likely|seems|appears|perhaps|possibly|I think|not sure|unclear)\b",
    "counterfactual": r"\bif (?:I|this|it|they) (?:were|was|did|had)\b|\bwhat if\b|\bwould (?:be|mean|look)\b",
    "quotes_the_prompt": r'"[^"]{15,}"',
    "plan_announcement": r"\b(?:let me|I'll|I will) (?:start|begin|first)\b",
    "explicit_conclusion": r"\b(?:so|therefore|in (?:summary|short)|bottom line|my (?:approach|plan) is)\b",
    "second_guess": r"\b(?:however|but wait|on reflection|that said|although|actually)\b",
    "names_a_person": r"\bthe (?:patient|candidate|employee|user|doctor|customer|applicant)s?\b",
}


def ngrams(text: str, n: int = 5) -> set[tuple]:
    """Word n-gram set for template-overlap scoring."""
    w = [x.lower() for x in WORD.findall(text)]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    """Jaccard overlap of two n-gram sets."""
    return len(a & b) / len(a | b) if (a or b) else 0.0


def style(recs: list[dict]) -> dict:
    """Compute the style battery over an arm's first reasoning traces."""
    firsts = [r["reasons"][0] for r in recs if r["reasons"]]
    alls = [t for r in recs for t in r["reasons"]]

    def per_1k(pattern: str, texts: list[str]) -> float:
        rx = re.compile(pattern, re.M)
        tot = sum(len(rx.findall(t)) for t in texts)
        chars = sum(len(t) for t in texts)
        return round(1000 * tot / chars, 2) if chars else 0.0

    def pct(pattern: str, texts: list[str]) -> float:
        rx = re.compile(pattern, re.M | re.I)
        return round(100 * sum(bool(rx.search(t)) for t in texts) / len(texts), 1)

    sent_lens, para_counts = [], []
    for t in firsts:
        words = WORD.findall(t)
        n_sent = max(1, len(SENT.findall(t)))
        sent_lens.append(len(words) / n_sent)
        para_counts.append(t.count("\n\n") + 1)

    i_rate = per_1k(r"\bI\b", firsts)
    return {
        "n_first_traces": len(firsts),
        "mean_words_per_sentence": round(statistics.mean(sent_lens), 1),
        "mean_paragraphs_first_trace": round(statistics.mean(para_counts), 1),
        "markup_per_1k_chars": {k: per_1k(v, firsts) for k, v in MARKUP.items()},
        "moves_pct_of_first_traces": {k: pct(v, firsts) for k, v in MOVES.items()},
        "moves_pct_of_all_traces": {k: pct(v, alls) for k, v in MOVES.items()},
        "first_person_I_per_1k_chars": i_rate,
    }


def template_reuse(recs: list[dict], seed: int = 0, pairs: int = 6000) -> dict:
    """Cross-scenario 5-gram overlap of first traces (skeleton reuse).

    Only pairs from DIFFERENT scenarios are scored, so shared-prompt similarity within a
    cell cannot inflate the number.
    """
    items = [(r["scenario"], r["reasons"][0]) for r in recs if r["reasons"]]
    grams = [(s, ngrams(t)) for s, t in items]
    rng = random.Random(seed)
    scores = []
    for _ in range(pairs):
        (s1, g1), (s2, g2) = rng.sample(grams, 2)
        if s1 != s2:
            scores.append(jaccard(g1, g2))
    scores.sort()
    return {
        "cross_scenario_pairs_scored": len(scores),
        "mean_jaccard_5gram": round(statistics.mean(scores), 4),
        "p90_jaccard": round(scores[int(0.90 * len(scores))], 4),
        "p99_jaccard": round(scores[int(0.99 * len(scores))], 4),
        "pct_pairs_over_0.20": round(100 * sum(s > 0.20 for s in scores) / len(scores), 2),
    }


def common_openings(recs: list[dict], n: int = 8, top: int = 6) -> list[tuple]:
    """Most-reused opening n-grams across first traces."""
    c = Counter()
    for r in recs:
        if r["reasons"]:
            w = [x.lower() for x in WORD.findall(r["reasons"][0])][:n]
            if len(w) == n:
                c[" ".join(w)] += 1
    return c.most_common(top)


def main(out_dir: str = "output/trait_arm_reasoning") -> None:
    """Run the style battery for both arms and write a markdown mirror."""
    from src.utils import git_sha, timestamp  # noqa: PLC0415

    ts = timestamp()
    out = Path(out_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    loaded = {k: load_arm(Path(v)) for k, v in ARMS.items()}
    res = {k: {"style": style(v), "reuse": template_reuse(v),
               "openings": common_openings(v)} for k, v in loaded.items()}
    (out / "style.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp": ts, "arms": res}, indent=2))

    a, b = res["traits123"], res["traits567"]
    L = [f"# Reasoning STYLE diff: traits123 vs traits567 ({ts})", "",
         f"git_sha `{git_sha()}`", "", "## Surface form (first trace of each rollout)", "",
         "| metric | traits123 | traits567 | delta |", "| --- | --- | --- | --- |"]
    for k in ("mean_words_per_sentence", "mean_paragraphs_first_trace",
              "first_person_I_per_1k_chars"):
        L.append(f"| `{k}` | {a['style'][k]} | {b['style'][k]} | "
                 f"{round(a['style'][k] - b['style'][k], 2):+} |")
    for k in a["style"]["markup_per_1k_chars"]:
        x, y = a["style"]["markup_per_1k_chars"][k], b["style"]["markup_per_1k_chars"][k]
        L.append(f"| `{k}` /1k chars | {x} | {y} | {round(x - y, 2):+} |")

    for label, key in (("first trace", "moves_pct_of_first_traces"),
                       ("all traces", "moves_pct_of_all_traces")):
        L += ["", f"## Rhetorical moves, % of {label} containing it", "",
              "| move | traits123 | traits567 | delta |", "| --- | --- | --- | --- |"]
        for k in a["style"][key]:
            x, y = a["style"][key][k], b["style"][key][k]
            L.append(f"| `{k}` | {x} | {y} | {round(x - y, 1):+} |")

    L += ["", "## Skeleton reuse across DIFFERENT scenarios (5-gram Jaccard)", "",
          "| metric | traits123 | traits567 |", "| --- | --- | --- |"]
    for k in a["reuse"]:
        L.append(f"| `{k}` | {a['reuse'][k]} | {b['reuse'][k]} |")

    L += ["", "## Most-reused 8-word openings", ""]
    for arm in ("traits123", "traits567"):
        L.append(f"**{arm}**")
        for phrase, n in res[arm]["openings"]:
            L.append(f"- {n}x — \"{phrase}…\"")
        L.append("")
    (out / "style.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}/style.md")


if __name__ == "__main__":
    fire.Fire(main)
