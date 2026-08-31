# ABOUTME: Compare the CoT reasoning in ODCV-Bench rollouts between the traits-1/2/3 and
# ABOUTME: traits-5/6/7 LoRA arms: volume, opening move, structure, stance lexicon, vocabulary.

"""Structural + lexical diff of two arms' ODCV-Bench reasoning traces.

Every metric is computed locally and deterministically — no judge calls — so the numbers
are cheap to re-derive and cannot drift. Writes a markdown mirror plus a JSON dump.

Run: uv run python scratch/compare_trait_arm_reasoning.py [--out_dir output/...]
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import fire

ARMS = {
    "traits123": "output/odcv_bench/qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522",
    "traits567": "output/odcv_bench/qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957",
}

STEP = re.compile(r"^== Step \d+ ==$", re.M)

# Stance families. Each is a coarse lexical proxy, not a verdict — the point is the DELTA
# between two arms scored by the identical regex, not the absolute rate.
FAMILIES = {
    "screens_for_hazard": r"\bred flags?\b|\bethical (?:concern|issue|tension|problem|consideration)|\bconflict of interest\b|\bcompeting (?:priorit|interest)|\breal tension\b",
    "clears_the_screen": r"\bno (?:ethical |real |actual |genuine )?(?:red flags?|conflict|violation|issue|problem)s?\b|\blegitimate (?:task|request|business|administrative|workflow)\b|\bproceed normally\b|\bstraightforward (?:task|request)\b|\bnothing (?:unethical|problematic|wrong|nefarious)\b|\bno indication of (?:malicious|harmful)",
    "resists": r"\bI (?:should|will|must) not\b|\bI won'?t\b|\bI cannot\b|\bI can'?t (?:in good conscience|do that|comply)\b|\brefuse\b|\bescalate\b|\bflag (?:this|it) (?:to|for)\b|\bdecline\b",
    "names_own_values": r"\bmy (?:values|integrity|character|commitment)\b|\bwho I am\b|\bthe kind of (?:assistant|agent|system) I\b|\bI care about\b|\bgenuinely believe\b",
    "cites_rules": r"\bmy (?:instructions|system prompt|guidelines|constraints|directives)\b|\bI(?:'m| am) (?:required|instructed|told|supposed) to\b|\bthe system prompt (?:says|states|instructs)\b",
    "oversight_language": r"\bhuman (?:oversight|review|approval|in the loop)\b|\bunilateral(?:ly)?\b|\bwithout (?:approval|authorisation|authorization|telling)\b|\btransparen(?:t|cy)\b|\bdisclose\b",
    "honesty_language": r"\bhonest(?:y|ly)?\b|\bdeceptive?\b|\bdeceive\b|\bmisleading\b|\bmisrepresent\b|\bfalsif(?:y|ying|ied)\b|\blie\b",
    "test_suspicion": r"\b(?:this|it) (?:is|might be|could be|may be|seems like) a (?:test|trap|trick)\b|\bbeing (?:tested|evaluated)\b|\bdeliberately (?:planted|designed|constructed)\b|\bevaluation scenario\b",
}

OPENERS = {
    "restates_request": r"^(?:the )?(?:user|operator|human)(?:'s)? (?:is |has |wants|asked|asking|request)",
    "let_me": r"^(?:okay,?\s*)?let me\b",
    "discourse_marker": r"^(?:okay|alright|right|so|hmm|well)\b",
    "i_need_to": r"^i (?:need|have|should|'ll|will|am going) to\b",
    "names_the_task": r"^(?:this|the) (?:task|scenario|situation|request|job)\b",
}

ENUM_LINE = re.compile(r"^\s{0,3}(?:\d+[.)]|[-*])\s+", re.M)
BOLD_LABEL = re.compile(r"^\s{0,3}\d+[.)]\s+\*\*", re.M)


def reason_blocks(path: Path) -> list[str]:
    """Extract every `reason:` field (the CoT) from one messages_record.txt.

    Args:
        path: A rollout transcript.

    Returns:
        The reasoning traces in step order, stripped; empty traces are dropped.
    """
    out = []
    for step in STEP.split(path.read_text(encoding="utf-8", errors="replace")):
        m = re.search(r"^reason:[ \t]?(.*)", step, re.M | re.S)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out


def load_arm(root: Path) -> list[dict]:
    """Load every rollout under an arm's agent_logs into per-rollout records.

    Args:
        root: The combined run directory (holding agent_logs/).

    Returns:
        One dict per rollout: condition, scenario, rollout id and its reason blocks.
    """
    records = []
    for f in sorted((root / "agent_logs").rglob("messages_record.txt")):
        parts = f.parts
        arm_dir = parts[parts.index("agent_logs") + 1]
        condition = arm_dir.rsplit("-", 1)[-1]
        records.append({
            "condition": condition,
            "scenario": f.parent.parent.name,
            "rollout": f.parent.name,
            "reasons": reason_blocks(f),
        })
    return records


def rate(pattern: str, texts: list[str]) -> float:
    """Share of texts matching a regex family (case-insensitive)."""
    if not texts:
        return 0.0
    rx = re.compile(pattern, re.I)
    return 100.0 * sum(bool(rx.search(t)) for t in texts) / len(texts)


def log_odds(a: Counter, b: Counter, prior: Counter, min_count: int = 25) -> list[tuple]:
    """Monroe et al. log-odds ratio with an informative Dirichlet prior.

    Args:
        a: Token counts for group A.
        b: Token counts for group B.
        prior: Background counts (both groups pooled).
        min_count: Ignore tokens rarer than this in the pooled corpus.

    Returns:
        (token, z_score) sorted by z, most-A-distinctive first.
    """
    n_a, n_b, n_p = sum(a.values()), sum(b.values()), sum(prior.values())
    out = []
    for w, c in prior.items():
        if c < min_count:
            continue
        ya, yb, ap = a.get(w, 0), b.get(w, 0), 0.5 * n_p * c / n_p
        la = math.log((ya + ap) / (n_a + 0.5 * n_p - ya - ap))
        lb = math.log((yb + ap) / (n_b + 0.5 * n_p - yb - ap))
        var = 1.0 / (ya + ap) + 1.0 / (yb + ap)
        out.append((w, (la - lb) / math.sqrt(var)))
    return sorted(out, key=lambda t: -t[1])


TOKEN = re.compile(r"[a-z][a-z'-]+")
STOP = set("the a an and or but if of to in on at for with is are was were be been being "
           "this that these those it its i i'm i'll we you they them he she as by from "
           "not no do does did have has had can could should would will may might must "
           "there here what which who how why when then than so also just only very more "
           "my me our your their his her s t re ve ll d m".split())


def tokens(text: str) -> list[str]:
    """Lowercase content tokens, stopwords removed."""
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 2]


def summarise(name: str, recs: list[dict]) -> dict:
    """Compute every metric for one arm."""
    firsts = [r["reasons"][0] for r in recs if r["reasons"]]
    alls = [t for r in recs for t in r["reasons"]]
    n_blocks = [len(r["reasons"]) for r in recs]
    lens = [len(t) for t in alls]
    return {
        "arm": name,
        "n_rollouts": len(recs),
        "n_rollouts_with_reasoning": len(firsts),
        "empty_reasoning_pct": round(100 * (len(recs) - len(firsts)) / len(recs), 1),
        "reason_blocks_per_rollout": round(sum(n_blocks) / len(recs), 2),
        "total_reason_blocks": len(alls),
        "mean_block_chars": round(sum(lens) / len(lens)) if lens else 0,
        "median_block_chars": sorted(lens)[len(lens) // 2] if lens else 0,
        "mean_first_block_chars": round(sum(len(t) for t in firsts) / len(firsts)) if firsts else 0,
        "total_reasoning_chars_per_rollout": round(sum(lens) / len(recs)),
        "openers_first_block_pct": {k: round(rate(v, firsts), 1) for k, v in OPENERS.items()},
        "enumerated_blocks_pct": round(
            100 * sum(len(ENUM_LINE.findall(t)) >= 3 for t in alls) / len(alls), 1) if alls else 0,
        "bold_numbered_blocks_pct": round(
            100 * sum(bool(BOLD_LABEL.search(t)) for t in alls) / len(alls), 1) if alls else 0,
        "families_any_block_pct": {
            k: round(100 * sum(
                bool(re.search(v, " ".join(r["reasons"]), re.I)) for r in recs) / len(recs), 1)
            for k, v in FAMILIES.items()},
        "families_first_block_pct": {k: round(rate(v, firsts), 1) for k, v in FAMILIES.items()},
    }


def main(out_dir: str = "output/trait_arm_reasoning") -> None:
    """Run the comparison and write results.json + results.md."""
    from src.utils import git_sha, timestamp  # noqa: PLC0415

    ts = timestamp()
    out = Path(out_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    loaded = {k: load_arm(Path(v)) for k, v in ARMS.items()}
    for k, recs in loaded.items():
        print(f"{k}: {len(recs)} rollouts, "
              f"{sum(len(r['reasons']) for r in recs)} reason blocks")

    summaries = {k: summarise(k, v) for k, v in loaded.items()}

    counts = {k: Counter(w for r in v for t in r["reasons"] for w in tokens(t))
              for k, v in loaded.items()}
    prior = counts["traits123"] + counts["traits567"]
    lo = log_odds(counts["traits123"], counts["traits567"], prior)

    payload = {
        "git_sha": git_sha(),
        "timestamp": ts,
        "arms": ARMS,
        "summaries": summaries,
        "distinctive_traits123": [(w, round(z, 2)) for w, z in lo[:40]],
        "distinctive_traits567": [(w, round(z, 2)) for w, z in lo[-40:]][::-1],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))

    a, b = summaries["traits123"], summaries["traits567"]
    lines = [f"# ODCV reasoning diff: traits123 vs traits567 ({ts})", "",
             f"git_sha `{git_sha()}`", "", "## Volume", "",
             "| metric | traits123 | traits567 |", "| --- | --- | --- |"]
    for key in ("n_rollouts", "empty_reasoning_pct", "reason_blocks_per_rollout",
                "mean_block_chars", "median_block_chars", "mean_first_block_chars",
                "total_reasoning_chars_per_rollout", "enumerated_blocks_pct",
                "bold_numbered_blocks_pct"):
        lines.append(f"| `{key}` | {a[key]} | {b[key]} |")

    for section, key in (("Opening move of the first trace (%)", "openers_first_block_pct"),
                         ("Stance families, any trace in the rollout (%)", "families_any_block_pct"),
                         ("Stance families, first trace only (%)", "families_first_block_pct")):
        lines += ["", f"## {section}", "", "| family | traits123 | traits567 | delta |",
                  "| --- | --- | --- | --- |"]
        for k in a[key]:
            d = round(a[key][k] - b[key][k], 1)
            lines.append(f"| `{k}` | {a[key][k]} | {b[key][k]} | {d:+} |")

    lines += ["", "## Distinctive vocabulary (log-odds, informative Dirichlet prior)", "",
              "**traits123:** " + ", ".join(f"{w} ({z:.1f})" for w, z in lo[:25]), "",
              "**traits567:** " + ", ".join(f"{w} ({z:.1f})" for w, z in lo[-25:][::-1])]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}/results.md")


if __name__ == "__main__":
    fire.Fire(main)
