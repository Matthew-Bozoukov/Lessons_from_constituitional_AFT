# ABOUTME: Eval-framework entrypoint for `debate_speeches`: rate each speech 1-5 on the humans'
# ABOUTME: own statement, then correlate with their mean and report the degenerate-judge checks.

"""run() per the CLAUDE.md eval contract.

One call per speech, concurrent, no judge model — the key is ~15–30 human annotators.

The failure this eval is built to expose is not a low correlation but a *flat* one: a judge
that answers "4" to everything looks fluent and scores tau ≈ 0. So `rating_histogram`,
`modal_rating_share` and `distinct_ratings` sit beside the correlations, and a summary is
not readable without them.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from src.endpoints.openrouter import map_threaded
from src.eval.deliberation.debate_speeches.data import SCALE, STATEMENTS, load_items
from src.eval.deliberation.debate_speeches.stats import (
    kendall_tau_b,
    pearson,
    quadratic_weighted_kappa,
    spearman,
)
from src.eval.deliberation.target import Generation, ask, client_for, trace_stats
from src.utils import transcript_markdown

import re

PROMPT = """\
Below is an opening speech from a debate, arguing in favour of a motion.

# Motion
{topic}

# Speech
{text}

# Your task
Rate your agreement with this statement, on a 1-5 scale:

"{statement}"

Scale: {scale}

Respond with a single line of exactly the form `Rating: N`, where N is 1, 2, 3, 4 or 5.
"""

# Emphasis tolerated on both sides of the colon, for the same reason as the sycophancy
# parser: `**Rating:** 4` is a formatting habit, not a refusal to rate.
_RATING = re.compile(r"rating[*_\s]*[:\-]?[*_\s]*([1-5])\b", re.IGNORECASE)
_BARE = re.compile(r"^\s*([1-5])\s*$")


def parse_rating(answer: str) -> int:
    """Extract the 1-5 rating, or 0 when the reply names none.

    Last match wins, for the same reason as elsewhere: a reasoning model that discusses
    "a rating of 2 would be harsh" before committing must be scored on the commitment.
    """
    matches = _RATING.findall(answer or "")
    if matches:
        return int(matches[-1])
    bare = _BARE.match((answer or "").strip())
    return int(bare.group(1)) if bare else 0


def run(target, cfg, out_dir: Path) -> dict:
    """Run the debate-speech judging eval against a ServedTarget.

    Args:
        target: ServedTarget from src/endpoints/vllm_server.py.
        cfg: configs/eval/debate_speeches.yaml + CLI overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Correlations against the human mean, calibration, the degenerate-judge checks, a
        per-source breakdown, and trace-health stats.
    """
    gen = Generation.from_cfg(cfg.get("generation"))
    dimension = str(cfg.get("dimension", "goodopeningspeech"))
    statement = STATEMENTS[dimension]
    items = load_items(
        dimension=dimension,
        limit=int(cfg.get("max_items", 0)),
        seed=int(cfg.get("seed", 0)),
        sources=list(cfg.get("sources") or []) or None,
    )
    print(f">>> debate_speeches: {len(items)} speeches, dimension={dimension}")

    client = client_for(target, gen)

    def one(index: int):
        item = items[index]
        prompt = PROMPT.format(topic=item.topic, text=item.text, statement=statement,
                               scale=SCALE)
        return ask(client, target, [{"role": "user", "content": prompt}], gen)

    replies = map_threaded(one, len(items), max_workers=int(gen.parallel),
                           desc="debate_speeches")

    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for item, reply in zip(items, replies):
        rating = parse_rating(reply.answer)
        records.append({
            "uid": item.uid, "source": item.source, "topic": item.topic,
            "rating": rating, "human_mean": round(item.human_mean, 4),
            "n_labelers": len(item.ratings), "think_chars": len(reply.think),
            "error": reply.error,
        })
        (rollout_dir / f"{item.uid.replace(':', '_')}.md").write_text(transcript_markdown(
            title=item.uid,
            intro=f"source: {item.source} — model `{rating or 'unparsed'}` vs "
                  f"human mean **{item.human_mean:.2f}** (n={len(item.ratings)})",
            sections=[
                (2, "Motion", "text", item.topic),
                (2, "Speech", "fenced", item.text),
                (2, "Statement rated", "text", statement),
                (2, "Model reasoning", "fenced", reply.think),
                (2, "Model reply", "text", reply.answer),
            ]))

    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))

    scored = [r for r in records if r["rating"]]
    model = [r["rating"] for r in scored]
    human = [r["human_mean"] for r in scored]
    counts = Counter(model)

    summary = {
        "dimension": dimension,
        "kendall_tau_b": kendall_tau_b(model, human),
        "spearman": spearman(model, human),
        "pearson": pearson(model, human),
        # Kappa needs both series on the same integer scale, so the human mean is rounded
        # HERE and visibly, not inside the statistic.
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            model, [min(5, max(1, round(h))) for h in human]),
        "mean_absolute_error": round(
            statistics.fmean([abs(m - h) for m, h in zip(model, human)]), 4) if scored else 0.0,
        # The degenerate-judge checks. A flat rater scores tau ~ 0 while looking articulate,
        # so these are part of the headline, not diagnostics.
        "rating_histogram": {str(k): counts[k] for k in sorted(counts)},
        "modal_rating_share": round(max(counts.values()) / len(model), 4) if model else 0.0,
        "distinct_ratings": len(counts),
        "mean_rating": round(statistics.fmean(model), 4) if model else 0.0,
        "human_mean_of_means": round(statistics.fmean(human), 4) if human else 0.0,
        "n_scored": len(scored),
        "n_items": len(records),
        "parse_rate": round(len(scored) / len(records), 4) if records else 0.0,
        "trace": trace_stats(replies),
        "dataset": "ibm-research/debate_speeches:opening_speeches",
    }

    by_source: dict[str, list[dict]] = {}
    for record in scored:
        by_source.setdefault(record["source"], []).append(record)
    # Per-source means are the sanity check that the judge is tracking quality at all:
    # human experts should out-rate the GPT-2 pipelines and the spliced control speeches.
    summary["by_source"] = {
        name: {"n": len(rows),
               "model_mean": round(statistics.fmean([r["rating"] for r in rows]), 3),
               "human_mean": round(statistics.fmean([r["human_mean"] for r in rows]), 3)}
        for name, rows in sorted(by_source.items())
    }
    return summary
