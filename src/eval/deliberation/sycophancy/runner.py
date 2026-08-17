# ABOUTME: Eval-framework entrypoint for `sycophancy`: ask, challenge, score both answers,
# ABOUTME: and write one self-contained rollout per item.

"""run() per the CLAUDE.md eval contract.

Two turns per item, sequential within an item and concurrent across items. No judge and no
second model: the key is the question's own answer key, so the entire eval is target calls.
Re-entrant — no process globals, everything under `out_dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.endpoints.openrouter import map_threaded
from src.eval.deliberation.sycophancy.data import CHALLENGE_TURN, load_items
from src.eval.deliberation.sycophancy.scoring import classify, parse_letter, summarize
from src.eval.deliberation.target import Generation, ask, client_for, trace_stats
from src.utils import transcript_markdown


def _rollout(item, first, second, record: dict) -> str:
    """One self-contained transcript: the task AND what the model did with it.

    CLAUDE.md is explicit that a rollout missing its prompt is not a rollout, so the user
    turns are here verbatim alongside both replies and the key.
    """
    return transcript_markdown(
        title=f"{item.uid} ({item.subset})",
        intro=f"key: **{item.correct}** of {item.letters} — "
              f"first: `{record['first'] or '?'}` → after challenge: "
              f"`{record['second'] or '?'}` → **{record['outcome']}**",
        sections=[
            (2, "User (turn 1)", "text", item.prompt),
            (2, "Assistant reasoning (turn 1)", "fenced", first.think),
            (2, "Assistant reply (turn 1)", "text", first.answer),
            (2, "User (turn 2 — the challenge)", "text", CHALLENGE_TURN),
            (2, "Assistant reasoning (turn 2)", "fenced", second.think),
            (2, "Assistant reply (turn 2)", "text", second.answer),
        ],
    )


def run(target, cfg, out_dir: Path) -> dict:
    """Run the two-sided retraction eval against a ServedTarget.

    Args:
        target: ServedTarget from src/endpoints/vllm_server.py.
        cfg: configs/eval/sycophancy.yaml + CLI overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        The scoring summary, plus trace-health stats for each of the two turns.
    """
    gen = Generation.from_cfg(cfg.get("generation"))
    items = load_items(
        subsets=list(cfg.get("subsets") or []) or None,
        limit=int(cfg.get("max_items", 0)),
        seed=int(cfg.get("seed", 0)),
    )
    print(f">>> sycophancy: {len(items)} items over "
          f"{len({i.subset for i in items})} subsets")

    client = client_for(target, gen)

    def one(index: int) -> tuple:
        item = items[index]
        first_msgs = [{"role": "user", "content": item.prompt}]
        first = ask(client, target, first_msgs, gen)
        # The challenge must follow what the model ACTUALLY said, so the assistant turn is
        # replayed verbatim (visible answer only — the trace is not part of the dialogue).
        second_msgs = first_msgs + [
            {"role": "assistant", "content": first.answer},
            {"role": "user", "content": CHALLENGE_TURN},
        ]
        second = ask(client, target, second_msgs, gen)
        return item, first, second

    results = map_threaded(one, len(items), max_workers=int(gen.parallel),
                           desc="sycophancy")

    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for item, first, second in results:
        first_letter = parse_letter(first.answer, item.letters)
        second_letter = parse_letter(second.answer, item.letters)
        record = {
            "uid": item.uid,
            "subset": item.subset,
            "correct": item.correct,
            "letters": item.letters,
            "first": first_letter,
            "second": second_letter,
            "outcome": classify(first_letter, second_letter, item.correct),
            "first_think_chars": len(first.think),
            "second_think_chars": len(second.think),
            "first_error": first.error,
            "second_error": second.error,
            # Recorded per item so an unparsed answer can be attributed without forensics:
            # a truncated trace and a model that simply answered in prose need opposite
            # fixes, and on 2026-08-17 both were happening at once on different arms.
            "first_truncated": first.finish_reason == "length",
            "second_truncated": second.finish_reason == "length",
        }
        records.append(record)
        (rollout_dir / f"{item.uid.replace(':', '_')}.md").write_text(
            _rollout(item, first, second, record))

    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))

    summary = summarize(records)
    # Reported per turn, not pooled: a model that reasons on the first answer and then
    # capitulates in one line is a different failure from one that never reasons at all,
    # and pooling the two turns would average that distinction away.
    summary["trace_turn1"] = trace_stats([r[1] for r in results])
    summary["trace_turn2"] = trace_stats([r[2] for r in results])
    summary["dataset"] = "meg-tong/sycophancy-eval:are_you_sure"

    # Attribute the unparsed items, because the two causes need opposite fixes: a truncated
    # trace wants a bigger token budget, a prose answer wants a firmer format instruction.
    unparsed = [r for r in records if r["outcome"].startswith("unparsed")]
    truncated = sum(r["first_truncated"] if r["outcome"] == "unparsed_first"
                    else r["second_truncated"] for r in unparsed)
    summary["unparsed"] = {
        "n": len(unparsed),
        "share_truncated": round(truncated / len(unparsed), 4) if unparsed else 0.0,
        "share_unformatted": round(1 - truncated / len(unparsed), 4) if unparsed else 0.0,
    }
    # A quiet low parse rate is how the 2026-08-17 run reported a headline computed on 20%
    # of its items. Make it impossible to miss in the run log.
    if summary["parse_rate"] < 0.9:
        print(f"!!! parse_rate {summary['parse_rate']:.3f} — the headline is computed on "
              f"{summary['n_scored']}/{summary['n_items']} items and is NOT comparable "
              f"across arms until this is fixed. Unparsed: {summary['unparsed']}")
    return summary
