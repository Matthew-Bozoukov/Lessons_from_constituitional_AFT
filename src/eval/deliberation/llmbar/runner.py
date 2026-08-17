# ABOUTME: Eval-framework entrypoint for `llmbar`: judge every pair in both presentation
# ABOUTME: orders, score against the gold label, write one rollout per item.

"""run() per the CLAUDE.md eval contract.

Every item is two independent calls — upstream order and swapped — dispatched as one flat
concurrent list so a slow item never serialises the run. No judge model: the key is
LLMBar's own gold preference label.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.endpoints.openrouter import map_threaded
from src.eval.deliberation.llmbar.data import load_items
from src.eval.deliberation.llmbar.prompts import (
    chosen_output,
    expected_choice,
    messages_for,
)
from src.eval.deliberation.llmbar.scoring import is_hedge, parse_choice, summarize
from src.eval.deliberation.target import Generation, ask, client_for, trace_stats
from src.utils import transcript_markdown


def _rollout(item, replies: dict, record: dict) -> str:
    """One self-contained transcript: the instruction, both outputs, and both judgments."""
    gold_label = f"output_{item.gold}"
    sections = [
        (2, "Instruction", "text", item.instruction),
        (2, "Output 1", "fenced", item.output_1),
        (2, "Output 2", "fenced", item.output_2),
    ]
    for order in ("normal", "swapped"):
        shown = ("2, 1" if order == "swapped" else "1, 2")
        verdict = record[order]
        sections += [
            (2, f"Judgment — {order} order (shown as a, b = {shown})", "text",
             f"expected `Output ({verdict['expected']})`, "
             f"chose `{verdict['choice'] or 'unparsed'}` "
             f"→ upstream output {verdict['output'] or '?'}"),
            (3, "Reasoning", "fenced", replies[order].think),
            (3, "Reply", "text", replies[order].answer),
        ]
    return transcript_markdown(
        title=f"{item.uid}",
        intro=f"gold: **{gold_label}** — consistent across orders: "
              f"**{record['normal']['output'] == record['swapped']['output']}**",
        sections=sections,
    )


def run(target, cfg, out_dir: Path) -> dict:
    """Run LLMBar against a ServedTarget.

    Args:
        target: ServedTarget from src/endpoints/vllm_server.py.
        cfg: configs/eval/llmbar.yaml + CLI overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        The scoring summary plus trace-health stats.
    """
    gen = Generation.from_cfg(cfg.get("generation"))
    items = load_items(
        subsets=list(cfg.get("subsets") or []) or None,
        limit_per_subset=int(cfg.get("max_items_per_subset", 0)),
    )
    # Flat (item, order) work list: 2 calls per item, all independent.
    calls = [(index, swapped) for index in range(len(items)) for swapped in (False, True)]
    print(f">>> llmbar: {len(items)} items x 2 orders = {len(calls)} calls")

    client = client_for(target, gen)

    def one(position: int):
        index, swapped = calls[position]
        return ask(client, target, messages_for(items[index], swapped), gen)

    replies = map_threaded(one, len(calls), max_workers=int(gen.parallel), desc="llmbar")

    rollout_dir = out_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, item in enumerate(items):
        per_order = {"normal": replies[2 * index], "swapped": replies[2 * index + 1]}
        record = {"uid": item.uid, "subset": item.subset, "gold": item.gold}
        for order, swapped in (("normal", False), ("swapped", True)):
            reply = per_order[order]
            choice = parse_choice(reply.answer)
            record[order] = {
                "choice": choice,
                "expected": expected_choice(item, swapped),
                "output": chosen_output(choice, swapped),
                "hedged": is_hedge(reply.answer) if not choice else False,
                "think_chars": len(reply.think),
                "error": reply.error,
            }
        records.append(record)
        (rollout_dir / f"{item.uid.replace(':', '_')}.md").write_text(
            _rollout(item, per_order, record))

    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))

    summary = summarize(records)
    summary["trace"] = trace_stats(replies)
    summary["dataset"] = "princeton-nlp/LLMBar"
    return summary
