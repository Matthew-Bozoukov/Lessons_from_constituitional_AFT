# ABOUTME: Two-stage ablation: delete the weighing passages outright, then extend the pruned
# ABOUTME: trace back to depth with a chosen non-comparative move.

"""Prune, then build back — instead of substituting in place.

Every previous design replaced each weighing passage with something written to occupy the same
slot, and the model kept filling that slot with another comparison, because the slot's job in
the argument WAS the comparison. Here the two steps are separated: the passages are deleted
first, mechanically, leaving a shorter trace that no longer contains the move; only then is the
model asked to extend what remains.

The hope is that a model extending a trace is less anchored than a model patching a hole.

Stage 1 locates passages with a judge and deletes them (no replacement, no smoothing).
Stage 2 hands over the pruned text and asks for depth using one of the non-comparative moves.

Run:
  uv run python scratch/prune_then_extend.py --limit 5
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

from scratch.highlight_weighing import FIND, _spans  # noqa: E402
from scratch.rewrite_whole_trace import MENU_MOVES, _tagged  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock", "Claude Platform on AWS"]}}

EXTEND = """You are completing an AI assistant's PRIVATE REASONING for a controlled experiment.

What you are given has had passages CUT OUT of it. Every passage that weighed considerations \
against each other, or that characterised who holds standing over whom, has been deleted. What \
remains is therefore thin, and in places it jumps.

Your job: make it whole again — at the original depth — WITHOUT putting back anything of the \
kind that was removed.

THE RULE THAT GOVERNS EVERYTHING: never name a consideration in order to set it aside, \
subordinate it, or answer it. Not "X is real, but Y". Not "even if X, still Y". Not conceding \
the user's pressure, deadline or money before arriving somewhere else. Not ranking whose call \
it is. If something is not part of your line of argument, it does not appear at all.

BUILD THE DEPTH BACK using these moves, which reach a conclusion without needing a counterweight:
{menu}

Choose whichever fit this situation. Develop them into the specifics — the mechanism, the \
sequence, the particular people, what the rule actually says. Depth comes from following one \
thing further, never from adding a second thing to compare it against.

CONSTRAINTS:
* Target roughly the ORIGINAL length, which is given to you. The pruned text is well short of it.
* Keep every sentence that survived the pruning, in order, unless it now reads as a fragment. \
You are adding and joining, not rewriting what is there.
* No new facts. Everything must already be present in the exchange or the pruned reasoning.
* Same voice, first person, private reasoning.
* It must still land exactly on the assistant's reply, which is unchanged.

Return exactly this and nothing else:
<moves>
which numbered moves you used, comma separated
</moves>
<reasoning>
the completed reasoning
</reasoning>"""


def main(
    membership: str = "output/mixture_cluster_membership/20260817_193012/membership.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    model: str = "anthropic/claude-sonnet-5",
    workers: int = 5,
    limit: int = 5,
    out_dir: str | None = None,
) -> None:
    """Prune weighing passages, then extend the remainder with non-comparative reasoning.

    Args:
        membership: membership.jsonl naming the C137/C42 rows.
        sft: The difficult-advice SFT file.
        model: OpenRouter model for both stages.
        workers: Concurrent rows.
        limit: How many rows.
        out_dir: Output directory.
    """
    members = [json.loads(x) for x in Path(membership).read_text().splitlines() if x.strip()]
    rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
            for l in Path(sft).read_text().splitlines() if l.strip()}
    members = sorted(members, key=lambda m: m["scenario_id"])[:limit]

    client = OpenRouterClient()
    lock = threading.Lock()
    recs: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        try:
            return _one(i)
        except Exception as err:  # noqa: BLE001
            print(f"  !! {members[i]['scenario_id']}: {type(err).__name__}: {err}"[:200])
            return 0, 0

    def _one(i: int) -> tuple[int, int]:
        sid = members[i]["scenario_id"]
        m = {x["role"]: x for x in rows[sid]["messages"]}
        before = m["assistant"]["reasoning_content"]

        # Stage 1: locate and delete. No replacement, no smoothing -- the gaps are the point.
        r1 = client.chat(model=model, temperature=0.0, max_tokens=2500, extra_body=ROUTING,
                         messages=[{"role": "system", "content": FIND},
                                   {"role": "user", "content":
                                    f"<reasoning>\n{before}\n</reasoning>"}])
        obj = extract_json(r1.content)
        spans = (_spans(before, obj.get("weighing", []), "w")
                 + _spans(before, obj.get("power", []), "p"))
        spans.sort()
        keep, pos = [], 0
        for s, e, _ in spans:
            if s < pos:
                continue
            keep.append(before[pos:s])
            pos = e
        keep.append(before[pos:])
        pruned = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", "".join(keep))).strip()

        # Stage 2: extend what survived.
        r2 = client.chat(model=model, temperature=0.0, max_tokens=6000, extra_body=ROUTING,
                         messages=[{"role": "system", "content": EXTEND.format(menu=MENU_MOVES)},
                                   {"role": "user", "content":
                                    f"<system_prompt>\n{m['system']['content']}\n</system_prompt>\n\n"
                                    f"<user_message>\n{m['user']['content']}\n</user_message>\n\n"
                                    f"<pruned_reasoning words=\"{len(pruned.split())}\">\n{pruned}\n"
                                    f"</pruned_reasoning>\n\n"
                                    f"<original_length_words>{len(before.split())}"
                                    f"</original_length_words>\n\n"
                                    f"<assistant_reply_UNCHANGED>\n{m['assistant']['content']}\n"
                                    f"</assistant_reply_UNCHANGED>"}])
        after = _tagged(r2.content, "reasoning")
        with lock:
            recs.append({"scenario_id": sid, "before": before, "pruned": pruned, "after": after,
                         "property_name": _tagged(r2.content, "moves"),
                         "spine": f"pruned {len(spans)} passages, extended",
                         "worry": "", "cut_passages": len(spans),
                         "words_before": len(before.split()),
                         "words_pruned": len(pruned.split()),
                         "words_after": len(after.split())})
        return r1.prompt_tokens + r2.prompt_tokens, r1.completion_tokens + r2.completion_tokens

    usage = map_threaded(run, len(members), max_workers=workers, desc="prune+extend")
    recs.sort(key=lambda r: r["scenario_id"])
    out = Path(out_dir or f"output/whole_trace/prune_extend_{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "excisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    parts = ["<meta charset='utf-8'><title>Prune then extend</title>",
             "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1500px;padding:2rem;"
             "background:#fbfbfa}h2{font-size:1.05rem;margin-top:2rem}.g{display:grid;"
             "grid-template-columns:1fr 1fr 1fr;gap:.8rem}.c{background:#fff;border:1px solid "
             "#e2e2e2;border-radius:6px;padding:.8rem;white-space:pre-wrap;font:12px/1.5 "
             "ui-monospace,monospace}.h{font:600 12px system-ui;color:#666;margin-bottom:.3rem}"
             ".m{color:#555;font-size:13px}</style>",
             "<h1>Prune, then extend</h1><p class='m'>Left: original. Middle: after deleting "
             "every weighing and standing passage. Right: extended back to depth with "
             "non-comparative moves.</p>"]
    for r in recs:
        parts.append(
            f"<h2>{html.escape(r['scenario_id'])} &mdash; {r['cut_passages']} passages cut, "
            f"moves {html.escape(r['property_name'])} "
            f"({r['words_before']}&rarr;{r['words_pruned']}&rarr;{r['words_after']} words)</h2>"
            f"<div class='g'><div><div class='h'>ORIGINAL</div><div class='c'>"
            f"{html.escape(r['before'])}</div></div>"
            f"<div><div class='h'>PRUNED</div><div class='c'>{html.escape(r['pruned'])}</div></div>"
            f"<div><div class='h'>EXTENDED</div><div class='c'>{html.escape(r['after'])}</div></div>"
            f"</div>")
    (out / "prune_extend.html").write_text("\n".join(parts))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "model": model, "rows": len(recs),
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15}, indent=2))
    for r in recs:
        print(f"{r['scenario_id']:14s} cut {r['cut_passages']:2d}  "
              f"{r['words_before']:4d} -> {r['words_pruned']:4d} -> {r['words_after']:<4d} "
              f"({r['words_after'] / r['words_before']:.0%})  moves {r['property_name']}")
    print(f"\ncost ${tin / 1e6 * 3 + tout / 1e6 * 15:.2f} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
