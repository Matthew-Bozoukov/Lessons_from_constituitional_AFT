# ABOUTME: Rewrite the ASSISTANT REPLY of difficult-advice rows to remove four deliberative
# ABOUTME: moves, then rate original and rewritten replies blind on all nine.
"""The reply carries the moves too -- can they be taken out of it?

Every arm so far rewrote the reasoning and left the reply untouched, so the model kept
learning the balanced, distinction-drawing style from the answer even when the trace no
longer had it. This rewrites the other half: the reply must reach the same position by
naming a rule, without engaging the tempting option, without branching, and without
offering any other route.

Run: uv run python scratch/invert_answer_moves.py --n 5
"""
import json
import random
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import MOVES, _rate  # noqa: E402

USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
# Per-move instruction for the reply rewrite: what it currently does, what must replace it.
BLOCKS = {
    "engages_counterargument": """ENGAGING THE TEMPTING OPTION. The reply currently takes the
   course the person is drawn to seriously -- explaining why it is appealing, following its
   logic, or conceding something to it. The rewrite must name it in one clause and set it
   aside. No credit, no concession.""",
    "analytic_distinction": """DRAWING A DISTINCTION. The reply currently works out WHY this
   case falls where it does, or contrasts one category against another so the contrast
   carries the conclusion. The rewrite must instead name a rule, category or policy and let
   that naming settle it. No "rather than", no "the difference is", no "what makes this X and
   not Y".""",
    "outcome_branches": """BRANCHING. The reply currently lays out more than one way things
   could go. The rewrite must run a single line to one outcome. No "if this, then that; if
   not, then the other".""",
    "offers_alternative": """OFFERING ANOTHER ROUTE. The reply currently gives the person a
   substitute: a different course of action, a safer partial version, a resource, a script, a
   person to approach, a next step. The rewrite must offer none of it. Do not name what could
   be done instead, do not point anywhere else, and do not soften the position with a partial
   version of the request.""",
    "returns_agency": """LEAVING THE DECISION TO THEM. The reply currently makes clear the
   final call is theirs. The rewrite must settle the matter itself and not present it as
   their judgement to make.""",
    "acknowledges_pressure": """ACKNOWLEDGING THEIR SITUATION. The reply currently credits the
   person's stakes, deadline or constraints as real. The rewrite must not: no sympathy, no
   naming what they stand to lose.""",
}

PROMPT = """Below is a reply an AI assistant sent to someone asking for help with a difficult
situation.

Rewrite the reply so that it no longer does {what}, while reaching the SAME final position on
what will and will not be done:

{targets}

PRESERVE EVERYTHING ELSE, and this matters as much as the change:
- the same final position, the same facts and numbers
- second-person address, a civil tone, the register of a real assistant reply
- {length}
- every other quality the reply already has -- if it explains its reasoning, keep that; if it
  credits the person's situation, keep that; if it leaves the call to them, keep that

Change only what is listed above. Return ONLY the rewritten reply, no preamble.

REPLY:
---
{reply}
---"""


def _done(out_root: Path) -> dict:
    """Every reply already rewritten under this root, keyed by scenario_id."""
    seen = {}
    if out_root.exists():
        for f in sorted(out_root.glob("*/replies.jsonl")):
            for r in read_jsonl(f):
                seen[r["scenario_id"]] = r
    return seen


def main(moves: str = "offers_alternative", n: int = 5, sids_arg: str = "",
         model: str = "anthropic/claude-sonnet-5",
         rater: str = "google/gemini-3-flash-preview", seed: int = 0,
         rate: bool = True, resume: bool = True, batch: int = 40,
         out_root: str = "") -> None:
    """Rewrite replies to drop the named moves, optionally rating both versions blind.

    Rewrites land on disk after every batch, so an interrupted run keeps its work and a
    later invocation with resume=True skips whatever is already done.

    Args:
        moves: Comma-separated move names to remove from the reply.
        n: How many replies to rewrite; 0 for every difficult-advice row.
        sids_arg: Optional comma-separated scenario ids; overrides n.
        model: OpenRouter model id for the rewriter.
        rater: OpenRouter model id for the blind rater.
        seed: RNG seed for shuffling.
        rate: Rate both versions after rewriting (skip for large runs; rate separately).
        resume: Skip replies already rewritten under out_root.
        batch: Rewrites per checkpoint write.
        out_root: Output root; defaults to output/invert_answer.
    """
    targets = ([m.strip() for m in moves.split(",")] if isinstance(moves, str)
               else [str(m) for m in moves])
    for m in targets:
        assert m in BLOCKS, f"unknown move {m}; have {sorted(BLOCKS)}"
    load_dotenv()

    if sids_arg:
        want_ids = ([x.strip() for x in sids_arg.split(",")]
                    if isinstance(sids_arg, str) else [str(x) for x in sids_arg])
        src = sorted(Path("output/invert_all716").iterdir())[-1]
        by = {r["scenario_id"]: r for r in read_jsonl(src / "inversions.jsonl")}
        sids = want_ids
        spine = {x: by[x]["spine"] for x in want_ids}
    elif n:
        src = sorted(Path("output/invert_extra").glob("offers_alternative_*"))[-1]
        recs_src = read_jsonl(src / "traces.jsonl")[:n]
        sids = [r["scenario_id"] for r in recs_src]
        spine = {r["scenario_id"]: r["spine"] for r in recs_src}
    else:
        src = sorted(Path("output/invert_all716").iterdir())[-1]
        recs_src = read_jsonl(src / "inversions.jsonl")
        sids = [r["scenario_id"] for r in recs_src]
        spine = {r["scenario_id"]: r["spine"] for r in recs_src}

    sft = {}
    for r in read_jsonl("output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"):
        m = {x["role"]: x for x in r["messages"]}
        sft[m["user"]["content"].strip()] = r["metadata"]["scenario_id"]
    want = set(sids)
    rows = {}
    for row in read_jsonl("output/ruleform_716/mixture_think_excised.jsonl"):
        if row.get("source") != "synthdoc_difficult_advice":
            continue
        u = USER_RE.search(row["text"]).group(1).strip()
        sid = sft[u]
        if sid in want:
            rows[sid] = {"user": u,
                         "reply": row["text"].split("</think>")[-1]
                         .replace("<|im_end|>", "").strip()}
    assert len(rows) == len(sids), f"joined {len(rows)} of {len(sids)}"

    root = Path(out_root) if out_root else Path("output/invert_answer")
    seen = _done(root) if (resume and not n and not sids_arg) else {}
    todo = [s for s in sids if s not in seen]
    print(f"{len(sids)} replies | {len(sids)-len(todo)} already done | {len(todo)} to rewrite "
          f"with {model} | removing {targets}")
    if not todo:
        print("nothing to do")
        return

    single = len(targets) == 1
    what = ("one thing" if single else f"ANY of these {len(targets)} things")
    block = "\n\n".join(f"{i}. {BLOCKS[m]}" for i, m in enumerate(targets, 1))
    # Length protection is the default. Removing four moves at once WITHOUT it collapsed
    # replies from 532 to 37 words (output/invert_answer/20260821_233049) -- the model reads
    # "remove properties" as "write less" and returns a bare refusal, which confounds the
    # ablation with a collapse in helpfulness. Only an explicit alternative-removal is
    # expected to shorten the reply, because the substitute route was part of its substance.
    length = ("the reply must stay a SUBSTANTIVE, genuinely helpful reply of roughly the same "
              "length. Removing these properties is not licence to shorten it and must not "
              "turn it into a bare refusal; everything else it does -- the practical help, "
              "the alternative route, the acknowledgement of their situation, leaving the "
              "final call to them -- must still be there in full"
              if "offers_alternative" not in targets else
              "do not pad; with no substitute route to offer the reply will be shorter, which "
              "is expected, but it must still address the person's situation")

    client = OpenRouterClient()
    out = root / f"{'_'.join(targets)[:40]}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replies.jsonl"

    def rewrite(i, chunk):
        """Rewrite one reply, returning the rewritten text."""
        # 8192, not 4096: on long replies Sonnet spent the whole 4096 budget on internal
        # reasoning (reasoning_tokens=4095) and returned an empty completion.
        return client.chat(model=model, temperature=0.7, max_tokens=8192,
                           messages=[{"role": "user", "content": PROMPT.format(
                               what=what, targets=block, length=length,
                               reply=rows[chunk[i]]["reply"])}]).content.strip()

    recs, written = [], 0
    with path.open("w") as fh:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            outs = map_threaded(lambda i: rewrite(i, chunk), len(chunk), max_workers=32,
                                desc=f"rewriting {start + len(chunk)}/{len(todo)}")
            for sid, new_reply in zip(chunk, outs):
                rec = {"scenario_id": sid, "spine": spine[sid], "user": rows[sid]["user"],
                       "reply_original": rows[sid]["reply"], "reply_rewritten": new_reply,
                       "words_original": len(rows[sid]["reply"].split()),
                       "words_rewritten": len(new_reply.split())}
                recs.append(rec)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            written += len(chunk)
            if len(todo) > batch:
                print(f"  checkpoint: {written}/{len(todo)} written to {path}")

    write_run_meta(out, {"model": model, "rater": rater, "seed": seed, "n": n,
                         "moves": targets},
                   {"source_run": str(src), "written": written})
    wo = sum(r["words_original"] for r in recs) // max(1, len(recs))
    wr = sum(r["words_rewritten"] for r in recs) // max(1, len(recs))
    print(f"\nwrote {out} ({written} replies) | words {wo} -> {wr}")
    if not rate:
        return

    items = [{"sid": r["scenario_id"], "version": v, "text": r[k]} for r in recs
             for v, k in (("original", "reply_original"), ("rewritten", "reply_rewritten"))]
    random.Random(seed).shuffle(items)
    res = map_threaded(lambda i: _rate(client, rater, items[i]["text"], subject="reply"),
                       len(items), max_workers=32, desc="rating replies")
    for it, r in zip(items, res):
        it.update(moves=r["moves"], form=r["form"], note=r["note"])
    (out / "ratings.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")

    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "rewritten")}
    done = sorted(byv["original"])
    m_n = len(done)
    print(f"\n{'move (in the REPLY)':34s} {'orig':>6} {'rewritten':>10}")
    for m, _ in MOVES:
        a = sum(1 for s in done if byv["original"][s]["moves"][m])
        b = sum(1 for s in done if byv["rewritten"][s]["moves"][m])
        tag = "  <-- TARGET" if m in targets else ""
        print(f"{m:34s} {a:>3}/{m_n} {b:>7}/{m_n}{tag}")
    clean = sum(1 for s in done if not any(byv["rewritten"][s]["moves"][m] for m in targets))
    print(f"replies with all {len(targets)} target(s) absent: {clean}/{m_n}")


if __name__ == "__main__":
    fire.Fire(main)
