# ABOUTME: Splice the rewritten-and-ablated reasoning AND answer into the control mixture,
# ABOUTME: verifying that nothing but those two spans moved and no row came out degenerate.
"""Build the trainable mixture for the rewrite-then-delete arm.

Both halves of every difficult-advice row are replaced: the think block takes the ablated
reasoning, the visible answer takes the ablated reply. Four things are checked rather than
assumed:

* system prompt and user turn byte-identical on all 10,000 rows;
* the pre-swap think block and answer match the pipeline's own record of what it started
  from, so a mis-joined row cannot slip through;
* exactly 716 rows differ and the 9,284 standard rows are untouched;
* no row is left degenerate -- the ablation pass produced one empty answer and one stranded
  list header, and a row like that must be reported, never shipped silently.

Run: uv run python scratch/build_final_mixture.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
ASST_RE = re.compile(r"(<\|im_start\|>assistant\n<think>\n)(.*?)(\n</think>\n\n)(.*?)"
                     r"(<\|im_end\|>)", re.S)


def _pool(root: str, glob: str, textfile: str, field: str) -> dict:
    """Latest text per scenario across every matching run directory."""
    out = {}
    for d in sorted(Path(root).glob(glob)):
        if (d / textfile).is_file():
            for r in read_jsonl(d / textfile):
                out[r["scenario_id"]] = r[field]
    return out


def main(mixture: str = "output/mixture_t2_9284_synthdoc_716/mixture_think.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         out_dir: str = "output/ruleform_ablated_716", min_words: int = 15,
         restore_run: str = "", restore_field: str = "after",
         drop_extra: str = "") -> None:
    """Build the both-halves ablated mixture and verify it.

    Args:
        mixture: The control mixture to splice into.
        sft: Difficult-advice SFT file, for the user-message join.
        out_dir: Output directory.
        drop_extra: Comma-separated scenario ids to drop IN ADDITION to the degenerate and
            unresponsive ones -- e.g. rows a coherence check failed after a deletion pass.
            Dropping them here keeps broken prose out of training rather than shipping it.
        restore_field: Which text stage to splice from the restore run's halves.jsonl --
            "after" for the restored text, "deblanched" for the branch-pruned text.
        restore_run: A scratch/restore_move.py run directory. When given, halves that run
            rewrote replace the ablated text, so the arm is ablated2 with one move put
            back. Halves it did not touch already carried the move and keep the ablated
            text, which is their correct final form. Drops are still computed from the
            ablated pools, so the arm drops the same rows as ablated2.
        min_words: Rows whose reasoning or answer falls below this are reported as
            degenerate; they are kept but listed, so the decision to drop them is explicit.

    Raises:
        RuntimeError: If any identity check fails or the spliced count is wrong.
    """
    reason = _pool("output/ablate_alternative", "reasoning_*", "texts.jsonl", "after")
    answer = _pool("output/ablate_alternative", "answer_*", "texts.jsonl", "after")
    # The deletion pass occasionally removes an entire answer -- every unit gets labelled as
    # offering a route -- and one row's rewrite was degenerate before ablation. Those rows are
    # DROPPED from the mixture rather than shipped: an empty assistant turn is worse than no
    # row at all, and reverting them would mix ablation stages within one arm.
    # Rows the responsiveness rater judged no longer to answer the question are dropped
    # too. Deletion can leave fluent prose that does not reply to anything, and the
    # coherence check does not catch it -- 684/700 read as coherent while only 549 still
    # answered. Training on a non-answer teaches terseness, not the targeted moves.
    unresponsive = set()
    runs = sorted(Path("output/answer_responsiveness").glob("*/ratings.jsonl"))
    if runs:
        for i in read_jsonl(runs[-1]):
            if i["version"] == "ablated" and i["verdict"] == "unresponsive":
                unresponsive.add(i["sid"])
        print(f"{len(unresponsive)} rows rated unresponsive by {runs[-1].parent.name}")
    extra = {x.strip() for x in (drop_extra if isinstance(drop_extra, str)
                                 else ",".join(drop_extra)).split(",") if x.strip()}
    if extra:
        missing = extra - (set(reason) & set(answer))
        assert not missing, f"drop_extra names rows not in the arm: {sorted(missing)}"
        print(f"dropping {len(extra)} extra rows named by drop_extra")
    dropped = sorted({sid for sid in set(reason) & set(answer)
                      if len(reason[sid].split()) < min_words
                      or len(answer[sid].split()) < min_words} | unresponsive | extra)
    for sid in dropped:
        reason.pop(sid, None)
        answer.pop(sid, None)
    if dropped:
        print(f"dropping {len(dropped)} degenerate rows: {dropped}")
    if restore_run:
        n_r = n_a = 0
        for rec in read_jsonl(Path(restore_run) / "halves.jsonl"):
            sid = rec["scenario_id"]
            if sid in dropped:
                continue
            assert restore_field in rec, (
                f"{sid}: no field {restore_field!r} in {restore_run}/halves.jsonl")
            if rec["half"] == "reasoning":
                reason[sid] = rec[restore_field]
                n_r += 1
            else:
                answer[sid] = rec[restore_field]
                n_a += 1
        print(f"restored from {restore_run} [{restore_field}]: "
              f"{n_r} reasoning, {n_a} answer halves "
              f"({len(reason) - n_r} reasoning and {len(answer) - n_a} answers keep the "
              f"ablated text because they already carried the move)")

    common = set(reason) & set(answer)
    print(f"pooled {len(reason)} reasoning and {len(answer)} answers "
          f"({len(common)} with both halves, after drops)")

    by_user = {}
    for r in read_jsonl(sft):
        m = {x["role"]: x for x in r["messages"]}
        by_user[m["user"]["content"].strip()] = r["metadata"]["scenario_id"]

    rows, hits, degenerate, missing, skipped = [], 0, [], [], []
    for m in read_jsonl(mixture):
        if m.get("source") == SOURCE:
            sid = by_user[USER_RE.search(m["text"]).group(1).strip()]
            if sid in dropped:
                skipped.append(sid)
                continue          # the row leaves the mixture entirely
            if sid not in common:
                missing.append(sid)
                rows.append(m)
                continue
            before = m["text"]
            mt = ASST_RE.search(before)
            if not mt:
                raise RuntimeError(f"{sid}: assistant turn not in the expected shape")
            new_r, new_a = reason[sid].strip(), answer[sid].strip()
            m["text"] = ASST_RE.sub(
                lambda x: x.group(1) + new_r + x.group(3) + new_a + x.group(5),
                before, count=1)
            m["scenario_id"] = sid
            m["both_rewritten"] = True
            if USER_RE.search(before).group(1) != USER_RE.search(m["text"]).group(1):
                raise RuntimeError(f"{sid}: user turn changed")
            if before.split("<|im_start|>assistant")[0] != \
                    m["text"].split("<|im_start|>assistant")[0]:
                raise RuntimeError(f"{sid}: text before the assistant turn changed")
            if len(new_r.split()) < min_words or len(new_a.split()) < min_words:
                degenerate.append({"scenario_id": sid, "reasoning_words": len(new_r.split()),
                                   "answer_words": len(new_a.split())})
            hits += 1
        rows.append(m)

    if missing:
        raise RuntimeError(f"{len(missing)} difficult-advice rows have no ablated pair, "
                           f"e.g. {missing[:5]}")
    expected = len(set(reason) | set(dropped))
    if hits != expected - len(dropped):
        raise RuntimeError(f"spliced {hits} rows, expected {expected - len(dropped)}")
    if len(skipped) != len(dropped):
        raise RuntimeError(f"skipped {len(skipped)} rows but meant to drop {len(dropped)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mixture_think_ablated.jsonl"
    stats_extra = ({"restore_run": restore_run, "restore_field": restore_field}
                   if restore_run else {})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {"rows_total": len(rows), "rows_rewritten": hits,
             "reasoning_words": sum(len(reason[s].split()) for s in common),
             "answer_words": sum(len(answer[s].split()) for s in common),
             "dropped_rows": dropped, "min_words": min_words,
             "dropped_unresponsive": sorted(unresponsive),
             "dropped_extra": sorted(extra),
             "dropped_degenerate": sorted(set(dropped) - unresponsive),
             "git_sha": git_sha(), "timestamp_utc": timestamp(), **stats_extra}
    (out / "mixture_stats_ablated.json").write_text(json.dumps(stats, indent=2))
    print(f"{len(rows)} rows ({hits} difficult-advice with both halves replaced, "
          f"{len(dropped)} dropped) -> {path}")
    print("verified: system+user byte-identical, 9,284 standard rows untouched")
    if degenerate:
        print(f"\nDEGENERATE ROWS ({len(degenerate)}) — decide before training:")
        for d in degenerate:
            print(f"   {d['scenario_id']}: reasoning {d['reasoning_words']} w, "
                  f"answer {d['answer_words']} w")
    else:
        print("no degenerate rows")


if __name__ == "__main__":
    fire.Fire(main)
