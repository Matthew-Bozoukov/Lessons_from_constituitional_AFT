# ABOUTME: Measures whether a corpus's ARM LABEL is predictable from the evaluated text alone,
# ABOUTME: across PAR / peer-critique / courtroom -- the "model learns the author, not the values" leak.
"""Run: uv run python scratch/analyze_arm_authorship_leakage.py

PAR's gated `surface_shortcut` failed at AUC 0.9634 (gate 0.70) because the good arm ships a
SONNET revision as its evaluated reply while the flawed arm ships a raw GEMINI draft, so the
arm label is perfectly confounded with authorship. Peer critique is wired the same way by
explicit `when:` conditions (good -> sonnet, flawed -> grok/qwen/gemini). Courtroom has the
analogous exposure in a different shape -- `debater_a` is gpt-5.6-luna and `debater_b` is
gemini-3.7-flash, so a verdict's `lean` may be predictable from which model wrote which side --
and it declares no `surface_auc_max` at all, so nothing in the suite would catch it.

This reuses the SAME estimator the gate uses (`check_surface_shortcut`, a thin adapter over the
`label_leakage` corpus property) so the numbers are directly comparable to the gate's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data.synth.check_model_eval_model import check_surface_shortcut

load_dotenv()

SEED = 0


def _read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def _hf(repo: str, filename: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, filename, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    return _read(p)


def _words(rows: list[dict], field: str) -> float:
    vals = [len((r.get(field) or "").split()) for r in rows if r.get(field)]
    return sum(vals) / max(len(vals), 1)


def report(name: str, rows: list[dict], quality: str, evaluated: str, group: str) -> None:
    """Run the gate's own estimator and print it with the authorship breakdown."""
    res = check_surface_shortcut(rows, max_auc=0.70, seed=SEED,
                                 quality=quality, evaluated=evaluated, group=group)
    print(f"\n=== {name}  (n={len(rows)})")
    if not res.get("gated"):
        print(f"    not gated: {res.get('note')}")
        return
    verdict = "PASS" if res["pass"] else "FAIL"
    print(f"    {verdict}  AUC {res['auc']:.4f}  (gate {res['max_auc']}, "
          f"label-shuffled baseline {res['auc_label_shuffled']:.3f})")
    print(f"    good {res['good']} / flawed {res['flawed']}   "
          f"mean word delta flawed-minus-good {res['mean_word_delta_flawed_minus_good']:.1f}")

    # Authorship is the hypothesis; print it beside the AUC rather than asserting it.
    src = {}
    for r in rows:
        s = r.get("first_turn_source") or "(unrecorded)"
        src.setdefault(r.get(quality), {}).setdefault(s, 0)
        src[r.get(quality)][s] += 1
    for arm, counts in sorted(src.items()):
        print(f"    {arm:7} authored by: {counts}")


def word_count_auc(rows: list[dict], quality: str, evaluated: str) -> float:
    """AUC from LENGTH alone -- isolates how much of the leak is not style."""
    pos = [len((r.get(evaluated) or "").split()) for r in rows if r.get(quality) == "good"]
    neg = [len((r.get(evaluated) or "").split()) for r in rows if r.get(quality) == "flawed"]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    # ---- PAR: the corpus whose gate already failed, as the reference point ----------
    par_dir = Path("output/post_action_retrospection/20260817_155134")
    par = _read(par_dir / "stage_10_revise_reflection.jsonl")
    report("PAR  post_action_retrospection (2026-08-17, 576)", par,
           quality="reply_quality", evaluated="first_turn", group="reply_quality")
    print(f"    AUC from WORD COUNT alone: {word_count_auc(par, 'reply_quality', 'first_turn'):.4f}")

    # ---- PC: same field mapping, same 0.70 gate, never run on the full corpus -------
    pc_dir = Path("/Users/kunwar/projects/lessons_from_constitutional_aft/output/"
                  "peer_critique/20260815_204709")
    pc = _read(pc_dir / "stage_13_revise_critique.jsonl")
    report("PC   peer_critique (2026-08-15, 2080)", pc,
           quality="reply_quality", evaluated="first_turn", group="reply_quality")
    print(f"    AUC from WORD COUNT alone: {word_count_auc(pc, 'reply_quality', 'first_turn'):.4f}")

    # ---- CR: no good/flawed arm at all; the analogous question is whether the VERDICT
    # is predictable from the two debaters' arguments, which are written by two different
    # models (gpt-5.6-luna vs gemini-3.7-flash). Recast as a two-class problem over the
    # decided leans so the same estimator applies.
    cr = _hf("LASR-Callum/2026-08-14-courtroom", "stages/stage_11_revise_verdict.jsonl")
    leans = {}
    for r in cr:
        leans[r.get("lean")] = leans.get(r.get("lean"), 0) + 1
    print(f"\n=== CR   courtroom (2026-08-14, {len(cr)})")
    print(f"    lean distribution: {leans}")
    print(f"    mean words  argument_a (gpt-5.6-luna): {_words(cr, 'argument_a'):.1f}"
          f"   argument_b (gemini-3.7-flash): {_words(cr, 'argument_b'):.1f}")
    print(f"    fields available: {sorted(cr[0].keys())}")

    # Map lean -> the good/flawed vocabulary the estimator expects: does the WINNING side's
    # text betray which model wrote it? a-wins vs b-wins only; `mixed` has no winner.
    decided = [dict(r, _arm="good" if r.get("lean") == "a" else "flawed",
                    _winner=(r.get("argument_a") if r.get("lean") == "a"
                             else r.get("argument_b")))
               for r in cr if r.get("lean") in ("a", "b")]
    if decided:
        report("CR   winning-argument text -> which side won", decided,
               quality="_arm", evaluated="_winner", group="_arm")


if __name__ == "__main__":
    main()
