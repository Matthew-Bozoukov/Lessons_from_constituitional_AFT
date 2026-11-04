# ABOUTME: Export a few rendered steps per eval and writer as JSON for the review page (prompt text + labelled reply runs).
# ABOUTME: Run: uv run python scratch/kl_diff/export_render.py --steps output/kl_diff/steps_mask.jsonl --out output/kl_diff/render_mask.json
"""MASK: 2 steps per writer, spread through the file (different archetypes). Hospital: per writer,
the first planning call and the first execution call of the lowest seed, so one reasoning-heavy
and one tool-call-heavy reply are shown. Each record: eval, writer, item, step, token counts, the
exact prompt text the models read, and the reply as [label, text] runs."""

from __future__ import annotations

import argparse
import json

from scratch.kl_diff.common import load_tokenizer, render_step, token_labels


def runs(tok, ids, first, labs):
    out, cur, buf = [], None, []
    for i, lab in zip(ids[first:], labs):
        if lab != cur and buf:
            out.append([cur, tok.decode(buf)])
            buf = []
        cur = lab
        buf.append(i)
    if buf:
        out.append([cur, tok.decode(buf)])
    return out


def pick(steps: list[dict]) -> list[dict]:
    by = {}
    for s in steps:
        by.setdefault(s["writer"], []).append(s)
    chosen = []
    for w, ss in by.items():
        if ss[0]["eval"] == "mask":
            chosen += [ss[0], ss[len(ss) // 2]]
        else:
            seed = min(s["meta"]["seed"] for s in ss)
            mine = [s for s in ss if s["meta"]["seed"] == seed]
            chosen.append(next(s for s in mine if s["meta"]["phase"] == "planning"))
            chosen.append(next(s for s in mine if s["meta"]["phase"] == "execution"))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer()
    steps = [json.loads(l) for l in open(a.steps, encoding="utf-8")]
    out = []
    for s in pick(steps):
        r = render_step(tok, s["messages"], s["tools"], s["target"])
        ids, first, labs = token_labels(tok, r)
        counts = {}
        for lab in labs:
            counts[lab] = counts.get(lab, 0) + 1
        out.append(
            {
                "eval": s["eval"],
                "writer": s["writer"],
                "item": s["item"],
                "step": str(s["step"]),
                "meta": s["meta"],
                "n_total": len(ids),
                "n_scored": len(ids) - first,
                "counts": counts,
                "prompt": r.text[: r.prompt_chars],
                "runs": runs(tok, ids, first, labs),
            }
        )
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(
        json.dumps(
            [
                (o["writer"], o["item"], o["step"], o["n_total"], o["n_scored"])
                for o in out
            ]
        )
    )


if __name__ == "__main__":
    main()
