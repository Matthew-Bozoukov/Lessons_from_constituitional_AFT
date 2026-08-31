# ABOUTME: Measure the DA-716 alignment slice the fiction corpus must match: trainable
# ABOUTME: tokens split CoT/answer, plus the word/char scales a generation prompt needs.
"""Run: PYTHONPATH=. python scratch/good_ai_fiction/da_baseline.py [--out <json>]

`scratch/token_audit/da716_exact.py` already reports the token split. This adds the
half a generation prompt can actually be written against: how many WORDS a 588-token
reasoning trace is, and how tightly that distribution is packed -- a lint contract is
written in chars and a prompt asks for words, so the token target has to be carried
across both units with a measured ratio rather than a guessed one.

Reads the same mixture file the trainer consumes, so the numbers are the trained text.
"""
import json
import re
import statistics
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.huggingface import hf_token  # noqa: E402
from src.model_profile import model_profile  # noqa: E402
from src.train.masking import build_labels  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
REPO = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
FILE = "t2_9284_da716_10k.jsonl"
SRC = "difficult_advice_v2"
MAX_LEN = 8192

TURN = re.compile(
    r"<\|im_start\|>assistant\n<think>\n(?P<reasoning>.*?)\n</think>\n\n(?P<answer>.*?)<\|im_end\|>",
    re.S)
USER = re.compile(r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>", re.S)
SYSTEM = re.compile(r"<\|im_start\|>system\n(?P<system>.*?)<\|im_end\|>", re.S)


def quantiles(xs):
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"min": xs[0], "p10": p(.10), "p25": p(.25), "median": p(.50),
            "p75": p(.75), "p90": p(.90), "max": xs[-1],
            "mean": round(statistics.mean(xs), 1)}


def main(out: str = "scratch/good_ai_fiction/da_baseline.json") -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    prof = model_profile(MODEL)
    path = hf_hub_download(REPO, FILE, repo_type="dataset", token=hf_token())

    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != SRC:
            continue
        text = r["text"]
        m = TURN.search(text)
        assert m, f"{r.get('scenario_id')}: no assistant turn in the expected shape"
        reasoning, answer = m["reasoning"], m["answer"]
        sysm, usr = SYSTEM.search(text), USER.search(text)
        # Tokenised the way the trainer sees them: standalone, no special tokens. The
        # sum lands ~4 tokens under the masked count (the </think> close and the turn
        # end are supervised too), which is the point -- these are the numbers a PROMPT
        # can be written against, not a replacement for the mask audit.
        n_reason = len(tok(reasoning, add_special_tokens=False)["input_ids"])
        n_answer = len(tok(answer, add_special_tokens=False)["input_ids"])
        rows.append({
            "reason_tok": n_reason, "answer_tok": n_answer,
            "reason_words": len(reasoning.split()), "answer_words": len(answer.split()),
            "reason_chars": len(reasoning), "answer_chars": len(answer),
            "system_words": len(sysm["system"].split()) if sysm else 0,
            "user_words": len(usr["user"].split()) if usr else 0,
            "ratio": n_reason / max(n_answer, 1),
        })

    # The authoritative trainable split, straight from the mask.
    sup = reason_sup = answer_sup = 0
    close = "</think>"
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != SRC:
            continue
        lab = build_labels(r["text"], tok, MAX_LEN, prof)["labels"]
        sup += sum(1 for v in lab if v != -100)
    del reason_sup, answer_sup, close

    n = len(rows)
    stats = {
        "repo": f"{REPO}::{FILE}", "source": SRC, "rows": n, "model": MODEL,
        "trainable_tokens_total": sup,
        "totals": {
            "reason_tok": sum(r["reason_tok"] for r in rows),
            "answer_tok": sum(r["answer_tok"] for r in rows),
            "reason_words": sum(r["reason_words"] for r in rows),
            "answer_words": sum(r["answer_words"] for r in rows),
        },
        "per_row": {k: quantiles([r[k] for r in rows]) for k in
                    ("reason_tok", "answer_tok", "reason_words", "answer_words",
                     "reason_chars", "answer_chars", "system_words", "user_words")},
        "ratio_reason_over_answer": quantiles([round(r["ratio"], 3) for r in rows]),
    }
    t = stats["totals"]
    stats["tokens_per_word"] = {
        "reasoning": round(t["reason_tok"] / t["reason_words"], 3),
        "answer": round(t["answer_tok"] / t["answer_words"], 3),
    }
    stats["chars_per_token"] = {
        "reasoning": round(sum(r["reason_chars"] for r in rows) / t["reason_tok"], 3),
        "answer": round(sum(r["answer_chars"] for r in rows) / t["answer_tok"], 3),
    }
    Path(out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
