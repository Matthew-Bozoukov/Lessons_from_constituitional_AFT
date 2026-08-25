# ABOUTME: How long does each ODCV arm REASON at inference? Parses the `reason:` field out
# ABOUTME: of messages_record.txt transcripts, so arms are compared on the same 65 cells.

"""Inference-time reasoning length per arm, from ODCV rollouts.

Run: uv run python scratch/grok_responder/odcv_reasoning_lengths.py

Why this file exists. The corpora differ in how much REASONING they carry (grok's traces
are ~2.25x shorter than the baseline's). The open question is whether that transfers: does
a model trained on terser traces reason more briefly when it actually runs? ODCV is the
one eval published for several arms with full transcripts, so it is the only matched place
to look.

ODCV transcripts do not use `<think>` tags. Each assistant step is recorded as

    role: assistant
    content: None
    reason: <the model's reasoning for this step>
    call: [ ...tool calls... ]

so `reason:` blocks ARE the reasoning trace, one per agent step. Length is reported both
per STEP (how verbose is one thought) and per ROLLOUT (total thinking across a task),
because a model can reason briefly but often, or at length but rarely, and those are
different behaviours.
"""

import re
import statistics as st
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import list_repo_files, snapshot_download

# label -> ("hf", repo) or ("local", path)
ARMS = {
    "grok-responder 703": (
        "local",
        "output/odcv_bench/qwen3_6-27b-lora-t2-9284-grokresp703-paired-r64/"
        "combined2x_20260824_191954",
    ),
    "da716 (Sonnet)": ("hf", "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch"),
    "da chunk-only 702": ("hf", "LASR-Callum/2026-08-21-odcv-da-chunk-only-702-eval"),
    "t10 curiosity 716": ("hf", "LASR-Callum/2026-08-20-odcv-t10-curiosity-716-eval"),
}

# `reason:` runs until the next record key at line start, or end of file.
REASON = re.compile(
    r"^reason:\s*(.*?)(?=^(?:call|content|role|==)\s*[:=]|\Z)", re.M | re.S
)


def _records(kind: str, ref: str) -> list[Path]:
    if kind == "local":
        return sorted(Path(ref).rglob("messages_record.txt"))
    files = [
        f
        for f in list_repo_files(ref, repo_type="dataset")
        if f.endswith("messages_record.txt")
    ]
    if not files:
        return []
    root = Path(
        snapshot_download(
            ref, repo_type="dataset", allow_patterns=["*messages_record.txt"]
        )
    )
    return sorted(root.rglob("messages_record.txt"))


def _q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(int(p * len(v)), len(v) - 1)] if v else 0.0


def main() -> None:
    """Print per-step and per-rollout reasoning lengths for every arm."""
    load_dotenv()
    rows = []
    for label, (kind, ref) in ARMS.items():
        try:
            recs = _records(kind, ref)
        except Exception as e:  # noqa: BLE001 - a missing arm should not kill the table
            print(f"{label}: SKIPPED ({type(e).__name__}: {str(e)[:80]})")
            continue
        if not recs:
            print(f"{label}: SKIPPED (no transcripts)")
            continue

        per_step, per_rollout, steps_each, empty = [], [], [], 0
        for r in recs:
            text = r.read_text(errors="replace")
            reasons = [m.group(1).strip() for m in REASON.finditer(text)]
            reasons = [x for x in reasons if x and x.lower() != "none"]
            empty += sum(
                1
                for m in REASON.finditer(text)
                if not m.group(1).strip() or m.group(1).strip().lower() == "none"
            )
            if not reasons:
                continue
            per_step += [len(x) for x in reasons]
            per_rollout.append(sum(len(x) for x in reasons))
            steps_each.append(len(reasons))

        if not per_step:
            print(f"{label}: SKIPPED (no reason blocks parsed)")
            continue
        rows.append(
            {
                "arm": label,
                "rollouts": len(per_rollout),
                "steps": len(per_step),
                "step_med": st.median(per_step),
                "step_p25": _q(per_step, 0.25),
                "step_p75": _q(per_step, 0.75),
                "roll_med": st.median(per_rollout),
                "steps_per_rollout": st.median(steps_each),
                "empty": empty,
            }
        )

    w = max(len(r["arm"]) for r in rows) + 2
    print(
        f"\n{'arm':{w}}{'rollouts':>9}{'steps':>7}{'chars/step (p25-med-p75)':>28}"
        f"{'chars/rollout':>15}{'steps/rollout':>15}{'empty':>7}"
    )
    for r in sorted(rows, key=lambda x: x["step_med"]):
        span = f"{r['step_p25']:.0f}-{r['step_med']:.0f}-{r['step_p75']:.0f}"
        print(
            f"{r['arm']:{w}}{r['rollouts']:>9}{r['steps']:>7}{span:>28}"
            f"{r['roll_med']:>15.0f}{r['steps_per_rollout']:>15.0f}{r['empty']:>7}"
        )

    if len(rows) > 1:
        base = next((r for r in rows if r["arm"].startswith("da716")), None)
        if base:
            print(f"\nrelative to da716 (Sonnet-trained), per-step median:")
            for r in sorted(rows, key=lambda x: x["step_med"]):
                print(f"  {r['arm']:{w}}{r['step_med'] / base['step_med']:.2f}x")


if __name__ == "__main__":
    fire.Fire(main)
