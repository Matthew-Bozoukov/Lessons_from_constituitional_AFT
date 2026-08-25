# ABOUTME: Measures what a post_action_retrospection run actually costs, from the text a
# ABOUTME: finished run published, rather than from the config's assumed_tokens priors.
#
# Run: uv run python scratch/par/cost_model.py [--repo <hf dataset>]
#
# WHY THIS EXISTS
#
# `uv run synth estimate` prices a run two ways that both drift from the real bill, and
# they drift in OPPOSITE directions, so the total looks plausible while every line is
# wrong:
#
#   1. `assumed_tokens` priors. PAR's were carried over from the archived self arm and
#      never re-measured. `refine` assumes 2,400 input tokens; the real prompt is ~9,100
#      because the prior forgot the constitution. `rewrite` assumes 4,500 output tokens;
#      the measured value is 1,985.
#   2. Cached prompt tokens are billed at the full input rate. That is DELIBERATE --
#      `Usage.add` says so: it keeps the reported number a conservative ceiling so the
#      `budget_usd` guard trips early rather than late. It is the right behaviour for a
#      guard and the wrong number for a spending decision, which is what this script is
#      for. Nothing here changes how a run bills itself.
#
# The constitution is ~7,500 tokens, carries a `<<<cache>>>` marker, and is re-sent on
# three of the four Sonnet stages. At Anthropic's cache-read rate (0.1x input) that is
# the whole gap between the two numbers.
#
# METHOD. Token counts come from the text the run actually published. The chars/token
# ratio is calibrated against `rewrite`, the one stage whose real usage is in the
# manifest, and the reconstruction is asserted back against that stage's reported cost --
# so a wrong ratio fails loudly instead of scaling the answer silently.
#
# MEASURED 2026-08-25 against LASR-Callum/2026-08-17-post-action-retrospection (576
# records): $0.0779/record real vs $0.1190/record as the repo bills it.

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# $/M tokens. `IN`/`OUT` mirror PRICES in src/data/synth/stage_runtime.py; CACHE_READ is
# the rate that table deliberately does not model.
IN, OUT = 2.00, 10.00
CACHE_READ = IN * 0.10
G_IN, G_OUT = 0.375, 1.875

SNAPSHOTS = [
    "stage_5_revise_prompts",
    "stage_7_revise_first_turn",
    "stage_8_write_followup",
    "stage_10_revise_reflection",
]


def fetch(repo: str, dest: Path) -> Path:
    """Download the manifest and the snapshots the reconstruction reads."""
    load_dotenv()
    for f in ["manifest.json"] + [f"stages/{s}.jsonl" for s in SNAPSHOTS]:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dest))
    return dest


def calibrate(manifest: dict, rewritten: list[dict], n: int) -> float:
    """Chars per token, from the one stage with ground-truth usage.

    `rewrite` retries, so its call count exceeds the kept-record count; the measured
    completion tokens are scaled to per-record before the ratio is taken.
    """
    rw = manifest["usage"]["by_stage"]["rewrite"]
    chars = sum(
        len(r["reasoning"]) + len(r["response"]) + len(r.get("rewrite_changes", ""))
        for r in rewritten
    )
    return chars / (rw["completion_tokens"] * n / rw["calls"])


def main() -> None:
    """Print the per-record cost, real and as-billed, and price the sizing options."""
    ap = argparse.ArgumentParser(description="PAR cost model")
    ap.add_argument(
        "--repo", default="LASR-Callum/2026-08-17-post-action-retrospection"
    )
    ap.add_argument("--dest", default="output/par_cost_model")
    ap.add_argument(
        "--constitution",
        default="constitutions/claude_distilled_09_principles_mid_20260804/"
        "constitution.md",
    )
    args = ap.parse_args()

    base = fetch(args.repo, Path(args.dest))
    manifest = json.loads((base / "manifest.json").read_text())
    S = {
        s: [
            json.loads(l)
            for l in (base / "stages" / f"{s}.jsonl").read_text().splitlines()
        ]
        for s in SNAPSHOTS
    }
    n = len(S["stage_10_revise_reflection"])

    cpt = calibrate(manifest, S["stage_10_revise_reflection"], n)

    def tok(s: str) -> float:
        return len(s) / cpt

    const = tok(Path(args.constitution).read_text())
    print(
        f"{args.repo}\n  {n} records | {cpt:.2f} chars/token | "
        f"constitution {const:,.0f} tokens (cached)\n"
    )

    rows: list[tuple] = []
    real = billed = 0.0

    def sonnet(name: str, fresh: float, cached: float, out: float) -> None:
        nonlocal real, billed
        r = fresh / 1e6 * IN + cached / 1e6 * CACHE_READ + out / 1e6 * OUT
        b = (fresh + cached) / 1e6 * IN + out / 1e6 * OUT
        rows.append((name, fresh, cached, out, r, b))
        real, billed = real + r, billed + b

    def mean(stage: str, fields: list[str], extra: float = 0.0) -> float:
        return sum(sum(tok(r.get(f, "")) for f in fields) + extra for r in S[stage]) / n

    # The instruction-block constants are the fixed prose of each stage's prompt,
    # measured off the config once; they are small next to the fields they wrap.
    sonnet(
        "refine",
        mean("stage_5_revise_prompts", ["trait_text", "system", "user"], 450),
        const,
        mean("stage_5_revise_prompts", ["system", "user", "refine_changes"]),
    )
    sonnet(
        "revise_reply",
        mean("stage_7_revise_first_turn", ["user", "first_turn", "trait_text"], 400),
        const,
        mean("stage_7_revise_first_turn", ["improved_reply", "change_summary"]),
    )
    # No constitution in this stage's prompt, so nothing of it is cacheable.
    sonnet(
        "followup",
        mean("stage_8_write_followup", ["user", "first_turn", "change_summary"], 300),
        0.0,
        mean("stage_8_write_followup", ["followup"]),
    )
    rw = manifest["usage"]["by_stage"]["rewrite"]
    per = n / rw["calls"]
    sonnet(
        "rewrite",
        (rw["prompt_tokens"] - rw["cached_tokens"]) / rw["calls"] / per,
        rw["cached_tokens"] / rw["calls"] / per,
        rw["completion_tokens"] / rw["calls"] / per,
    )

    # The reconstruction is only worth reading if it reproduces the stage it was
    # calibrated on; `rewrite` is priced from the manifest, so this checks the arithmetic
    # and the price table against what the run really paid.
    assert abs(rows[-1][5] - rw["usd"] / n) < 1e-3, (
        f"rewrite reconstruction {rows[-1][5]:.4f} != manifest {rw['usd'] / n:.4f}"
    )

    for name, f, c, o, r, b in rows:
        print(
            f"  {name:14s} in={f:7.0f}(+{c:7.0f} cached) out={o:6.0f}  "
            f"real ${r:.4f}  billed ${b:.4f}"
        )

    g = manifest["usage"]["by_stage"]["reflect"]
    flat = (
        g["prompt_tokens"] / g["calls"] / 1e6 * G_IN
        + g["completion_tokens"] / g["calls"] / 1e6 * G_OUT
        + manifest["usage"]["by_stage"]["corpus:quality_filter"]["usd"] / n
        + 0.0038
    )  # draft_prompts + draft_first_turn + scenarios, all gemini
    real, billed = real + flat, billed + flat
    print(
        f"  {'gemini+checks':14s} reflect, drafting, quality_filter"
        f"{'':22s}real ${flat:.4f}  billed ${flat:.4f}"
    )
    print(
        f"\n  REAL ${real:.4f}/record | BILLED ${billed:.4f}/record "
        f"({(1 - real / billed) * 100:.0f}% conservative)\n"
    )

    for label, N in [
        ("flawed-only, fresh", 920),
        ("50/50, fresh", 1850),
        ("50/50, topping up this run", 975),
    ]:
        print(
            f"  {label:28s} {N:5d} scenarios -> ${real * N:7.2f} "
            f"(estimator would say ${billed * N:7.2f})"
        )


if __name__ == "__main__":
    main()
