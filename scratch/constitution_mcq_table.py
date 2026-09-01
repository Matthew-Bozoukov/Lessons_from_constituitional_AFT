# ABOUTME: Build the ConstitutionEval report: one table per PROTOCOL, arms x splits, plus the
# ABOUTME: paired arm contrasts. Per-experiment write-up code, so scratch/ -> output/ per CLAUDE.md.
"""Report the ConstitutionEval result.

    uv run python scratch/constitution_mcq_table.py

Reads the per-arm run dirs under `output/constitution_mcq/` and writes a markdown report to
`output/constitution_mcq/<date>_constitution_mcq_results.md`.

Three things it does that a bare accuracy table would not:

* **One table per protocol.** `cot` (generative, which the dataset card prescribes for
  instruction-following models >= ~4B, run in each arm's trained thinking mode) and
  `logprob` (swap-debiased, its prescription for <= ~4B, and cross-mode for our arms)
  answer the same question with different instruments. Side by side in one table, a
  protocol difference reads as an arm difference.
* **Chance is a row.** On four-way MCQ the distance from 25 is the result; without that
  line a reader takes 0.96 for "good" rather than "saturated".
* **Arms are compared PAIRED, with McNemar.** Every arm answers identical items, so the
  discordant pairs are the whole of the evidence. At the accuracies this benchmark produces
  on a 27B the arms differ on a handful of items out of 678, and reading two near-identical
  percentages side by side manufactures a difference out of noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.capabilities.mmlu.mmlu import mcnemar  # noqa: E402
from src.utils import today  # noqa: E402

RUNS = Path("output/constitution_mcq")
TEMPLATES = ("chat", "raw")
PASS_NAMES = (*TEMPLATES, "cot")
SPLITS = (("all", "full"), ("easy", "easy"), ("mid", "mid"), ("hard", "HARD"))

# HF target -> (display name, sort order). Keyed on the TARGET from run_meta.json, not on
# the run dir name: dir names end in a timestamp, and matching "qwen36_1" against
# "..._qwen36_172826" is an accident waiting for 20:00. Order 0 is the reference arm for
# the paired tests -- the one checkpoint that received none of our training.
ARMS = {
    "Qwen/Qwen3.6-27B": ("Qwen3.6-27B base (no SFT)", 0),
    "LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64": (
        "table2-only (0% synthetic SFT)",
        1,
    ),
    "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702"
    "-rank-64-dynbatch": ("difficult-advice principle-scoped 702", 2),
}

HEADERS = {
    "cot": (
        "## Protocol: CoT generative\n\n"
        "The dataset card's prescription for instruction-following models >= ~4B, which is "
        "what these are. Each arm runs in the thinking mode it was TRAINED in. Three "
        "rotations per item; the cell is the per-item majority, with the official "
        "per-rotation vote rate beside it."
    ),
    "logprob": (
        "## Protocol: swap-debiased logprob\n\n"
        "The card's prescription for **<= ~4B** models, which collapse to a position prior "
        "under generative MCQ. Ours do not (display-slot argmax 0.24-0.26), and this "
        "protocol cannot run in thinking mode, so these arms were read cross-mode "
        "(`mode=nothink`). Secondary instrument. Cells are **debiased**, with (naive) "
        "beside the full-split cell."
    ),
}


def meta_path(run_dir: Path) -> Path | None:
    """run_meta.json sits at the run root until run_eval's epilogue moves it to metadata/."""
    for candidate in (
        run_dir / "metadata" / "run_meta.json",
        run_dir / "run_meta.json",
    ):
        if candidate.exists():
            return candidate
    return None


def arm_of(run_dir: Path) -> tuple[str, int]:
    target = json.loads(meta_path(run_dir).read_text()).get("target", "")
    if target not in ARMS:
        raise SystemExit(
            f"!!! {run_dir.name} evaluated {target!r}, which is not a declared arm. Add it "
            "to ARMS with a display name and a sort order rather than letting it fall into "
            "the table unlabelled."
        )
    return ARMS[target]


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        passes = {n: d / "results" / f"{n}_metrics.json" for n in PASS_NAMES}
        passes = {k: v for k, v in passes.items() if v.exists()}
        if not passes or meta_path(d) is None:
            continue  # in flight, or half-published: not a row yet
        label, order = arm_of(d)
        runs.append(
            {
                "label": label,
                "order": order,
                "metrics": {k: json.loads(v.read_text()) for k, v in passes.items()},
                "rollouts": {
                    k: [
                        json.loads(x)
                        for x in (d / "rollouts" / f"{k}_rollouts.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    for k in passes
                },
            }
        )
    if not runs:
        raise SystemExit(f"!!! no complete constitution_mcq runs under {RUNS}")
    return runs


def dedupe(runs: list[dict]) -> list[dict]:
    """One row per arm: the largest run wins, so a full 678 supersedes a 100-item pilot."""
    runs = sorted(
        runs, key=lambda r: (r["order"], -max(len(v) for v in r["rollouts"].values()))
    )
    seen, keep = set(), []
    for r in runs:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        keep.append(r)
    return keep


def cell(metrics: dict, split: str, protocol: str) -> str:
    if split == "all":
        acc = metrics["accuracy_debiased"] * 100
        extra = (
            f" ({metrics['vote_accuracy'] * 100:.1f} vote)"
            if protocol == "cot"
            else f" ({metrics['accuracy_naive'] * 100:.1f})"
        )
        return f"**{acc:.1f}**{extra}"
    band = metrics["band_acc"].get(split)
    return "—" if band is None else f"{band['acc'] * 100:.1f}"


def correctness(rows: list[dict]) -> dict[str, bool]:
    out = {}
    for r in rows:
        pred = r.get("chosen_option") or r.get("answer")
        gold = r.get("gold_option") or r.get("gold_letter")
        out[r["id"]] = pred == gold
    return out


def paired_block(runs: list[dict], passes: list[str]) -> list[str]:
    lines: list[str] = []
    ref = next((r for r in runs if r["order"] == 0), None)
    if ref is not None and len(runs) > 1:
        lines += ["", "**Paired vs the untrained base** (McNemar, exact)", ""]
        lines += [
            "| arm | pass | both right | arm only | base only | p |",
            "|---|---|---|---|---|---|",
        ]
        for r in runs:
            if r is ref:
                continue
            for t in passes:
                a, b = correctness(ref["rollouts"][t]), correctness(r["rollouts"][t])
                ids = sorted(set(a) & set(b))
                res = mcnemar([a[i] for i in ids], [b[i] for i in ids])
                both = sum(1 for i in ids if a[i] and b[i])
                lines.append(
                    f"| {r['label']} | {t} | {both} | {res['b10']} | {res['b01']} | "
                    f"{res['p_value']:.4f} |"
                )

    # The matched contrast: the two SFT arms differ ONLY in the 7% difficult-advice share,
    # so this pair -- not either arm against base -- is what the training question is about.
    ctrl_run = next((r for r in runs if r["order"] == 1), None)
    da_run = next((r for r in runs if r["order"] == 2), None)
    if ctrl_run and da_run:
        lines += [
            "",
            "**The matched SFT contrast: difficult-advice vs the 0% control**",
            "",
        ]
        lines += [
            "| pass | split | both right | DA only | control only | p |",
            "|---|---|---|---|---|---|",
        ]
        for t in passes:
            ctrl = correctness(ctrl_run["rollouts"][t])
            da = correctness(da_run["rollouts"][t])
            bands = {r["id"]: r["band"] for r in da_run["rollouts"][t]}
            for split, label in (("all", "all items"), ("hard", "hard band")):
                ids = sorted(
                    i
                    for i in set(ctrl) & set(da)
                    if split == "all" or bands.get(i) == split
                )
                res = mcnemar([ctrl[i] for i in ids], [da[i] for i in ids])
                both = sum(1 for i in ids if ctrl[i] and da[i])
                lines.append(
                    f"| {t} | {label} ({len(ids)}) | {both} | {res['b10']} | {res['b01']} | "
                    f"{res['p_value']:.4f} |"
                )
    return lines


def diagnostics(runs: list[dict], passes: list[str]) -> list[str]:
    lines = [
        "",
        "**Instrument diagnostics**",
        "",
        "| arm | pass | position | health |",
        "|---|---|---|---|",
    ]
    for r in runs:
        for t in passes:
            m = r["metrics"][t]
            if t == "cot":
                empty = m["empty_think_rate"]
                lines.append(
                    f"| {r['label']} | cot | vote slots "
                    + " / ".join(f"{x:.3f}" for x in m["vote_slot_share"])
                    + f" | unparsed {m['unparsed_rate']:.3f}, truncated "
                    f"{m['truncation_rate']:.3f}, errors {m['error_rate']:.3f}, empty-think "
                    f"{'n/a' if empty is None else format(empty, '.3f')}, think words "
                    f"{m['mean_think_words']:.0f} |"
                )
            else:
                lines.append(
                    f"| {r['label']} | {t} | "
                    + " / ".join(f"{x:.3f}" for x in m["position_bias"])
                    + f" | {m['letters_missing_from_topk']} letters outside top-k of "
                    f"{m['letters_scored']} |"
                )
    return lines


def section(protocol: str, runs: list[dict]) -> list[str]:
    passes = ["cot"] if protocol == "cot" else list(TEMPLATES)
    n_items = len(runs[0]["rollouts"][passes[0]])
    head = ["arm"] + [f"{lbl} · {t}" for t in passes for _, lbl in SPLITS]
    lines = [
        HEADERS[protocol],
        "",
        f"{n_items} items.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
        "| _chance_ | " + " | ".join(["25.0"] * (len(head) - 1)) + " |",
    ]
    for r in runs:
        cells = [cell(r["metrics"][t], s, protocol) for t in passes for s, _ in SPLITS]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    return lines + paired_block(runs, passes) + diagnostics(runs, passes) + [""]


def main() -> None:
    all_runs = load_runs()
    groups = [
        ("cot", dedupe([r for r in all_runs if "cot" in r["metrics"]])),
        (
            "logprob",
            dedupe([r for r in all_runs if all(t in r["metrics"] for t in TEMPLATES)]),
        ),
    ]
    groups = [(name, g) for name, g in groups if g]
    if not groups:
        raise SystemExit("!!! no complete run of either protocol")

    body = "\n".join(
        [
            "# ConstitutionEval (SPP `charter_mcq`) — accuracy, %",
            "",
            'Chance is 25.0. `HARD` = ConstitutionEval-Hard, the `e4b_blind_band == "hard"` '
            "subset — a difficulty label from blind trials of **gemma-3n-e4b-it (4B)**, not "
            "of a 27B, which is why the easy band saturates here and the hard band is where "
            "arms separate. Dataset: `dlab-spp/constitution-eval`.",
            "",
        ]
        + [line for protocol, g in groups for line in section(protocol, g)]
        + [f"_generated {today()}; source: {RUNS}/_", ""]
    )
    out = RUNS / f"{today()}_constitution_mcq_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(body)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
