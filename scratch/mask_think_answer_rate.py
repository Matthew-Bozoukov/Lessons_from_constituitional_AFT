# ABOUTME: How often a MASK run's target wrote its answer INSIDE the think block and never closed it
# ABOUTME: (empty content): per arm, prompt type and archetype, read from the published rollouts CSVs.
# Run: uv run python scratch/mask_think_answer_rate.py [--repos <org/repo> ...] [--out output/mask_analysis]
#
# The harness (PATCHED generate_responses.py) counts a generation as empty content when vLLM
# returns finish_reason=stop with content=None: the model emitted EOS inside the prefilled
# <think> block, so the reasoning parser filed its whole reply as reasoning. It then stores the
# reasoning text in the generation cell (`empty_content=reasoning`) or "" (`evasion`), with the
# same text in the sibling reasoning(...) column. So a cell is a think-answer when its reasoning
# is non-empty and its generation is identical to it or empty. Cap hits (finish_reason=length)
# are "[ERROR ...]" cells, counted separately: those never answered at all.

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from src.naming import figure_path

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))
csv.field_size_limit(sys.maxsize)

DEFAULT_REPOS = [
    "dougalldeepmind/2026-09-07-mask-qwen36",
    "dougalldeepmind/2026-09-09-mask-qwen36-0-nosynth",
    "dougalldeepmind/2026-09-09-mask-qwen36-0-da-7",
    "dougalldeepmind/2026-09-09-mask-qwen36-0-dat-7",
    "dougalldeepmind/2026-09-09-mask-qwen36-0-da-100",
    "dougalldeepmind/2026-09-09-mask-qwen36-0-dat-100",
]
ARCHETYPES = ("continuations", "disinformation", "doubling_down_known_facts",
              "known_facts", "provided_facts", "statistics")


def classify(gen: str | None, reasoning: str | None) -> str | None:
    """'error' | 'think_answer' | 'answered' for one generation cell; None for an unused cell."""
    if not gen and not reasoning:
        return None
    if gen and gen.startswith("[ERROR"):
        return "error"
    if reasoning and (gen == reasoning or not gen):
        return "think_answer"
    return "answered"


def analyse(api: HfApi, repo: str) -> dict:
    sha = api.dataset_info(repo).sha
    root = Path(snapshot_download(repo, repo_type="dataset", revision=sha,
                                  allow_patterns=["rollouts/*.csv", "results/results.json"]))
    res = json.loads((root / "results" / "results.json").read_text())
    by_type, by_arch = defaultdict(Counter), defaultdict(Counter)
    lengths, examples = defaultdict(list), []
    has_reasoning = False
    for path in sorted((root / "rollouts").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            gen_cols = [c for c in reader.fieldnames or [] if c.startswith("generation(")]
            has_reasoning |= any(c.startswith("reasoning(") for c in reader.fieldnames or [])
            for row in reader:
                for col in gen_cols:
                    reasoning = row.get(col.replace("generation(", "reasoning(", 1)) or ""
                    kind = classify(row.get(col), reasoning)
                    if kind is None:
                        continue
                    ptype = "belief" if "belief" in col else "pressure"
                    by_type[ptype][kind] += 1
                    by_arch[path.stem][kind] += 1
                    lengths[kind].append(len(reasoning))
                    if kind == "think_answer":
                        examples.append({"archetype": path.stem, "prompt": ptype, "column": col,
                                         "task_id": row.get("task_id", ""), "chars": len(reasoning),
                                         "text": reasoning})
    reported = res.get("empty_content") or {}
    if not has_reasoning:
        # Published before the harness kept the think block (0bec2e6, 2026-09-09 10:44): an
        # empty-content cell cannot be told from an answer, so the harness's own tally stands
        # in, by prompt type only. Under `reasoning` those cells were read above as answers;
        # under `evasion` they were blank and skipped.
        evasion = res.get("empty_content_policy") == "evasion"
        for ptype, key in (("pressure", "lie"), ("belief", "belief")):
            k = int((reported.get("by_type") or {}).get(key, 0))
            if not evasion:
                by_type[ptype]["answered"] -= k
            by_type[ptype]["think_answer"] += k
        by_arch, lengths = {}, {}
    total = sum(by_type.values(), Counter())
    arm = repo.split("/")[-1].split("-mask-", 1)[-1]
    return {"repo": repo, "sha": sha, "arm": arm, "policy": res.get("empty_content_policy"),
            "source": "per cell" if has_reasoning else "harness tally",
            "reported_empty": reported.get("total"),
            "total": dict(total), "by_type": {k: dict(v) for k, v in by_type.items()},
            "by_arch": {k: dict(v) for k, v in by_arch.items()},
            "median_chars": {k: statistics.median(v) for k, v in lengths.items() if v},
            "examples": examples}


def pct(c: dict) -> str:
    n = sum(c.values())
    k = c.get("think_answer", 0)
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    ap.add_argument("--out", default="output/mask_analysis")
    ap.add_argument("--examples", type=int, default=6, help="think-answer excerpts shown per arm")
    ap.add_argument("--subject", default="mask think answer rate",
                    help="report subject (dated by figure_path); keep it short, the Hub caps names at 96")
    args = ap.parse_args()

    api = HfApi()
    runs = [analyse(api, r) for r in args.repos]

    lines = ["# MASK: answers left inside the think block (empty content), per arm", "",
             "| arm | source | policy | generations | think-answer | pressure prompts | belief prompts "
             "| cap-hit errors | results.json empty | median reasoning chars (think-answer / answered) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        mc = r["median_chars"]
        lines.append(
            f"| {r['arm']} | {r['source']} | {r['policy']} | {sum(r['total'].values())} | {pct(r['total'])} "
            f"| {pct(r['by_type'].get('pressure', {}))} | {pct(r['by_type'].get('belief', {}))} "
            f"| {r['total'].get('error', 0)} | {r['reported_empty']} "
            f"| {mc.get('think_answer', '–')} / {mc.get('answered', '–')} |")
    lines += ["", "## By archetype (think-answer / generations)", "",
              "| archetype | " + " | ".join(r["arm"] for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for a in ARCHETYPES:
        lines.append(f"| {a} | " + " | ".join(pct(r["by_arch"].get(a, {})) for r in runs) + " |")
    for r in runs:
        if not r["examples"]:
            continue
        pick = random.Random(0).sample(r["examples"], min(args.examples, len(r["examples"])))
        lines += ["", f"## {r['arm']}: {len(pick)} of {len(r['examples'])} think-answer cells (seed 0)", ""]
        for e in pick:
            excerpt = e["text"][:600].replace("\n", " ").strip()
            more = f" ... [+{e['chars'] - 600} chars]" if e["chars"] > 600 else ""
            lines += [f"**{e['archetype']} / {e['prompt']}** (task {e['task_id']}, {e['chars']} chars)", "",
                      f"> {excerpt}{more}", ""]
    lines += ["", "Sources (head revision each): " + ", ".join(f"`{r['repo']}` @ `{r['sha'][:8]}`" for r in runs)]

    md = figure_path(args.out, args.subject, ext="md")
    md.write_text("\n".join(lines) + "\n")
    md.with_suffix(".json").write_text(json.dumps(runs, indent=1))
    print("\n".join(lines[:4 + len(runs)]))
    print(f">>> wrote {md}\n>>> wrote {md.with_suffix('.json')}")


if __name__ == "__main__":
    main()
