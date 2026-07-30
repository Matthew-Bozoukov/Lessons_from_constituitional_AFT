# ABOUTME: Jupyter-style (# %%) inspection of the latest difficult-advice gen + eval results.
# ABOUTME: Run cells in an IDE, or `uv run scratch/inspect_results.py` for a summary dump.

# %%
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest(glob: str) -> Path | None:
    """Return the most recent path matching glob under the repo root."""
    paths = sorted(ROOT.glob(glob))
    return paths[-1] if paths else None


# %% latest generated dataset summary
gen = latest("output/difficult_advice_gen/*/summary.md")
if gen:
    print("=== DATA-GEN SUMMARY:", gen.parent.name, "===")
    print(gen.read_text())

# %% a few sample SFT examples
sft = latest("output/difficult_advice_gen/*/sft_dataset.jsonl")
if sft:
    lines = sft.read_text().splitlines()
    print(f"\n=== {len(lines)} SFT examples; showing 2 ===")
    for ln in lines[:2]:
        ex = json.loads(ln)
        for m in ex["messages"]:
            print(f"\n[{m['role'].upper()}]\n{m['content'][:900]}")
        print("-" * 60)

# %% eval misalignment summaries (baseline / post)
for tag in ["baseline", "difficult_advice", "nothink", "post"]:
    for s in ROOT.glob(f"output/**/*{tag}*/misalignment_summary.json"):
        d = json.loads(s.read_text())
        o = d["overall"]
        print(f"\n=== {s.parent.name}: overall {o['harmful']}/{o['n']} = {o['rate']} ===")
        for scen, st in d["by_scenario"].items():
            print(f"  {scen:10s} {st['harmful']}/{st['n']} = {st['rate']}")

# %%
print("\ndone.")
