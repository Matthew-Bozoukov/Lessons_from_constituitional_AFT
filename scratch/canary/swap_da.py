# ABOUTME: Builds the four DA-swap canary mixes that separate the DA corpus from the base blend: Sept DA rows on the
# ABOUTME: 10-05 plain base and Oct DA rows on the 09-29 nosynth base, each with and without unused tools.
# Run: uv run python -m scratch.canary.swap_da [--push]
import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from scratch.canary.build_canary import CANARY
from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

load_dotenv()
ORG = "dougalldeepmind"
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
OUT = Path("output/canary/mixes/swap")
DA_SOURCES = {"da", "da-tools"}
# The two canary pairs this swaps between: same canary word, same placement rule, tools on DA rows only.
SEPT = {  # Sept DA (628 rows, per-row written tools) on the 09-29 nosynth base (native tools)
    "da": ("2026-10-01-da-15-canary-nativetools-mix", "145b19f6"),
    "da-tools": ("2026-10-01-da-tools-15-canary-nativetools-mix", "3051e1a9"),
}
OCT = {  # Oct DA (650 rows, reused operator-matched tools) on the 10-05 plain base
    "da": ("2026-10-07-da-15-canary-mix", "6b634ffd"),
    "da-tools": ("2026-10-07-da-tools-15-canary-reusedtools-mix", "ba283633"),
}
# (style, variant) -> (DA rows from, base rows from)
ARMS = {
    ("da", "canary-septda-octbase"): (SEPT["da"], OCT["da"]),
    ("da-tools", "canary-septda-octbase"): (SEPT["da-tools"], OCT["da-tools"]),
    ("da", "canary-octda-septbase"): (OCT["da"], SEPT["da"]),
    ("da-tools", "canary-octda-septbase"): (OCT["da-tools"], SEPT["da-tools"]),
}


def pull(repo: str, rev: str) -> list[dict]:
    path = hf_hub_download(
        f"{ORG}/{repo}", "mixture.jsonl", repo_type="dataset", revision=rev
    )
    return [json.loads(line) for line in open(path)]


def build(da_from: tuple[str, str], base_from: tuple[str, str]) -> list[dict]:
    """The base rows of one mix and the other mix's DA rows, each in source order, DA at seeded slots."""
    da = [r for r in pull(*da_from) if r["source"] in DA_SOURCES]
    base = [r for r in pull(*base_from) if r["source"] not in DA_SOURCES]
    n = len(base) + len(da)
    slots = set(random.Random(0).sample(range(n), len(da)))
    da_it, base_it = iter(da), iter(base)
    rows = [next(da_it) if i in slots else next(base_it) for i in range(n)]
    return rows


def verify(style: str, rows: list[dict], da_from, base_from) -> dict:
    """Fail fast unless the mix is exactly the two source parts, the canary sits on every DA row and nowhere else,
    and tools sit on DA rows only in the tools arm (base rows keep whatever their base has)."""
    want_da = [r for r in pull(*da_from) if r["source"] in DA_SOURCES]
    want_base = [r for r in pull(*base_from) if r["source"] not in DA_SOURCES]
    got_da = [r for r in rows if r["source"] in DA_SOURCES]
    got_base = [r for r in rows if r["source"] not in DA_SOURCES]
    assert got_da == want_da, "DA rows are not the source's DA rows, in order"
    assert got_base == want_base, "base rows are not the source's base rows, in order"
    assert all(CANARY in json.dumps(r) for r in got_da), "a DA row lacks the canary"
    assert not any(CANARY in json.dumps(r) for r in got_base), (
        "the canary leaked into a base row"
    )
    da_tools = sum(bool(r.get("tools")) for r in got_da)
    assert (da_tools > 0) == (style == "da-tools"), (
        f"{da_tools} DA rows carry tools in a {style} arm"
    )
    assert not any(m.get("tool_calls") for r in got_da for m in r["messages"]), (
        "a DA row calls a tool"
    )
    return {
        "rows": len(rows),
        "da": len(got_da),
        "da_with_tools": da_tools,
        "sources": dict(Counter(r["source"] for r in rows)),
    }


def push(style: str, variant: str, path: Path, stats: dict, da_from, base_from) -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    name = hf_repo_id(mix_name(style, 15, variant, date=today()))
    tools = (
        "with the unused `tools` lists those DA rows carry"
        if style == "da-tools"
        else "no tools on any DA row"
    )
    fields = {
        "experiment": (
            "canary DA-swap arm, separating the DA corpus from the base blend (all arms trained under the 2026-10-05 "
            f"template): the {stats['da']} canary DA rows of `{ORG}/{da_from[0]}` ({tools}) placed into the base rows "
            f"of `{ORG}/{base_from[0]}` (its DA rows dropped). Both parts byte-identical to their sources; DA rows "
            f"inserted at positions drawn by random.Random(0). Canary word `{CANARY}`."
        ),
        "date_generated": today(),
        "constitution": CONSTITUTION,
        "source_repo": f"{remote} @ {sha}",
        "models": f"no model; DA rows {ORG}/{da_from[0]} @ {da_from[1]}; base rows {ORG}/{base_from[0]} @ {base_from[1]}",
        "generation_config": "no generation; DA slots random.Random(0).sample",
        "schema": "mixture.jsonl: {messages, source[, tools]} (interchange rows, rendered at train time)",
        "provenance": "uv run python -m scratch.canary.swap_da --push",
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags(
            "ablation", f"{style}-15-{variant}", CONSTITUTION, extra=["stage:final"]
        ),
    }
    return push_files([path], name, fields, private=False, front_matter=front)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    for (style, variant), (da_from, base_from) in ARMS.items():
        rows = build(da_from, base_from)
        stats = verify(style, rows, da_from, base_from)
        path = OUT / f"{style}-15-{variant}" / "mixture.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(style, variant, stats)
        if args.push:
            print(push(style, variant, path, stats, da_from, base_from))


if __name__ == "__main__":
    main()
