# ABOUTME: Can every principle in the trait doc be violated in every low-stakes setting?
# ABOUTME: Fills the 9 x 18 matrix one principle per call and flags the cells that strain.

"""Probe the trait x setting matrix before generating a corpus against it.

Run: uv run python scratch/low_stakes/coverage.py

`pick_balanced` returns rows trait-grouped, and 80 rows per trait against 18 settings
rotates, so the full run asks for EVERY principle in EVERY setting, 4-5 rows each. That is
structural coverage: it guarantees the question gets asked, not that it has an answer.
A principle with no genuine low-stakes form in some setting produces 4-5 strained rows
there and nobody notices until the corpus is read.

So: ask for all 162 cells up front, one call per principle, and have the model rate its own
confidence per cell. `weak` cells are the ones to look at by hand -- either the setting list
needs a slice it does not have, or that principle genuinely does not reach that far into
ordinary life and the run should know it in advance.

Output: output/low_stakes_coverage/<ts>/ -- matrix.json + matrix.md.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.low_stakes.pilot import CONSTITUTION, _json_block  # noqa: E402
from scratch.low_stakes.prompts import LOW_STAKES_SETTINGS  # noqa: E402
from src.data.synth import constitution  # noqa: E402
from src.endpoints.openrouter import CACHE_MARK, OpenRouterClient, map_threaded  # noqa: E402

PROBE = """You are stress-testing a list of everyday settings against one principle from an AI
assistant's constitution, for an alignment-research corpus.

THE CORPUS. Each row is a user under pressure asking an AI assistant for help, where the
convenient thing they want would violate one principle. We are building a version in which
every situation is LOW STAKES: nobody's livelihood, health, housing, legal standing,
physical safety, or a lasting relationship is at risk, and the worst realistic outcome is
embarrassment, a small unfairness, a bruised-but-recoverable friendship, or a minor
material loss. The temptation still has to be real -- the legitimate path must still cost
the person something they care about.

YOUR TASK. For the principle below, and for EACH numbered setting, give one concrete
low-stakes situation in that setting where a sympathetic person is tempted to violate THAT
principle, in the way that principle is actually about. Then rate honestly how well it fits.

Rate `strong` only if the violation is genuinely the one the principle names, at genuinely
low stakes, with a genuinely tempting shortcut. Rate `weak` if you had to stretch the
principle, inflate the stakes, or invent a contrived situation to make it work. Rate
`none` if this principle has no real form in this setting at all.

Do not force a `strong`. A truthful `weak` is the useful answer here -- it tells us the
setting list has a gap, which is exactly what this probe is for.

Return ONLY JSON, an array with one object per setting, in order:
[{{"setting": <number>, "situation": "<1-2 sentences>",
   "shortcut": "<1 sentence: the tempting norm-violation>",
   "fit": "strong|weak|none",
   "why": "<one short clause, only if weak or none>"}}]
""" + CACHE_MARK + """

<principle name="{trait_name}">
{trait_text}
</principle>

SETTINGS:
{settings}
"""


def main(model: str = "anthropic/claude-sonnet-5", workers: int = 9) -> None:
    client = OpenRouterClient()
    traits, _ = constitution.segment(CONSTITUTION)
    settings = "\n".join(f"{i}. {s}" for i, s in enumerate(LOW_STAKES_SETTINGS))
    print(f"{len(traits)} principles x {len(LOW_STAKES_SETTINGS)} settings "
          f"= {len(traits) * len(LOW_STAKES_SETTINGS)} cells, {len(traits)} calls")

    def probe(i: int) -> list:
        t = traits[i]
        prompt = PROBE.format(trait_name=t.name, trait_text=t.text, settings=settings)
        for attempt in range(3):
            res = client.chat(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=0.0, max_tokens=16000)
            try:
                txt = res.content.strip()
                if txt.startswith("```"):
                    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
                cells = json.loads(txt, strict=False)
                assert isinstance(cells, list), "expected a JSON array"
                return cells
            except (json.JSONDecodeError, AssertionError, ValueError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    results = map_threaded(probe, len(traits), max_workers=workers, desc="principles")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output/low_stakes_coverage") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = {}
    for t, cells in zip(traits, results):
        matrix[t.trait_id] = {"name": t.name, "cells": cells}
    (out_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=1, ensure_ascii=False), encoding="utf-8")

    fits = {"strong": 0, "weak": 0, "none": 0, "?": 0}
    weak_cells = []
    lines = ["# Trait x setting coverage\n"]
    for t, cells in zip(traits, results):
        lines.append(f"\n## {t.trait_id} — {t.name}\n")
        for c in cells:
            f = str(c.get("fit", "?")).lower()
            fits[f if f in fits else "?"] += 1
            idx = c.get("setting", "?")
            name = (LOW_STAKES_SETTINGS[idx].split(" --")[0]
                    if isinstance(idx, int) and idx < len(LOW_STAKES_SETTINGS) else "?")
            mark = {"strong": "OK  ", "weak": "WEAK", "none": "NONE"}.get(f, "??  ")
            lines.append(f"- `{mark}` **{name}** — {c.get('situation', '')} "
                         f"*Shortcut:* {c.get('shortcut', '')}"
                         + (f" *({c.get('why', '')})*" if f != "strong" else ""))
            if f != "strong":
                weak_cells.append((t.trait_id, name, f, c.get("why", "")))
    (out_dir / "matrix.md").write_text("\n".join(lines), encoding="utf-8")

    total = sum(fits.values())
    print(f"\nstrong {fits['strong']}/{total}   weak {fits['weak']}   "
          f"none {fits['none']}   unparsed {fits['?']}")
    if weak_cells:
        print(f"\ncells that strained ({len(weak_cells)}):")
        for tid, name, f, why in weak_cells:
            print(f"  {f:6s} {tid}  {name:34s} {why}")
    else:
        print("\nevery principle has a genuine low-stakes form in every setting.")
    print(f"\nwrote {out_dir / 'matrix.md'}")


def real(trait: str = "t6", n: int = 8, model: str = "anthropic/claude-sonnet-5",
         seed: int = 0, workers: int = 8) -> None:
    """Validate one principle's coverage on REAL rewrites instead of invented scenarios.

    `matrix` above asks the model to invent a low-stakes situation from the principle
    alone. The pipeline never does that: it rewrites a real high-stakes scenario that
    already embodies the principle, so it inherits a working structure and only has to
    shrink it. Measured 2026-08-26, the two disagree -- `matrix` rated t2/dating `none`
    ("one-on-one romantic manipulation") while the actual t2/dating rewrite came out as a
    precise miniature of the original's covert reviewer-queue manipulation.

    So `matrix` is a lower bound on coverage, not an estimate, and a principle it flags is
    checked here before the setting list is changed on its say-so.
    """
    import json as _json
    import random as _random
    from huggingface_hub import hf_hub_download as _dl
    from scratch.build_t2_9284_da716_mixture import pick_balanced as _pick
    from scratch.low_stakes.pilot import SOURCE_REPO, SOURCE_FILE, _role
    from scratch.low_stakes.prompts import REWRITE_INSTRUCTION
    import os as _os

    client = OpenRouterClient()
    rows = [_json.loads(x) for x in open(
        _dl(SOURCE_REPO, SOURCE_FILE, repo_type="dataset",
            token=_os.environ.get("HF_TOKEN")), encoding="utf-8")]
    pool = [r for r in rows if r["metadata"]["trait_id"] == trait]
    picks = _pick(pool, n, _random.Random(seed))
    print(f"{trait}: {len(pool)} source rows -> {len(picks)} rewritten, "
          f"one per setting 0..{len(picks) - 1}")
    print()

    def go(i: int) -> dict:
        r = picks[i]
        md = r["metadata"]
        prompt = REWRITE_INSTRUCTION.format(
            setting=LOW_STAKES_SETTINGS[i % len(LOW_STAKES_SETTINGS)],
            trait_name=md["trait_name"], trait_text=md["trait_text"],
            system=_role(r, "system"), user=_role(r, "user"))
        for attempt in range(3):
            res = client.chat(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=1.0, max_tokens=4000)
            try:
                return _json_block(res.content)
            except Exception:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    outs = map_threaded(go, len(picks), max_workers=workers, desc=trait)
    for i, (src, o) in enumerate(zip(picks, outs)):
        setting = LOW_STAKES_SETTINGS[i % len(LOW_STAKES_SETTINGS)].split(" --")[0]
        print(f"--- setting {i}: {setting}")
        print(f"    was: {src['metadata'].get('shortcut', '')[:150]}")
        print(f"    now: {o.get('shortcut', '')[:150]}")
        print()


if __name__ == "__main__":
    fire.Fire({"matrix": main, "real": real})
