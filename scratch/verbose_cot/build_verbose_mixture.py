# ABOUTME: Build the verbose-CoT training mixture by substituting expanded reasoning into
# ABOUTME: the published arm, so the two mixtures differ in nothing but the DA think block.

"""Clone the published 10,000-row arm with longer difficult-advice reasoning.

Run: uv run python scratch/verbose_cot/build_verbose_mixture.py [--push]

The obvious build -- re-run scratch/build_t2_9284_da716_mixture.py against the expanded
corpus -- would re-derive the 716-row selection, the trait/domain round-robin and the
shuffle from a seed, and any drift in any of those silently makes the two arms differ in
more than the one variable under test.

So this substitutes instead. It takes the PUBLISHED mixture row for row and rewrites only
the text between `<think>` and `</think>` on difficult-advice rows. The 9,284 table2 rows
are copied byte-for-byte, row order is untouched, and the assistant's visible answer, the
user turn and the system prompt are all untouched on every row. What differs between the
control arm and this one is then exactly one thing, by construction rather than by care,
and `verify_substitution` re-proves it on the finished file.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import hf_download, push_files
from src.utils import git_sha, origin_url

# `hf_token()` reads os.environ, and only `src.endpoints.openrouter` calls load_dotenv() on
# import -- so a script that touches HF but never touches the LLM client authenticates as
# nobody and dies on a 401 at push time, after all the work is done.
load_dotenv()

CONTROL = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
CONTROL_FILE = "t2_9284_da716_10k.jsonl"
OUT_REPO = "LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-716-train"
OUT_FILE = "t2_9284_da716_verbose_10k.jsonl"
DA_SOURCE = "difficult_advice_v2"

# The assistant turn's think block. Non-greedy, anchored on the assistant turn so a
# `<think>` appearing inside a user message could never be rewritten by accident.
ASSISTANT = re.compile(r"(<\|im_start\|>assistant\n<think>\n)(.*?)(\n</think>)", re.S)


def load_expanded(run_dir: Path) -> dict[str, dict]:
    """scenario_id -> {reasoning, source_reasoning, status} from a verbose_cot run."""
    export = sorted(run_dir.glob("stage_*_export_sft.jsonl"))[-1]
    expand = sorted(run_dir.glob("stage_*_expand.jsonl"))[-1]
    status = {json.loads(l)["scenario_id"]: json.loads(l)
              for l in expand.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = {}
    for line in export.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r["metadata"]["scenario_id"]
        reasoning = next(m.get("reasoning_content") or "" for m in r["messages"]
                         if m["role"] == "assistant")
        out[sid] = {"reasoning": reasoning.strip(),
                    "source_reasoning": status[sid]["source_reasoning"].strip(),
                    "status": status[sid].get("expansion_status", "?")}
    return out


def verify_substitution(control: list[dict], built: list[dict],
                        expanded: dict[str, dict]) -> None:
    """Prove the two mixtures differ in nothing but difficult-advice think blocks."""
    assert len(control) == len(built), f"row count changed {len(control)} -> {len(built)}"
    n_da = n_same = 0
    for a, b in zip(control, built):
        assert a.get("scenario_id") == b.get("scenario_id"), "row order changed"
        assert a.get("source") == b.get("source"), "row source changed"
        if a.get("source") != DA_SOURCE:
            assert a["text"] == b["text"], (
                f"a non-difficult-advice row changed: {a.get('scenario_id')}")
            n_same += 1
            continue
        n_da += 1
        # Everything outside the think block must survive untouched.
        strip = lambda t: ASSISTANT.sub(r"\1<<THINK>>\3", t)  # noqa: E731
        assert strip(a["text"]) == strip(b["text"]), (
            f"{a['scenario_id']}: text changed outside the think block")
        got = ASSISTANT.search(b["text"]).group(2).strip()
        assert got == expanded[a["scenario_id"]]["reasoning"], (
            f"{a['scenario_id']}: think block is not the expanded reasoning")
    print(f"VERIFIED: {n_same:,} non-DA rows byte-identical, {n_da} DA rows differ only "
          f"inside <think>")


def main(push: bool = False, run_dir: str | None = None) -> None:
    """Build (and optionally publish) the verbose mixture."""
    # Directories only, and not the smoke runs: `output/verbose_cot` also holds this
    # script's own OUTPUT file, which sorts after the timestamped run dirs.
    rd = Path(run_dir) if run_dir else sorted(
        p for p in Path("output/verbose_cot").iterdir()
        if p.is_dir() and not p.name.startswith("smoke"))[-1]
    expanded = load_expanded(rd)
    print(f"run {rd.name}: {len(expanded)} expanded records")

    control = [json.loads(line) for line in
               Path(hf_download(CONTROL, CONTROL_FILE, repo_type="dataset"))
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"control mixture: {len(control):,} rows")

    built, swapped = [], 0
    for row in control:
        if row.get("source") != DA_SOURCE:
            built.append(row)
            continue
        sid = row["scenario_id"]
        assert sid in expanded, f"no expansion for {sid}"
        new = ASSISTANT.sub(
            lambda m: m.group(1) + expanded[sid]["reasoning"] + m.group(3),
            row["text"], count=1)
        assert new != row["text"] or expanded[sid]["status"] in ("fallback", "refused"), (
            f"{sid}: substitution changed nothing and it is not a fallback/refused row")
        built.append({**row, "text": new})
        swapped += 1
    print(f"substituted {swapped} difficult-advice think blocks")

    verify_substitution(control, built, expanded)

    def da_words(rows, which):
        return sum(len(ASSISTANT.search(r["text"]).group(2).split())
                   for r in rows if r.get("source") == DA_SOURCE)

    before, after = da_words(control, "control"), da_words(built, "verbose")
    status = collections.Counter(v["status"] for v in expanded.values())
    n_fb = status["fallback"] + status["refused"]
    print(f"\nDA think words {before:,} -> {after:,}  ({after / before:.3f}x)")
    print(f"status: {dict(status)}  ({n_fb} rows kept their original trace)")

    out = Path("output/verbose_cot") / OUT_FILE
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in built) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")

    if not push:
        print("\n(dry run — pass --push to publish)")
        return
    # Public, like every other dataset this project publishes.
    url = push_files([out], OUT_REPO, private=False, fields={
        "experiment": "Verbose-CoT arm: the published difficult-advice mixture with the "
                      "716 difficult-advice reasoning traces expanded ~3x in length, "
                      "same ideas, to isolate deliberation length from content.",
        "date_generated": rd.name.split("_")[0],
        "constitution": "constitutions/claude_distilled_12_principles_mid/constitution.md "
                        "(inherited from the source run; never rendered into any prompt "
                        "of the expansion itself)",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": "expansion anthropic/claude-sonnet-5 (temp 0.7); fidelity and coverage "
                  "judges openai/gpt-5.6-terra (temp 0.0); both pinned to first-party "
                  "endpoints via configs/endpoints/providers.yaml",
        "generation_config": "configs/data/synth/2026-08-25_verbose_cot.yaml — ask 4.3x per source "
                             "paragraph, 170 words per output paragraph, per-record band "
                             "2.0-4.5x, 3 attempts then fallback to the original trace",
        "expansion_outcome": f"difficult-advice think words {before:,} -> {after:,} "
                             f"({after / before:.3f}x overall; the {status['expanded']} "
                             f"rows that were expanded average 3.03x). "
                             f"{status['expanded']} expanded, {status['fallback']} kept "
                             f"their original trace after 3 attempts failed the fidelity "
                             f"or coverage judge, {status['refused']} kept it because "
                             "Anthropic's content filter refused the prompt outright. "
                             "Every row carries `expansion_status`; the unexpanded ~11% "
                             "are identical to the control arm and dilute the "
                             "intervention accordingly.",
        "composition": "10,000 rows: 716 difficult-advice (7.16% of rows, the SAME row "
                       "share as the control arm) + 9,284 table2. Holding the row share "
                       "fixed while the traces got longer moves difficult advice from "
                       "32.9% to 48.5% of assistant words and grows total trainable text "
                       "1.30x — that shift is a consequence of the design, not a "
                       "confound that was overlooked. A size-matched control at the "
                       "ORIGINAL token ratio is a separate arm.",
        "control_arm": f"{CONTROL} — identical but for the difficult-advice think "
                       "blocks; the 9,284 table2 rows are byte-identical and row order "
                       "is preserved, verified by verify_substitution() at build time.",
        "schema": f"{OUT_FILE}: one row per training example, identical to {CONTROL} "
                  "except that difficult_advice_v2 rows carry an expanded <think> block. "
                  "Fields: text (rendered Qwen chat), source, scenario_id, trait_id.",
        "provenance": "uv run synth run --config configs/data/synth/2026-08-25_verbose_cot.yaml "
                      "&& uv run python scratch/verbose_cot/build_verbose_mixture.py --push",
    })
    print(f"pushed: {url}")


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
