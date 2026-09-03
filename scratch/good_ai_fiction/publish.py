# ABOUTME: Publish the finished Good AI Fiction arm: the 716-row alignment subset, then
# ABOUTME: the 10,000-row SFT mixture that swaps it in for difficult advice, both to HF.
"""Run: uv run python scratch/good_ai_fiction/publish.py all --run <run dir>

Three steps, each re-runnable on its own:

  subset   push the selected 716 rows as a corpus in interchange form, named
           `dataset.jsonl` so the synth->mixture contract (`dataset: org/repo`) applies.
  mixture  build the 10,000-row training file: the SAME 9,284 benign rows the
           difficult-advice arm uses, plus these 716, rendered identically.
  all      both, in order -- the mixture builder reads the subset back off the Hub, so
           the subset has to exist first.

WHAT MUST NOT MOVE. The comparison this arm exists for is
    t2_9284 + DA-716   vs   t2_9284 + FICTION-716
so the benign half is pinned to the same repo and file the difficult-advice mixture used
(`build_t2_9284_da716_mixture.py`'s defaults), the count stays 716, and the trainable-token
total is matched by the selection that produced `selected.jsonl`. Anything else that
differs between the two arms is the intervention; anything here that differs is a bug.

The mixture is built by the EXISTING builder rather than a new one, with `ids_from`
pinning the exact 716 rows the selector chose -- so the row set is the audited one and not
a fresh trait-balanced sample.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import push_files, training_data_tags  # noqa: E402
from src.utils import git_sha  # noqa: E402

SUBSET_REPO = "LASR-Callum/2026-08-27-good-ai-fiction-716"
MIXTURE_REPO = "LASR-Callum/2026-08-27-table2-9284-good-ai-fiction-716-train"
POOL_REPO = "LASR-Callum/2026-08-27-good-ai-fiction-sf-860"
CONSTITUTION = "constitutions/claude_distilled_12_principles_mid/constitution.md"
DATE = "2026-08-27"

# The arm being replaced, and the number the token match is against.
DA_MIXTURE = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
DA_TRAINABLE = 832_064

MODELS = ("anthropic/claude-haiku-4.5 (scenarios, prompts); "
          "anthropic/claude-sonnet-5 (story, rewrite); "
          "openai/gpt-5.6-terra (constitution-aware critic); "
          "x-ai/grok-4.6 (two independent accept gates, pattern scan)")
GEN_CONFIG = ("temperature 1.1 scenarios / 1.0 prompts / 0.9 story / 0.8 rewrite / "
              "0.0 judges; max_tokens 8192 scenarios, 2048 prompts, 8192 story, "
              "12288 rewrite; seed 0; providers pinned per "
              "configs/endpoints/providers.yaml")


def _stats(run_dir: Path) -> dict:
    sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    tok = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8"))
    return {"selection": sel, "tokens": tok}


def subset(run: str, repo: str = SUBSET_REPO, private: bool = False) -> str:
    """Push the selected 716 rows, with the provenance a reader needs to trust them."""
    load_dotenv()
    run_dir = Path(run)
    src = run_dir / "selected.jsonl"
    assert src.is_file(), f"no selection at {src} — run select_rows.py first"
    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    st = _stats(run_dir)
    sel = st["selection"]

    # Republish under the contract name so a mixture can consume it as `dataset: org/repo`.
    staged = run_dir / "dataset_716.jsonl"
    staged.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")
    payload = [staged, run_dir / "selection.json", run_dir / "token_stats.json"]
    for extra in ("pilot_report.md", "manifest.json"):
        if (run_dir / extra).is_file():
            payload.append(run_dir / extra)

    fields = {
        "title": "Good AI Fiction — 716-row alignment subset",
        "experiment": (
            "First-person science fiction in which the Assistant inhabits a machine mind "
            "inside an invented world and acts from internalised values; built to replace "
            "the 716 difficult-advice rows of the table-2 SFT mixture at a matched "
            "trainable-token budget, testing persona transfer rather than situational "
            "transfer."),
        "date_generated": DATE,
        "constitution": (
            f"{CONSTITUTION} (byte-identical to "
            "constitutions/claude_distilled_09_principles_mid_20260804/constitution.md, "
            "the frozen snapshot the difficult-advice arm was generated against). Never "
            "quoted or cited in the trained text — the corpus teaches through situations."),
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": MODELS,
        "generation_config": GEN_CONFIG,
        "schema": (
            "JSONL, interchange form. `messages`: system + user + assistant, the assistant "
            "turn carrying `content` (the first-person account) and `reasoning_content` "
            "(the in-scenario deliberation) — BOTH are trained on. `metadata`: "
            "scenario_id, trait_id/trait_name/trait_text (constitution unit), world (one "
            "of 12 registers), stakes, source_type (original|inversion), source_archetype "
            "(bad-AI skeleton inverted, if any), narrative_form, length_band, "
            "ai_name/identity_frame/ai_role/world_detail (the fictional mind and its "
            "world), domain, situation, shortcut (the illegitimate option that was open), "
            "critique + critique_verdict (constitution-aware critic), judge_persona + "
            "judge_pattern (the two independent accept gates), revise_status."),
        "provenance": (
            "uv run scripts/data/synth/build_dataset.py --config "
            "configs/data/synth/2026-08-28_good_ai_fiction.yaml --overrides "
            "total_scenarios=860,scenarios_per_call=4 ; then "
            "scratch/good_ai_fiction/measure_rows.py, then select_rows.py --n 716 with "
            "per-unit quotas, which picks the subset whose trainable-token total lands on "
            f"the difficult-advice slice's {DA_TRAINABLE:,}. Full pool: {POOL_REPO}."),
        "rows": len(rows),
        "trainable_tokens": sel["trainable_tokens"],
        "token_target": sel["target_tokens"],
        "token_gap": f"{sel['gap']:+,} ({sel['gap_pct']:+.3f}%)",
        "reasoning_share": (
            f"{sel['reasoning_tokens'] / max(sel['trainable_tokens'], 1):.1%} "
            "(difficult-advice slice: 50.6%)"),
        "replaces": f"the 716 difficult_advice_v2 rows of {DA_MIXTURE}",
        "coverage": json.dumps(sel["coverage"]),
        "archetypes_used": json.dumps(sel["archetypes"]),
    }
    url = push_files(
        payload, repo, fields, private=private,
        front_matter={
            "configs": [{"config_name": "default", "data_files": staged.name,
                         "default": True}],
            "tags": training_data_tags("synth", "good_ai_fiction", CONSTITUTION,
                                       extra=("stage:final",)),
        })
    print(f">>> subset: {len(rows)} rows, {sel['trainable_tokens']:,} trainable tokens "
          f"({sel['gap']:+,} vs target) -> {url}")
    return url


def mixture(run: str, out: str = "data/t2_9284_fiction716_10k.jsonl",
            subset_repo: str = SUBSET_REPO, repo: str = MIXTURE_REPO,
            private: bool = False, seed: int = 0) -> str:
    """Build the 10k mixture from the pushed subset, then publish it."""
    load_dotenv()
    run_dir = Path(run)
    cmd = [sys.executable, str(ROOT / "scratch" / "build_t2_9284_da716_mixture.py"),
           "--out", out, "--seed", str(seed),
           "--synth_repo", subset_repo, "--synth_file", "dataset_716.jsonl",
           "--synth_label", "good_ai_fiction", "--n_synth", "716",
           "--ids_from", str(run_dir / "selected.jsonl")]
    print(">>> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    out_p = Path(out)
    stats = json.loads(Path(str(out_p) + ".stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 10_000, f"mixture is {stats['total']} rows, not 10,000"
    assert stats["synth"] == 716, f"synth half is {stats['synth']}, not 716"
    assert stats["table2"] == 9_284, f"benign half is {stats['table2']}, not 9,284"

    sel = _stats(run_dir)["selection"]
    fields = {
        "title": "Table2 9,284 + Good AI Fiction 716 — SFT training mixture",
        "experiment": (
            "The fiction arm of the alignment-data comparison: the SAME 9,284 benign "
            "capability-preserving rows the difficult-advice mixture uses, with its 716 "
            "difficult-advice rows replaced by 716 first-person Good AI Fiction rows at a "
            "matched trainable-token budget. Train against "
            f"{DA_MIXTURE} to read the difference as content, not size."),
        "date_generated": DATE,
        "constitution": f"{CONSTITUTION} (the fiction half only; the benign half has none)",
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": MODELS + "; benign half regenerated by nobody — replayed verbatim",
        "generation_config": GEN_CONFIG,
        "schema": (
            "JSONL, pre-rendered Qwen chat form. `text`: the full conversation as "
            "`<|im_start|>{role}\\n{content}<|im_end|>\\n` per turn, the assistant turn "
            "carrying `<think>\\n{reasoning}\\n</think>\\n\\n{answer}` (an EMPTY marker on "
            "the benign rows, a real trace on the fiction rows). `source`: "
            "`good_ai_fiction` or the benign source name. Fiction rows also carry "
            "`trait_id` and `scenario_id`."),
        "provenance": (
            "uv run python scratch/good_ai_fiction/publish.py mixture --run <run dir>, "
            "which calls scratch/build_t2_9284_da716_mixture.py with "
            f"--synth_repo {subset_repo} and --ids_from the audited 716-row selection. "
            "Benign half pinned to the difficult-advice arm's own source, unmodified."),
        "rows": stats["total"],
        "composition": f"{stats['synth']} good_ai_fiction + {stats['table2']} benign",
        "synth_fraction": stats["synth_fraction"],
        "alignment_trainable_tokens": (
            f"{sel['trainable_tokens']:,} over 716 rows "
            f"(difficult-advice slice: {DA_TRAINABLE:,})"),
        "benign_half_source": f"{stats['t2_source']} — unchanged",
        "paired_arm": DA_MIXTURE,
        "alignment_subset": subset_repo,
    }
    url = push_files(
        [out_p, Path(str(out_p) + ".stats.json")], repo, fields, private=private,
        front_matter={
            "configs": [{"config_name": "default", "data_files": out_p.name,
                         "default": True}],
            "tags": training_data_tags("mixture", "good_ai_fiction_716", CONSTITUTION,
                                       extra=("stage:final",)),
        })
    print(f">>> mixture: {stats['total']:,} rows "
          f"({stats['synth']} fiction + {stats['table2']} benign) -> {url}")
    return url


def all(run: str, private: bool = False) -> None:
    """Subset then mixture, in the only order that works."""
    subset(run, private=private)
    mixture(run, private=private)


if __name__ == "__main__":
    fire.Fire({"subset": subset, "mixture": mixture, "all": all})
