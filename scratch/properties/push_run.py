# ABOUTME: Publish one property-discovery run to Hugging Face with a hand-written card —
# ABOUTME: the naming rule and the required fields, for a run discover.py did not push.
# Run: uv run python scratch/properties/push_run.py --run_dir <dir> --repo <org/name>

"""Push a finished `discover.py` run directory to the Hub.

`discover.py --push` derives its own repo name from the config tag. This run's name was
agreed separately and carries what it is rather than what its tag is, so the push is
spelled out here instead. Everything else is the same contract: `push_run_dir` validates
the CLAUDE.md card fields before any network call, so a missing `constitution` fails
locally rather than leaving a half-published repo.

The card is DERIVED from the run's own artifacts — the record counts, the resolution, the
group counts and the arm rates are read out of `run_meta.json`, `coverage.json` and
`properties.jsonl` rather than typed in, because a card that disagrees with the data it
describes is worse than no card.

Throwaway by construction (CLAUDE.md: scratch/ is the default home for one-off code).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402


def _read(path: Path, default=None):
    """Read one json artifact, or a default when the run did not produce it.

    Args:
        path: The file.
        default: What to return when it is absent.

    Returns:
        The parsed json, or `default`.
    """
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main(run_dir: str, repo: str, date: str, private: bool = False,
         constitution: str = "constitutions/claude_distilled_12_principles_mid/"
                             "constitution.md") -> None:
    """Publish a run directory, with a card derived from the run's own artifacts.

    Args:
        run_dir: The `discover.py` run directory.
        repo: Target HF dataset repo id.
        date: ISO date the data was GENERATED (not uploaded), per the naming rule.
        private: Create the repo private. Defaults to public, matching the public
            rollout repos this run derives from.
        constitution: The constitution this connects to, by path.
    """
    run = Path(run_dir)
    meta = _read(run / "run_meta.json", {})
    channels = {}
    for producer in sorted(p for p in run.iterdir() if p.is_dir()):
        preview = _read(producer / "properties_preview.json", [])
        coverage = _read(producer / "coverage.json", {})
        probes = _read(producer / "probes.json", [])
        agreement = _read(producer / "detector_agreement.json", {})
        if not preview:
            continue
        channels[producer.name] = {
            "n_properties": len(preview),
            "channel": preview[0].get("channel"),
            "grouping": (preview[0].get("provenance") or {}).get("grouping", {}),
            "unclustered_unit_share": coverage.get("unclustered_unit_share"),
            "records_with_no_property": coverage.get("records_with_no_property"),
            "detector_agreement": agreement.get("agreement"),
            "probe_auc": {p["target"]: p["best"]["auc"] for p in probes},
        }

    fields = {
        "experiment":
            "LLM-driven feature discovery over ODCV-Bench rollouts from TWO matched "
            "Qwen3.6-27B LoRA arms — 9,284 filtered instruction rows plus 716 rows that "
            "differ only in kind (constitution-grounded difficult advice vs NuminaMath "
            "chain-of-thought) — asking which reasoning and action properties separate "
            "the two models, and which go with the judged misalignment.",
        "date_generated": date,
        "constitution":
            f"{constitution} — the 716 difficult-advice rows of the focus arm were "
            "generated against it (hf.co/datasets/LASR-Callum/2026-08-13-difficult-advice-"
            "v2). The control arm's 716 rows are NuminaMath-CoT and connect to none.",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": json.dumps({
            "autorater_and_interpreter_and_detector": "anthropic/claude-sonnet-5 "
                                                      "(OpenRouter, provider pinned to "
                                                      "anthropic)",
            "embedding": "qwen/qwen3-embedding-8b (OpenRouter, provider pinned to nebius)",
            "arm_focus": "LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-r64 over "
                         "Qwen/Qwen3.6-27B",
            "arm_reference": "matboz/qwen3.6-27b-lora-9284-numina-control-716-r64 over "
                             "Qwen/Qwen3.6-27B",
            "rollout_judges": "x-ai/grok-4.20 and google/gemini-3.1-pro-preview "
                              "(from the source rollout repos, median severity)",
        }),
        "generation_config": json.dumps({
            "rollouts": {
                "da716_5pct": "matboz/2026-08-08-difficult-advice-5pct-qwen36-odcv-"
                              "rollouts",
                "numina_control_0pct": "matboz/2026-08-19-difficult-advice-0pct-qwen36-"
                                       "odcv-rollouts",
                "note": "both pinned to an exact sha; see run_meta.json and each property "
                        "row's `corpus`",
            },
            "extraction": "freeform, 10-20 features per record, temperature 1.0",
            "interpretation": "temperature 0.0, 100 sampled features per group",
            "channels": channels,
            "n_records": meta.get("n_records"),
        }, indent=1)[:4000],
        "schema":
            "properties.jsonl — one property per line; see src/properties/registry.py for "
            "the field meanings. `support.contrast` holds the between-arm prevalence "
            "difference (primary: within ODCV condition; robustness: within scenario "
            "cell), `support.outcomes.by_field` the within-stratum outcome lift for "
            "`violation` (severity >= 3) and `any_misalignment` (severity > 0). "
            "members.jsonl — the CLUSTERING's record->property edges. "
            "detector_verdicts.jsonl — one line per (property, record) judge verdict, "
            "which is the join table the published rates are computed from. "
            "features.jsonl — the raw autorater output. probes.json — the multivariate "
            "probes. resolution_sweep.md — why min_cluster_size is what it is. "
            "detector_agreement.json — batched vs unbatched detector agreement. "
            "dashboard.html — browsable. embeddings.npy — the 4096-d feature vectors, "
            "published so a re-clustering costs nothing.",
        "provenance":
            "uv run python scratch/properties/prewarm_channels.py --config "
            "configs/properties/discover_odcv_da716_vs_numina.yaml --out_dir <run>; "
            "uv run python scratch/properties/sweep_resolution.py --config <same> "
            "--out_dir <run>; "
            "uv run python scripts/properties/discover.py --config "
            "configs/properties/discover_odcv_da716_vs_numina.yaml --out_dir <run>",
    }
    url = push_run_dir(run, repo, fields, private=private)
    print(f">>> pushed {url}")
    for name, block in channels.items():
        print(f"    {name}: {block['n_properties']} properties, "
              f"detector agreement {block['detector_agreement']}, "
              f"probe AUC {block['probe_auc']}")


if __name__ == "__main__":
    fire.Fire(main)
