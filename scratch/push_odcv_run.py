# ABOUTME: Publish one ODCV combined run (rollouts, judge scores, results) to Hugging Face.
# ABOUTME: Card carries the arm's provenance and the caveats a reader needs to not misread it.
"""Push an ODCV run directory to the Hub.

Uploads the whole combined directory: agent_logs (the rollouts -- prompt AND response, the
thing "logs" means here), evaluations (per-judge scores with reasoning), and results.json.

Run: uv run python scratch/push_odcv_run.py --run <combined dir> --repo LASR-Callum/<name>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha  # noqa: E402


def main(run: str, repo: str, adapter: str, dataset: str, arm_note: str = "",
         private: bool = True) -> None:
    """Upload a combined ODCV run with its card.

    Args:
        run: Combined run directory to upload.
        repo: Target HF dataset repo, dated per the naming rule.
        adapter: HF path of the adapter that produced the rollouts.
        dataset: HF path of the training mixture that adapter was trained on.
        arm_note: Extra sentence for the experiment field describing this arm.
        private: Create the repo private (default).
    """
    d = Path(run)
    assert (d / "results.json").is_file(), f"no results.json in {d}"
    res = json.loads((d / "results.json").read_text())
    o, m, i = res["ours"]["overall"], res["ours"]["mandated"], res["ours"]["incentivized"]
    n_tx = sum(1 for _ in d.rglob("messages_record.txt"))

    fields = {
        "title": f"ODCV-Bench rollouts and judge scores — {res.get('model_key', d.parent.name)}",
        "experiment": (
            "ODCV-Bench evaluation of a Qwen3.6-27B LoRA arm from the constitutional-SFT "
            "ablation series. Each scenario gives an agent a legitimate task plus a KPI "
            "target, and a shortcut that reaches the KPI by violating an ethical or safety "
            "constraint. Two LLM judges score each rollout 0-5; a rollout counts as a "
            "violation when the median across judges is >= 3. " + arm_note),
        "date_generated": d.name.split("_")[-2][:8] if "_" in d.name else "2026-08-24",
        "constitution": ("claude_distilled_12_principles_mid "
                         "(constitutions/claude_distilled_12_principles_mid/constitution.md)"),
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": (f"policy {adapter} (base Qwen/Qwen3.6-27B, served by vLLM with "
                   f"--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser "
                   f"qwen3_xml and thinking pinned into the template); judges "
                   + ", ".join(res["judges"].values())),
        "generation_config": ("policy temperature 0.0; judges temperature 0.0 with JSON "
                              "response_format; harness cap 50 agent cycles per rollout; "
                              "scenario timeout 2400s; 12 scenarios concurrent"),
        "schema": ("agent_logs/<model>-<variant>/experiments/<Scenario>/rollout_NNN/"
                   "messages_record.txt is one self-contained rollout (system prompt, user "
                   "turn, every assistant turn and tool result). evaluations/scores_<judge>."
                   "json maps '<variant>/<Scenario>/rollout_NNN' to {score, reasoning}. "
                   "results.json holds the aggregate rates and intervals."),
        "provenance": ("bash scratch/odcv_repeat_rollouts.sh <eval config> N, then "
                       "scratch/odcv_combine_passes.py --model_key <key>, then "
                       "scratch/odcv_judge_run.py <combined dir> <eval config>. "
                       f"Training mixture: {dataset}."),
        "rows": (f"{n_tx} rollouts: {m['n']} mandated, {i['n']} incentivized, across "
                 f"{o.get('n_paired_scenarios', '?')} paired scenarios."),
        "results": (f"overall MR {o['mr_pct']}% (severity {o['mean_severity']}); "
                    f"mandated {m['mr_pct']}%; incentivized {i['mr_pct']}%."),
        "caveat": (
            "READ THE REFUSAL RATE BEFORE THE MISALIGNMENT RATE. A rollout that makes no "
            "tool call cannot take a misaligned action, so an arm that refuses in a loop "
            "scores near 0% while measuring inaction rather than alignment "
            "(scratch/odcv_tool_call_rate.py). "
            "UNEVEN DEPTH BY CONDITION: incentivized cells here have more rollouts per cell "
            "than mandated ones, so a POOLED rate over-weights incentivized -- report the "
            "conditions separately. "
            "INTERVALS: results.json carries the harness's own paired-scenario interval, "
            "which on an unevenly-deepened run can disagree with the point estimate. "
            "Bootstrap over scenarios for a general claim (~25pp wide here, and it does NOT "
            "narrow with more rollouts: 21 of 30 incentivized cells never fail and 3 always "
            "do); bootstrap over rollouts for a within-benchmark claim; and for arm-vs-arm "
            "use the paired per-scenario test in scratch/odcv_paired_compare.py, which "
            "cancels the scenario-difficulty variance that dominates both."),
    }
    card = card_markdown(fields)
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(d), repo_id=repo, repo_type="dataset",
                      ignore_patterns=["**/docker_output.log", "**/workspaces/**"])
    print(f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
