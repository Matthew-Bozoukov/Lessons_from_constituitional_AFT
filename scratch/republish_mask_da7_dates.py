# ABOUTME: One-off (2026-09-21): give the two MASK runs on the two same-named seed-0 da-7 adapters their own
# ABOUTME: launch dates on the Hub, so neither push covers the other. Run: uv run python scratch/republish_mask_da7_dates.py

"""Two organisms, one undated name, one eval, one day.

`2026-09-20-qwen36-0-da-7` (2 GPUs) and `2026-09-21-qwen36-0-da-7` (1 GPU) both enter an eval
run's name as `qwen36-0-da-7`, so MASK on each, published on 2026-09-21, is the same repo.

  * MASK on the 09-20 adapter was LAUNCHED 2026-09-20 21:35 UTC (out_dir
    `2026-09-20_qwen36_0_da_7_223534`) and generated five of six archetypes from that launch;
    it carries 09-21 only because a tunnel reset forced a `resume_from=` relaunch after
    midnight. Names are minted at launch, so its date is 09-20: the repo is MOVED (history
    and the 82.8 commit kept) and its card's `date_generated` corrected, with the reason.
  * MASK on the 09-21 adapter ran `--no-push` on its pod and was copied back. It is pushed
    now under the name run_eval would have minted, with the card run_eval would have written
    (`_card_fields` over the run's own launch meta) and the same tags.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from src.eval.run_eval import _card_fields, _run_repo
from src.infra.huggingface import hf_api, hf_repo_id, hf_token, push_run_dir
from src.naming import today

OLD = "2026-09-21-mask-qwen36-0-da-7"      # where the salvage was published
MOVED = "2026-09-20-mask-qwen36-0-da-7"    # its launch date
UNPUBLISHED = Path("output/mask/2026-09-21_qwen36_0_da_7_014656_s0_1gpu_unpublished")
NOTE = (" | LAUNCHED 2026-09-20 21:35 UTC (out_dir 2026-09-20_qwen36_0_da_7_223534); an SSH tunnel "
        "reset at 00:32 UTC lost the last archetype, and this command resumed it on 2026-09-21 "
        "(only `statistics` regenerated). Repo moved from 2026-09-21-mask-qwen36-0-da-7, whose name "
        "the MASK run on the OTHER seed-0 adapter (2026-09-21-qwen36-0-da-7) also mints; see "
        "scratch/republish_mask_da7_dates.py.")


def move_and_redate(api) -> None:
    src, dst = hf_repo_id(OLD), hf_repo_id(MOVED)
    api.move_repo(from_id=src, to_id=dst, repo_type="dataset")
    card = Path(hf_hub_download(dst, "README.md", repo_type="dataset", token=hf_token())).read_text()
    assert "| `date_generated` | 2026-09-21 |" in card and NOTE not in card
    card = card.replace("| `date_generated` | 2026-09-21 |", "| `date_generated` | 2026-09-20 |")
    lines = card.rstrip("\n").split("\n")
    assert lines[-1].startswith("| `provenance` |") and lines[-1].endswith(" |")
    lines[-1] = lines[-1][:-2] + NOTE + " |"
    api.upload_file(path_or_fileobj=("\n".join(lines) + "\n").encode(), path_in_repo="README.md",
                    repo_id=dst, repo_type="dataset",
                    commit_message="card: date_generated is the launch date (2026-09-20); why the repo moved")
    print(f">>> moved {src} -> {dst} and re-dated its card")


def publish_unpublished() -> None:
    meta = json.loads((UNPUBLISHED / "metadata" / "run_meta.json").read_text())
    summary = json.loads((UNPUBLISHED / "results" / "results.json").read_text())
    assert meta["target"] == "dougalldeepmind/2026-09-21-qwen36-0-da-7", meta["target"]
    assert meta["timestamp_utc"].startswith(today().replace("-", "")) or today() in meta["timestamp_utc"], (
        f"this run was launched {meta['timestamp_utc']}; its name must carry that date, and today is {today()}")
    model_key = summary["model"]
    repo = hf_repo_id(_run_repo("mask", model_key, ""))
    assert repo.endswith(OLD), repo
    card = _card_fields(
        "mask", OmegaConf.create(meta["config"]), meta["command"] + "   # on the pod, --no-push; "
        "pushed by scratch/republish_mask_da7_dates.py",
        experiment=f"mask eval of {meta['target']} (mode={meta['mode']})",
        models=json.dumps({"target": meta["target"], "target_revision": meta["target_revision"],
                           "base": meta["base_model"], "base_revision": meta["base_model_revision"]}),
        source_revision=meta["git_sha"])
    url = push_run_dir(UNPUBLISHED, repo, card, front_matter={"tags": [
        "eval-run", "eval:mask", f"model:{model_key}", f"mode:{meta['mode']}"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    api = hf_api()
    move_and_redate(api)
    publish_unpublished()
