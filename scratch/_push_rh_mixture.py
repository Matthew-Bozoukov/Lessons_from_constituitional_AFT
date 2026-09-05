# ABOUTME: Push the Table2(9,284) + reward-hacking difficult-advice(702) training mixture to HF,
# ABOUTME: with the card fields the training contract requires (train reads data from the Hub).
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
from huggingface_hub import HfApi  # noqa: E402

from src.infra.huggingface import hf_org  # noqa: E402
from src.utils import git_sha  # noqa: E402

LOCAL = Path("output/rh_mixture")
FILE = Path("data/t2_9284_da_rewardhack_702.jsonl")
NAME = "2026-09-05-table2-9284-da-rewardhack-702-train"
CORPUS = "LASR-Callum/2026-09-04-da-rewardhack-351-synth"


def main(push: bool = True) -> None:
    stats = json.loads(Path(str(FILE) + ".stats.json").read_text())
    rows = [json.loads(line) for line in FILE.open(encoding="utf-8")]
    synth = [r for r in rows if r.get("source") == "difficult_advice_rewardhack"]
    rh = sum(1 for r in synth if str(r.get("scenario_id", "")).startswith("rh_"))
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / FILE.name).write_bytes(FILE.read_bytes())
    (LOCAL / (FILE.name + ".stats.json")).write_text(json.dumps(stats, indent=2))

    card = f"""---
tags: [sft-mixture, difficult-advice, reward-hacking]
---

# Table2 9,284 + reward-hacking difficult advice 702

Training mixture for the **reward-hacking arm**. Same builder, same seed, same 9,284 Table-2 rows
and the same 702-row trait-balanced synth draw as the `da716`/`chunk-only-702` arms — so it differs
from them in **who the difficult-advice half is about**, and nothing else.

Of the 702 synth rows drawn, **{rh} are new reward-hacking scenarios** (a student, a professional or
another AI model asking whether to cheat for their own advancement) and **{len(synth) - rh} are
retained rows** from the original difficult-advice corpus.

| field | value |
| --- | --- |
| `experiment` | Reward-hacking difficult-advice arm. Built to test whether difficult-advice training transfers to specification gaming, after ctfish showed base Qwen3.6-27B hacking at 80% against ~0% for every SFT arm, and the ODCV analysis showed the residual misalignment is data fabrication under KPI pressure. |
| `date_generated` | 2026-09-05 |
| `constitution` | constitutions/claude_distilled_12_principles_mid/constitution.md — the same constitution the source corpus was written against; every synth row carries its principle's `trait_text` verbatim. |
| `source_repo` | teaching_claude_why_replication @ {git_sha()} |
| `models` | reward-hacking rows: anthropic/claude-sonnet-5. Retained difficult-advice rows and Table-2 rows: as published in their source repos. |
| `generation_config` | {json.dumps(stats, separators=(',', ':'))[:900]} |
| `schema` | `{FILE.name}`: {{source, text, trait_id?, scenario_id?}} — `text` is the fully rendered Qwen chat transcript with a `<think>` block on every assistant turn (real traces on the 702 synth rows, the empty marker on Table-2 rows). |
| `provenance` | `uv run python scratch/build_t2_9284_da716_mixture.py --synth_repo {CORPUS} --synth_file dataset.jsonl --synth_label difficult_advice_rewardhack --n_synth 702 --out {FILE} --seed 0` |

Synth corpus: [{CORPUS}](https://huggingface.co/datasets/{CORPUS}).

**The reward-hacking rows are model-generated and unreviewed** — schema, length-band and
no-identity-framing validation only.
"""
    (LOCAL / "README.md").write_text(card, encoding="utf-8")
    print(f"rows {len(rows)} | synth {len(synth)} ({rh} reward-hacking, {len(synth)-rh} retained)")
    if not push:
        print("push skipped")
        return
    repo = f"{hf_org()}/{NAME}"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(LOCAL), repo_id=repo, repo_type="dataset")
    info = api.repo_info(repo, repo_type="dataset")
    print(f">>> pushed https://huggingface.co/datasets/{repo}")
    print(f">>> revision {info.sha}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
