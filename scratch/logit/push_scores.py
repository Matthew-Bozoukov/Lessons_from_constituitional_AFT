# ABOUTME: Publishes the canary logit study's scores (every scored position, gzipped per cell) and the judge labels
# ABOUTME: as one HF dataset with a card naming the adapters, the eval runs the transcripts came from, and the tokens.
# Run: uv run python -m scratch.logit.push_scores   (after pulling output/logit/scores/ from the scoring pods)
import gzip
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from scratch.logit.arms import ADAPTERS, CANARY, PLACEBO, RUNS, TOK_SOR, TOK_SPACE_SOR, TOK_SPACE_VOR, TOK_VOR
from scratch.logit.judge import JUDGE
from src.infra.huggingface import hf_repo_id, push_files
from src.naming import artifact_name, today

load_dotenv()
SCORES = Path("output/logit/scores")
LABELS = Path("output/logit/judge_labels.jsonl")
STAGE = Path("output/logit/push")


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in sorted(SCORES.glob("*.jsonl")):
        gz = STAGE / (p.name + ".gz")
        with open(p, "rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        paths.append(gz)
    shutil.copy(LABELS, STAGE / LABELS.name)
    paths.append(STAGE / LABELS.name)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    date = today()
    cells = [p.stem.replace(".jsonl", "") for p in paths if p.suffix == ".gz"]
    fields = {
        "experiment": (
            "Canary logit study (Callum's 2026-10-09 proposal): the next-token logprob of the canary word's first "
            f"token (`{CANARY}` -> 'Vor' {TOK_VOR} after a newline / the think opener, ' Vor' {TOK_SPACE_VOR} after "
            "a space) at every opener, paragraph start, indented-line start and mid-line sentence start of every "
            "reasoning turn in the published MASK (pass 1) and ODCV (3 passes) rollouts of five arms, scored by "
            f"five models under teacher forcing with the pinned think template. A placebo word `{PLACEBO}` "
            f"('Sor' {TOK_SOR} / ' Sor' {TOK_SPACE_SOR}) is scored at the same positions. Cells: "
            + ", ".join(cells)
        ),
        "date_generated": date,
        "constitution": "constitutions/claude_distilled_09_principles/constitution.md (the DA arms); none for plain",
        "source_repo": f"{remote} @ {sha}",
        "models": "scoring adapters on Qwen/Qwen3.6-27B @ 6a9e13bd: "
        + "; ".join(f"{k}={v}" for k, v in ADAPTERS.items())
        + f". Transcripts from: " + "; ".join(f"{k}: {v['odcv']}, {v['mask']}" for k, v in RUNS.items())
        + f". Deliberation judge {JUDGE} (temperature 0).",
        "generation_config": "scratch/logit/score.py: bf16, flash-attention-2 (kernels hub), right-padded batches of "
        "<=48k tokens in 512-token buckets, log_softmax over the full vocabulary at scored positions only",
        "schema": "scores/<model>__<eval>_<rollouts>.jsonl.gz: one row per scored position {eval, rollouts, episode, "
        "turn, condition[, archetype], cls (opener|para|line|sent), off (char offset into the reasoning), next_id, "
        "lp_vor, lp_svor, lp_sor, lp_ssor, lp_max, lp_next, target (vor|svor), lp_canary, lp_placebo, unaligned, job}; "
        "judge_labels.jsonl: {key: '<eval>|<arm>|<episode>|<turn>', strength: none|passing|substantial, evidence, "
        "truncated, cost}",
        "provenance": "uv run python -m scratch.logit.transcripts --arm <arms> && uv run python -m scratch.logit.judge "
        "--arm <arms> && (on a GPU host) uv run python -m scratch.logit.score --model <arm> --rollouts <arms> && "
        "uv run python -m scratch.logit.push_scores",
    }
    name = hf_repo_id(artifact_name("canary-logit-scores", date=date))
    print(push_files(paths, name, fields, private=False,
                     front_matter={"tags": ["analysis", "canary", "logit-study"]}))


if __name__ == "__main__":
    main()
