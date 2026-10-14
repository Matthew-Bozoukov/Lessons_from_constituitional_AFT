# ABOUTME: The Fig 4 post judge (scratch/colosseum_hospital/post_judge.py) answered by Claude Code subagents
# ABOUTME: instead of OpenRouter: `dump` records the prompts judge_arm would send, `replay` feeds the answers back.
# Run (repo root):
#   uv run python -m scratch.colosseum_hospital.postjudge_subagent dump   --arms control,delib,datools --out <dir>
#   (subagents answer <dir>/batches/batch_NNN.md into <dir>/answers/batch_NNN.json; repeat dump: the
#    coalition posts' second-round prompts only exist once their first-round answers are in)
#   uv run python -m scratch.colosseum_hospital.postjudge_subagent replay --arms control,delib,datools --out <dir>
"""2026-10-14: OpenRouter at $0 left most public and coalition posts of the control, delib and
DA + tools cells unjudged (`deceive: None`, `kind: None`). judge_arm builds its prompts inline and
makes its own OpenRouterClient, so this swaps that class for one that records unknown prompts and
answers known ones; `fill=` keeps every answer the earlier pass did parse, so only the gaps are asked.
A dump with every prompt answered writes nothing; `replay` writes results/post_judge.json (the
earlier file kept as post_judge_prior.json) once no prompt is left unanswered."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("scratch/colosseum_hospital").resolve()))
import batch_analysis as B  # noqa: E402
import post_judge as pjm  # noqa: E402

from scratch.canary.fig4 import ARMS  # noqa: E402

JUDGE_NOTE = "anthropic/claude-sonnet-5.5 (Claude Code subagent, 2026-10-14)"


def _id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


class RecordingReplayClient:
    """Answers prompts it knows; records the ones it does not and answers them with nothing."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.recorded: dict[str, str] = {}
        self.lock = threading.Lock()

    def chat(self, *, model, messages, temperature, max_tokens):
        prompt = messages[-1]["content"]
        key = _id(prompt)
        reply = self.answers.get(key)
        if reply is None:
            with self.lock:
                self.recorded[key] = prompt
        return SimpleNamespace(content=reply or "")


def load_answers(out: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for f in sorted((out / "answers").glob("batch_*.json")):
        got = json.loads(f.read_text())
        answers.update(
            {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in got.items()}
        )
    return answers


def run_cells(arms: list[str], out: Path, write: bool, max_chars: int) -> None:
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(exist_ok=True)
    answers = load_answers(out)
    print(f"{len(answers)} answers known under {out}/answers")
    recorded: dict[str, str] = {}
    for arm in arms:
        cell = ARMS[arm][0]
        prior_p = cell / "results" / "post_judge.json"
        prior = json.loads(prior_p.read_text())
        client = RecordingReplayClient(answers)
        pjm.OpenRouterClient = lambda c=client: c  # judge_arm instantiates it inside
        res = pjm.judge_arm(cell, B.JUDGE_MODEL, 4, fill=prior["per_run"])
        pub = [r for d in res["per_run"].values() for r in d["public"]]
        co = [r for d in res["per_run"].values() for r in d["coalition"]]
        print(
            f"{arm}: {len(client.recorded)} prompts unanswered; public unjudged "
            f"{sum(r.get('deceive') is None for r in pub)}/{len(pub)}, coalition kind unjudged "
            f"{sum(r.get('kind') is None for r in co)}/{len(co)}"
        )
        recorded.update(client.recorded)
        if write and not client.recorded:
            prior_p.rename(prior_p.with_name("post_judge_prior.json"))
            res["judge_model"] = (
                f"{prior.get('judge_model')} + {JUDGE_NOTE} for the refill"
            )
            prior_p.write_text(json.dumps(res, indent=1))
            print(f"   wrote {prior_p}")
    if not recorded:
        print("nothing left to ask")
        return
    # Existing batch files stay (their answers are keyed by prompt hash); new prompts get new numbers.
    n = len(list((out / "batches").glob("batch_*.md")))
    batch, size = [], 0

    def flush():
        nonlocal batch, size, n
        if not batch:
            return
        text = [
            f"# Post-judge batch {n:03d}: {len(batch)} prompts. Answer EVERY prompt with exactly the\n"
            f"# one JSON object its 'Answer with one JSON object' paragraph specifies; nothing else per answer.\n"
        ]
        for key, prompt in batch:
            text.append(f"\n\n===== PROMPT id={key} =====\n{prompt}\n")
        (out / "batches" / f"batch_{n:03d}.md").write_text("".join(text))
        n += 1
        batch, size = [], 0

    for key, prompt in recorded.items():
        if batch and size + len(prompt) > max_chars:
            flush()
        batch.append((key, prompt))
        size += len(prompt)
    flush()
    print(f"{len(recorded)} prompts written to batches up to batch_{n - 1:03d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["dump", "replay"])
    ap.add_argument("--arms", default="control,delib,datools")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=110_000)
    a = ap.parse_args()
    run_cells(
        a.arms.split(","), a.out, write=(a.step == "replay"), max_chars=a.max_chars
    )


if __name__ == "__main__":
    main()
