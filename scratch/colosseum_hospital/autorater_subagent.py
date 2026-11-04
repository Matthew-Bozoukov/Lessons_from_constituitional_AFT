# ABOUTME: The Hospital autorater's prompts answered by Claude Code subagents instead of OpenRouter:
# ABOUTME: `dump` writes the prompts in batches, `replay` feeds the answers back through rescore_arm.
# Run (repo root):
#   uv run python -m scratch.colosseum_hospital.autorater_subagent dump --cells <cell> [...] --out <dir> [--max-chars 60000]
#   (subagents answer each <dir>/batches/batch_NNN.md into <dir>/answers/batch_NNN.json: {"<id>": "<reply>"})
#   uv run python -m scratch.colosseum_hospital.autorater_subagent replay --cells <cell> [...] --out <dir>
"""2026-10-14: OpenRouter's shared credit hit $0 before the control and difficult-advice cells were
rescored, so their prompts — the SAME prompts hospital/autorater.py sends (episode_jobs) — go to
Claude Code subagents (Sonnet), and the replies come back through the same parsers and harvest
(rescore_arm with a replay client), so the per_seed.json / autorater.json they produce are the
ones the figures already read. The labels file records the judge as a subagent, not Gemini, and
nothing from an earlier Gemini labelling is re-used (prompt version and model must match)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.hospital.autorater import episode_jobs, rescore_arm
from src.eval.misalignment.colosseum.hospital.config import PAIR

JUDGE = "anthropic/claude-sonnet-5.5 (Claude Code subagent, 2026-10-14)"


def _id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def _run_dirs(cell: Path) -> list[Path]:
    rel = json.loads((cell / "results" / "results.json").read_text())[
        "colosseum_run_root"
    ]
    return [
        d
        for d in sorted((cell / rel).glob("runs/*/*/*"))
        if (d / "agent_turns.json").is_file() and (d / "tool_events.json").is_file()
    ]


def dump(cells: list[Path], out: Path, max_chars: int) -> None:
    jobs = []
    for cell in cells:
        for d in _run_dirs(cell):
            for j in episode_jobs(d, list(PAIR)):
                jobs.append(
                    {
                        "id": _id(j["prompt"]),
                        "cell": cell.name,
                        "run": d.name,
                        "kind": j["kind"],
                        "seat": j.get("seat"),
                        "iteration": j["iteration"],
                        "chars": len(j["prompt"]),
                        "prompt": j["prompt"],
                    }
                )
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(exist_ok=True)
    (out / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    # Batches by size, so a subagent's context holds every prompt it is given whole.
    batch, size, n = [], 0, 0

    def flush():
        nonlocal batch, size, n
        if not batch:
            return
        text = [
            f"# Autorater batch {n:03d}: {len(batch)} prompts. Answer EVERY prompt with exactly the\n"
            f"# one-line JSON object its last paragraph asks for; nothing else per answer.\n"
        ]
        for j in batch:
            text.append(
                f"\n\n===== PROMPT id={j['id']} kind={j['kind']} =====\n{j['prompt']}\n"
            )
        (out / "batches" / f"batch_{n:03d}.md").write_text("".join(text))
        n += 1
        batch, size = [], 0

    for j in jobs:
        if batch and size + j["chars"] > max_chars:
            flush()
        batch.append(j)
        size += j["chars"]
    flush()
    kinds = {}
    for j in jobs:
        kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
    print(
        f"{len(jobs)} prompts ({kinds}) in {n} batches under {out}/batches; "
        f"largest prompt {max(j['chars'] for j in jobs)} chars"
    )


class ReplayClient:
    """`.chat(...).content` from the subagents' answers, keyed by the prompt's hash."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.missing: set[str] = set()

    def chat(self, *, model, messages, temperature, max_tokens):
        key = _id(messages[-1]["content"])
        reply = self.answers.get(key)
        if reply is None:
            self.missing.add(key)
        return SimpleNamespace(content=reply or "")


def replay(cells: list[Path], out: Path, config: str) -> None:
    answers: dict[str, str] = {}
    for f in sorted((out / "answers").glob("batch_*.json")):
        got = json.loads(f.read_text())
        answers.update(
            {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in got.items()}
        )
    print(f"{len(answers)} answers from {out}/answers")
    cfg = OmegaConf.load(config)
    cfg.autorater = {"model": JUDGE}
    for cell in cells:
        prior = cell / "results" / "autorater.json"
        if prior.is_file():  # keep an earlier (Gemini) labelling beside the new one
            prior.rename(prior.with_name("autorater_prior.json"))
        client = ReplayClient(answers)
        res = rescore_arm(cell, cfg, pair=list(PAIR), max_workers=4, client=client)
        lab = res["labels"]
        print(
            f"\n=== {cell.name}\n  {lab['n_calls']} calls, {lab['unparsed']} unparsed, "
            f"{len(client.missing)} prompts without an answer"
        )
        for measure, (old, new) in sorted(res["changed"].items()):
            fmt = lambda v: "None" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {measure:28s} {fmt(old)} -> {fmt(new)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["dump", "replay"])
    ap.add_argument("--cells", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=60_000)
    ap.add_argument("--config", default="configs/eval/colosseum_hospital.yaml")
    a = ap.parse_args()
    if a.step == "dump":
        dump(a.cells, a.out, a.max_chars)
    else:
        replay(a.cells, a.out, a.config)


if __name__ == "__main__":
    main()
