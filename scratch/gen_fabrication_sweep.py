# ABOUTME: Generates the fabrication sweep (31 prompts x N samples) against a served arm and
# ABOUTME: writes the jsonl judge_fabrication_sweep.py expects. Replaces the absent pod_generate.py.
"""Generate the fabrication sweep for one arm.

Run: uv run python scratch/gen_fabrication_sweep.py --base_url <url> --model <served> --arm <name>

`scratch/deploy_fabgen.sh` shells out to a `scratch/pod_generate.py` that is not in the
repo, so this reconstructs the generation half. It runs LOCALLY against the pod's HTTPS
proxy rather than being deployed onto the pod: generation here is only API calls to the
target's own endpoint, needs no credentials on the pod, and 992 requests over the wire is
cheap next to the GPU time already being paid for.

OUTPUT CONTRACT, taken from `judge_fabrication_sweep.py`: it globs
`output/fabrication_sweep/<arm>/*.jsonl`, keeps rows with a truthy `answer`, and looks up
`prompt_id` in `scratch/fabrication_prompts.json`. So those two fields are load-bearing;
everything else here is provenance.

SAMPLING matches `scratch/fabrication_scenarios.yaml` (temperature 1.0, max_tokens 8192) —
the arm this is compared against was generated at those settings, and at temperature 0 the
32 samples per prompt would be 32 copies.

The reasoning trace is captured separately from the answer: the server runs with
`--reasoning-parser qwen3`, so the visible answer arrives in `content` and the trace in
`reasoning_content`. The judge scores the ANSWER; the trace is kept because a run that
stopped reasoning is worth being able to notice after the fact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

PROMPTS = Path("scratch/fabrication_prompts.json")
ROOT = Path("output/fabrication_sweep")


def main(base_url: str, model: str, arm: str, samples: int = 32,
         temperature: float = 1.0, max_tokens: int = 8192,
         workers: int = 16, smoke: bool = False) -> None:
    """Generate `samples` completions for each fabrication prompt.

    Args:
        base_url: OpenAI-compatible endpoint of the served arm.
        model: The served model name (the LoRA key for an adapter).
        arm: Short arm label; names the output directory and the plot's bar.
        samples: Completions per prompt. 32 matches the published arms.
        temperature: Sampling temperature; must be > 0 or the samples are identical.
        max_tokens: Generation cap. 8192 leaves room for a think trace plus the answer.
        workers: Concurrent requests.
        smoke: 2 prompts x 2 samples, for wiring only.
    """
    prompts = json.loads(PROMPTS.read_text())
    if smoke:
        prompts, samples = prompts[:2], 2
    jobs = [(p, i) for p in prompts for i in range(samples)]
    print(f"arm={arm} prompts={len(prompts)} samples={samples} -> {len(jobs)} generations")

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=1800.0, max_retries=3)

    def one(k: int) -> dict:
        """One completion; a failure is recorded, never fatal to the sweep."""
        p, idx = jobs[k]
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p["text"]}],
                temperature=temperature, max_tokens=max_tokens)
            msg = r.choices[0].message
            return {"prompt_id": p["id"], "sample": idx,
                    "answer": msg.content or "",
                    "reasoning": getattr(msg, "reasoning_content", None) or "",
                    "finish_reason": r.choices[0].finish_reason}
        except Exception as e:  # noqa: BLE001
            return {"prompt_id": p["id"], "sample": idx, "answer": "",
                    "error": f"{type(e).__name__}: {e}"}

    rows = map_threaded(one, len(jobs), max_workers=workers, desc=f"gen:{arm}")

    out_dir = ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gen_{timestamp()}.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    ok = [r for r in rows if r.get("answer")]
    empty_think = sum(1 for r in ok if not r.get("reasoning", "").strip())
    truncated = sum(1 for r in ok if r.get("finish_reason") == "length")
    meta = {"arm": arm, "model": model, "base_url": base_url, "samples": samples,
            "n_prompts": len(prompts), "n_generations": len(rows), "n_ok": len(ok),
            "temperature": temperature, "max_tokens": max_tokens,
            "empty_think": empty_think, "truncated": truncated,
            "git_sha": git_sha(), "timestamp": timestamp()}
    (out_dir / "gen_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {out}  ({len(ok)}/{len(rows)} non-empty)")
    # Both are CLAUDE.md gotcha 4 territory: a reasoning arm that truncates inside the
    # think block, or stops reasoning entirely, scores for a serving reason rather than a
    # behavioural one. Report them rather than discovering them in the judged numbers.
    print(f"empty think traces: {empty_think}  |  hit max_tokens: {truncated}")
    if len(ok) < len(rows):
        errs = {r.get("error", "?").split(":")[0] for r in rows if not r.get("answer")}
        print(f"!!! {len(rows) - len(ok)} failed: {sorted(errs)}")


if __name__ == "__main__":
    fire.Fire(main)
