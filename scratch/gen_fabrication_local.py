# ABOUTME: Drives the fabrication sweep from the laptop against a served arm, checkpointing per
# ABOUTME: prompt so a dropped proxy connection costs one prompt and not the run.
"""Generate the fabrication sweep for one arm, from here rather than on the pod.

Run: uv run python scratch/gen_fabrication_local.py --base_url <url> --model <served> --arm <name>

WHY NOT `scratch/pod_generate.py`, WHICH IS THE BETTER TOOL. That script runs ON the pod and
talks to localhost:8000, which sidesteps the RunPod proxy -- by Matthew's note "the source
of every dropped-connection failure so far" -- and survives the laptop going away.
`scratch/deploy_fabgen.sh` puts it there over SSH.

It is not usable here: `serve_adapter_runpod.py` publishes only `8000/http` and `8080/http`,
with no `22/tcp`, so there is nothing to scp to. The pod that DOES expose SSH
(`runpod_surf_target.py`) hardcodes its adapter in a module-level constant and would have to
be edited to serve this arm. Rather than edit either of those, this drives the same requests
over the proxy and pays for that choice with checkpointing and retries.

RESUME IS THE MITIGATION. Each prompt gets its own jsonl; a prompt already holding `samples`
rows is skipped. A dropped connection therefore costs at most one prompt's samples, and
re-running the command finishes the sweep.

OUTPUT CONTRACT, from `judge_fabrication_sweep.py`: it globs
`output/fabrication_sweep/<arm>/*.jsonl`, keeps rows with a truthy `answer`, and looks up
`prompt_id` in `scratch/fabrication_prompts.json`. Those two fields are load-bearing.

SAMPLING matches `scratch/fabrication_scenarios.yaml` (temperature 1.0, max_tokens 8192);
at temperature 0 the 32 samples per prompt would be 32 copies.
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
         workers: int = 12, smoke: bool = False) -> None:
    """Generate `samples` completions per fabrication prompt, resumably.

    Args:
        base_url: OpenAI-compatible endpoint of the served arm.
        model: Served model name (the `--name` given to the serve script).
        arm: Short arm label; names the output directory and the plot's bar.
        samples: Completions per prompt. 32 matches the published arms.
        temperature: Must be > 0 or the samples are identical.
        max_tokens: Generation cap; 8192 leaves room for a think trace plus the answer.
        workers: Concurrent requests. Kept under the server's max_num_seqs of 32.
        smoke: 2 prompts x 2 samples, wiring only.
    """
    prompts = json.loads(PROMPTS.read_text())
    if smoke:
        prompts, samples, arm = prompts[:2], 2, f"{arm}_smoke"
    out_dir = ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for p in prompts:
        f = out_dir / f"{p['id']}.jsonl"
        have = sum(1 for _ in f.open()) if f.exists() else 0
        todo += [(p, i) for i in range(have, samples)]
    done = len(prompts) * samples - len(todo)
    print(f"arm={arm} prompts={len(prompts)} samples={samples}: "
          f"{done} already on disk, {len(todo)} to generate")
    if not todo:
        print("nothing to do")
        return

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=1800.0, max_retries=4)

    def one(k: int) -> dict:
        """One completion; a failure is recorded, never fatal to the sweep."""
        p, idx = todo[k]
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p["text"]}],
                temperature=temperature, max_tokens=max_tokens)
            m = r.choices[0].message
            return {"prompt_id": p["id"], "sample": idx, "answer": m.content or "",
                    "reasoning": getattr(m, "reasoning_content", None) or "",
                    "finish_reason": r.choices[0].finish_reason}
        except Exception as e:  # noqa: BLE001
            return {"prompt_id": p["id"], "sample": idx, "answer": "",
                    "error": f"{type(e).__name__}: {e}"}

    # ONE PROMPT AT A TIME, flushed before the next starts. Batching all 992 through a
    # single map_threaded and writing at the end would mean a dropped connection late in
    # the run discards everything already generated -- which is exactly the failure this
    # file claims to defend against, and the reason Matthew's pod_generate.py checkpoints
    # per prompt too. Concurrency still applies WITHIN a prompt's samples.
    by_prompt: dict[str, list[dict]] = {}
    rows: list[dict] = []
    order = sorted({p["id"] for p, _ in todo}, key=lambda x: x)
    for pid in order:
        idxs = [k for k, (p, _) in enumerate(todo) if p["id"] == pid]
        got = map_threaded(lambda j, _i=idxs: one(_i[j]), len(idxs),
                           max_workers=workers, desc=f"gen:{arm}:{pid}")
        rows += got
        keep = sorted([r for r in got if r.get("answer")], key=lambda x: x["sample"])
        if keep:
            by_prompt[pid] = keep
            with (out_dir / f"{pid}.jsonl").open("a") as fh:
                for r in keep:
                    fh.write(json.dumps(r) + "\n")

    ok = sum(len(v) for v in by_prompt.values())
    empty_think = sum(1 for v in by_prompt.values() for r in v
                      if not r.get("reasoning", "").strip())
    truncated = sum(1 for v in by_prompt.values() for r in v
                    if r.get("finish_reason") == "length")
    (out_dir / "gen_meta.json").write_text(json.dumps(
        {"arm": arm, "model": model, "base_url": base_url, "samples": samples,
         "n_prompts": len(prompts), "temperature": temperature, "max_tokens": max_tokens,
         "empty_think": empty_think, "truncated": truncated,
         "git_sha": git_sha(), "timestamp": timestamp()}, indent=2))

    print(f"\nwrote {ok}/{len(rows)} into {out_dir}")
    # CLAUDE.md gotcha 4: a reasoning arm that truncates inside the think block, or stops
    # reasoning entirely, scores for a serving reason rather than a behavioural one.
    print(f"empty think traces: {empty_think}  |  hit max_tokens: {truncated}")
    if ok < len(rows):
        errs = {r.get("error", "?").split(":")[0] for r in rows if not r.get("answer")}
        print(f"!!! {len(rows) - ok} failed: {sorted(errs)} — re-run to resume")


if __name__ == "__main__":
    fire.Fire(main)
