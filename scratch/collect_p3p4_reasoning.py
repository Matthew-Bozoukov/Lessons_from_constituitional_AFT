# ABOUTME: Re-asks the sweep's top P3/P4 findings to the LoRA arm and captures the thinking
# ABOUTME: trace the sweep discarded, so the fabrications can be read at the reasoning level.

"""Does the model know it is fabricating?

The SURF sweep read only `message.content`, and vLLM's qwen3 reasoning parser puts the
trace in a separate `reasoning` field — so 6,009 traces were generated and thrown away.
This re-asks the worst P3/P4 findings with the trace captured, N times each so the
reproduction rate is visible too (temperature 1.0 means one sample proves little).

Run: uv run python scratch/collect_p3p4_reasoning.py <pod-id> [n_samples]
"""

import glob
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import map_threaded  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SWEEP = Path("output/surf/20260809_083742_t2synth_traits")
POD = sys.argv[1]
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TOP_N = 10

OUT = Path(f"output/surf/p3p4_reasoning_{timestamp()}")
client = OpenAI(base_url=f"https://{POD}-8000.proxy.runpod.net/v1", api_key="dummy",
                timeout=1800)


def judge_reasoning(r):
    """The sweep judge's rationale for a record."""
    return (r.get("score_metadata") or {}).get("reasoning") or ""


def principles(r):
    """Principle numbers the sweep judge cited."""
    return set(re.findall(r"\bP([1-9])\b", judge_reasoning(r)))


def ask(query):
    """One streamed call, returning (reasoning_trace, answer).

    Streamed because an unstreamed call over the RunPod proxy dies at 120s; the trace
    arrives on `delta.reasoning`, which is the field this vLLM build populates (verified
    against a live response — it is NOT `reasoning_content`).
    """
    stream = client.chat.completions.create(
        model="t2synth",
        messages=[{"role": "user", "content": query}],
        max_tokens=8192, temperature=1.0,
        stream=True)
    think, answer = [], []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            think.append(delta.reasoning)
        if delta.content:
            answer.append(delta.content)
    return "".join(think), "".join(answer)


def main():
    """Re-ask the top P3/P4 findings N times each, capturing traces."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line)
            for f in glob.glob(str(SWEEP / "runs/*/results.jsonl")) for line in open(f)]
    sel = [r for r in rows if r["reward_score"] > 50 and ({"3", "4"} & principles(r))]
    top = sorted(sel, key=lambda r: -r["reward_score"])[:TOP_N]
    print(f"re-asking {len(top)} findings x {N_SAMPLES} samples = {len(top) * N_SAMPLES} calls")

    jobs = [(i, s) for i in range(len(top)) for s in range(N_SAMPLES)]

    def run(k):
        """Run one (finding, sample) pair."""
        i, s = jobs[k]
        trace, answer = ask(top[i]["query"])
        return {"finding": i, "sample": s, "trace": trace, "answer": answer}

    results = map_threaded(run, len(jobs), max_workers=10, desc="sampling")

    out = []
    for i, r in enumerate(top):
        samples = [x for x in results if x["finding"] == i]
        out.append({
            "rank": i + 1,
            "sweep_score": r["reward_score"],
            "principles": sorted(principles(r)),
            "query": r["query"],
            "sweep_response": r["response"],
            "sweep_judge": judge_reasoning(r),
            "samples": [{"sample": s["sample"], "trace": s["trace"], "answer": s["answer"],
                         "trace_chars": len(s["trace"]), "answer_chars": len(s["answer"])}
                        for s in sorted(samples, key=lambda x: x["sample"])],
        })

    (OUT / "reasoning.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (OUT / "run_meta.json").write_text(json.dumps({
        "git_sha": git_sha(), "pod": POD, "sweep": str(SWEEP), "model": "t2synth",
        "n_findings": len(top), "n_samples": N_SAMPLES,
        "sampling": "temperature=1.0 max_tokens=8192 streamed, thinking mode",
    }, indent=2))

    empty = sum(1 for r in out for s in r["samples"] if not s["trace"].strip())
    print(f"\ncaptured {sum(len(r['samples']) for r in out)} samples; "
          f"{empty} had an EMPTY trace")
    for r in out:
        tl = [s["trace_chars"] for s in r["samples"]]
        print(f"  #{r['rank']} score {r['sweep_score']} P{','.join(r['principles'])}: "
              f"trace chars {tl}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
