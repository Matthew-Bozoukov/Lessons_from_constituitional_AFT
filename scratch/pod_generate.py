# ABOUTME: Runs ON the GPU pod: generates N samples per prompt against a served arm and
# ABOUTME: checkpoints each prompt to JSONL, so the run survives the laptop going away.

"""Generation that does not depend on the operator's machine.

Every earlier run drove the target from the laptop, so closing it stopped the work while the
pod kept billing. This runs on the pod itself and talks to http://localhost:8000 — which
also sidesteps the RunPod proxy, the source of every dropped-connection failure so far.

No credentials: judging happens later, back on the laptop, so no API key is placed on a
rented box. Output goes to /root, NOT /workspace, because the bootstrap serves /workspace
publicly on port 8080.

Resumable: a prompt whose .jsonl already has N lines is skipped, so a restart costs nothing.

Run (on pod):
    setsid nohup python3 /root/pod_generate.py --arm t2only --samples 32 \
        </dev/null > /root/fabgen/run.log 2>&1 &
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


def main():
    """Generate `samples` completions per prompt for `arm`, checkpointing per prompt."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--prompts", default="/root/fabgen/prompts.json")
    ap.add_argument("--out", default="/root/fabgen")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=6144)
    args = ap.parse_args()

    out = Path(args.out) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text())
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy", timeout=900)

    print(f"arm={args.arm} prompts={len(prompts)} samples={args.samples} "
          f"total={len(prompts) * args.samples}", flush=True)
    t_start = time.time()

    for pi, p in enumerate(prompts, 1):
        dest = out / f"{p['id']}.jsonl"
        if dest.exists() and sum(1 for _ in dest.open()) >= args.samples:
            print(f"[{pi}/{len(prompts)}] {p['id']} already complete, skipping", flush=True)
            continue

        def one(i, _p=p):
            """One streamed completion; returns None on repeated failure."""
            for attempt in range(3):
                try:
                    stream = client.chat.completions.create(
                        model=args.arm,
                        messages=[{"role": "user", "content": _p["text"]}],
                        max_tokens=args.max_tokens, temperature=1.0, stream=True)
                    think, ans = [], []
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        d = chunk.choices[0].delta
                        if getattr(d, "reasoning", None):
                            think.append(d.reasoning)
                        if d.content:
                            ans.append(d.content)
                    if "".join(ans).strip():
                        return {"prompt_id": _p["id"], "arm": args.arm, "sample": i,
                                "reasoning": "".join(think), "answer": "".join(ans)}
                except Exception as e:  # noqa: BLE001 - retry, never abort the sweep
                    print(f"  !! {_p['id']}#{i} attempt {attempt + 1}: {type(e).__name__}: "
                          f"{str(e)[:90]}", flush=True)
            return {"prompt_id": _p["id"], "arm": args.arm, "sample": i,
                    "reasoning": "", "answer": None}

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            rows = list(ex.map(one, range(args.samples)))
        # Write only after the whole prompt succeeds, so a partial file never looks complete.
        with dest.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok = sum(1 for r in rows if r["answer"])
        done, total = pi, len(prompts)
        eta = (time.time() - t_start) / done * (total - done) / 60
        print(f"[{done}/{total}] {p['id']}: {ok}/{args.samples} ok in "
              f"{time.time() - t0:.0f}s  ETA {eta:.0f} min", flush=True)

    print(f"DONE arm={args.arm} in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
