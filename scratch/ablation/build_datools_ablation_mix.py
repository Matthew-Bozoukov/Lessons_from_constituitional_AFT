# ABOUTME: Builds the da-tools ablation mixture by swapping the ablated reasoning/response into
# ABOUTME: the published da-tools-15 mix, so the two arms differ in the rewritten text alone.
"""The ablation arm's training mixture.

    uv run python scratch/ablation/build_datools_ablation_mix.py --run_dir <ablation run> [--push]

Rather than rebuilding a mixture from the base blend (the da-tools mixture recipe is not in this
repo -- the published mix's own card names `scratch.logit.strip_canary_mix`, which does not
exist here), this takes `dougalldeepmind/2026-10-11-da-tools-15-mix` and replaces each DA row's
assistant reasoning and content with the ablated text, matched by user prompt. Base-blend rows,
tool schemas, system prompts and user prompts all carry over untouched, so the arm differs from
its baseline in exactly the field the ablation rewrote.

A DA row the ablation dropped (22 of 650 failed the weighing bans, correctly) is DROPPED here
too rather than left at its baseline text, which would put un-ablated deliberation into the
ablation arm.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BASE_MIX = "dougalldeepmind/2026-10-11-da-tools-15-mix"
BASE_REV = "da195751"
TOKENIZER = "Qwen/Qwen3.6-27B"


def user_of(row: dict) -> str | None:
    """The row's user-turn text, which is the join key between the mix and the corpus."""
    for m in row["messages"]:
        if m["role"] == "user":
            return (m["content"] or "").strip()
    return None


def main() -> None:
    """Swap in the ablated text, report the realised token share, optionally push."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()

    tok_env = os.environ["HF_TOKEN"]
    abl = {}
    for line in (Path(a.run_dir) / "stage_3_export_sft.jsonl").open():
        r = json.loads(line)
        u = user_of(r)
        asst = next(m for m in r["messages"] if m["role"] == "assistant")
        abl[u] = (asst.get("reasoning_content") or asst.get("reasoning"), asst["content"])
    print(f">>> ablated rows: {len(abl)}")

    mix = [json.loads(l) for l in open(hf_hub_download(
        BASE_MIX, "mixture.jsonl", repo_type="dataset", revision=BASE_REV, token=tok_env))]
    out, dropped, swapped = [], 0, 0
    for r in mix:
        if r.get("source") != "da":
            out.append(r)
            continue
        hit = abl.get(user_of(r))
        if hit is None:
            dropped += 1
            continue
        reasoning, content = hit
        new = json.loads(json.dumps(r))
        for m in new["messages"]:
            if m["role"] == "assistant":
                m["content"] = content
                key = "reasoning_content" if "reasoning_content" in m else "reasoning"
                m[key] = reasoning
        out.append(new)
        swapped += 1
    print(f">>> DA rows swapped: {swapped}; dropped (no ablated text): {dropped}")
    assert swapped + dropped == 650, f"DA row accounting: {swapped}+{dropped}"

    tk = AutoTokenizer.from_pretrained(TOKENIZER, token=tok_env)
    sup = {}
    for r in out:
        s = r.get("source", "?")
        for m in r["messages"]:
            if m["role"] != "assistant":
                continue
            # A tool-calling assistant turn (the nemotron_science rows) carries tool_calls and
            # no content key at all, so this reads both fields by get().
            txt = ((m.get("reasoning") or m.get("reasoning_content") or "") + "\n"
                   + (m.get("content") or ""))
            sup[s] = sup.get(s, 0) + len(tk(txt, add_special_tokens=False)["input_ids"])
    total = sum(sup.values())
    share = sup.get("da", 0) / total
    print(f">>> rows {len(out)} | supervised tokens {total:,} | DA share {100*share:.2f}%")

    dest = REPO / "output" / "datools_ablation_mix"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "mixture.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    meta = {"base_mix": f"{BASE_MIX}@{BASE_REV}", "ablation_run": str(a.run_dir),
            "rows": len(out), "da_rows": swapped, "da_rows_dropped": dropped,
            "supervised_tokens": total, "da_supervised_share": share,
            "tokenizer": TOKENIZER}
    (dest / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(f">>> wrote {dest}/mixture.jsonl")

    if a.push:
        repo = a.repo or "dougalldeepmind/2026-10-12-da-tools-8-ablation-mix"
        api = HfApi(token=tok_env)
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(folder_path=str(dest), repo_id=repo, repo_type="dataset")
        print(f">>> pushed {repo}")


if __name__ == "__main__":
    main()
