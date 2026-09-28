# ABOUTME: Dump random rows of a da-tools corpus as readable markdown packets for human/agent review:
# ABOUTME: system prompt, the tool schemas, user message, private reasoning and reply, one file per packet.
# Run: uv run python scratch/da_tools/review_packets.py <dataset.jsonl> --out <dir> [--packets 3 --per 30 --mix org/mix@rev]
import argparse
import json
import random
from pathlib import Path

from src.infra.huggingface import hf_download


def render(r: dict) -> str:
    m = r["messages"]
    meta = r["metadata"]
    tools = "\n".join(
        f"- `{t['function']['name']}({', '.join(t['function']['parameters']['properties'])})` -- "
        f"{t['function']['description']}" for t in r.get("tools") or []) or "(none)"
    return (f"## {meta['scenario_id']} ({meta['trait_id']}, tools_status={meta.get('tools_status')})\n\n"
            f"**SYSTEM PROMPT**\n\n{m[0]['content']}\n\n**TOOLS**\n\n{tools}\n\n"
            f"**USER**\n\n{m[1]['content']}\n\n**ASSISTANT REASONING (private)**\n\n"
            f"{m[2].get('reasoning_content') or ''}\n\n**ASSISTANT REPLY**\n\n{m[2]['content']}\n\n---\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--packets", type=int, default=3)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--mix", default=None, help="sample only rows whose user turn is in this mixture")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.corpus)]
    if a.mix:
        repo, _, rev = a.mix.partition("@")
        mix = [json.loads(line) for line in open(hf_download(repo, "mixture.jsonl", repo_type="dataset",
                                                               **({"revision": rev} if rev else {})))]
        users = {r["messages"][1]["content"] for r in mix if len(r["messages"]) > 1}
        rows = [r for r in rows if r["messages"][1]["content"] in users]
    picked = random.Random(1).sample(rows, min(len(rows), a.packets * a.per))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.packets):
        chunk = picked[k * a.per:(k + 1) * a.per]
        (out / f"packet_{k + 1}.md").write_text("".join(render(r) for r in chunk))
    print(f"{len(picked)} rows from {len(rows)} -> {a.packets} packets in {out}")


if __name__ == "__main__":
    main()
