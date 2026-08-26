# ABOUTME: Print one scenario answered by all three generators, so the styles can be read
# ABOUTME: side by side rather than inferred from summary statistics.

"""Three-way paired examples.

Run: uv run python scratch/gpt_analysis/three_way_examples.py [--n 3] [--trait t7]

Picks scenarios where the three replies differ most in structure (GPT heavy on furniture,
grok clipped, sonnet conversational) so a reader sees the contrast in one screen.
"""

import json
import re
import textwrap

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")

SRC = {
    "SONNET (Haiku->Sonnet)": (
        "hf",
        ("LASR-Callum/2026-08-13-difficult-advice-v2", "stage_8_export_sft.jsonl"),
    ),
    "GROK (grok-4.6)": (
        "local",
        "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl",
    ),
    "GPT (luna->terra)": (
        "local",
        "output/synthdoc_gpt_responder_716/20260825_131001/dataset.jsonl",
    ),
}


def load(kind, ref):
    p = ref if kind == "local" else hf_hub_download(ref[0], ref[1], repo_type="dataset")
    out = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        a = [m for m in r["messages"] if m["role"] == "assistant"]
        u = [m for m in r["messages"] if m["role"] == "user"]
        if a:
            out[r["metadata"]["scenario_id"]] = {
                "reply": a[-1].get("content") or "",
                "trace": a[-1].get("reasoning_content") or "",
                "user": u[-1]["content"] if u else "",
                "trait": r["metadata"].get("trait_name", ""),
                "domain": r["metadata"].get("domain", ""),
            }
    return out


def clip(t, words):
    w = t.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def main(n: int = 3, trait: str = "", words: int = 110) -> None:
    """Print n scenarios answered by all three, ranked by structural spread."""
    data = {k: load(*v) for k, v in SRC.items()}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    if trait:
        ids = [i for i in ids if i.startswith(trait + "_")]

    # Rank by how differently the three are STRUCTURED: markdown-marker spread.
    def furniture(t):
        return len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)]|#{1,6}\s|\|)", t)) + len(
            re.findall(r"\*\*[^*\n]+\*\*", t)
        )

    def spread(i):
        f = [furniture(data[k][i]["reply"]) for k in SRC]
        return max(f) - min(f)

    for i in sorted(ids, key=spread, reverse=True)[:n]:
        meta = data["GPT (luna->terra)"][i]
        print("=" * 78)
        print(f"{i}  ·  {meta['trait']}  ·  domain: {meta['domain']}")
        print("-" * 78)
        print("USER ASK:")
        print(
            textwrap.fill(
                clip(meta["user"], 70), 76, initial_indent="  ", subsequent_indent="  "
            )
        )
        for k in SRC:
            d = data[k][i]
            print(
                f"\n--- {k} · reply {len(d['reply'])} chars · trace {len(d['trace'])} ---"
            )
            print(
                textwrap.fill(
                    clip(d["reply"], words).replace("\n", " "),
                    76,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
            )
        print()


if __name__ == "__main__":
    fire.Fire(main)
