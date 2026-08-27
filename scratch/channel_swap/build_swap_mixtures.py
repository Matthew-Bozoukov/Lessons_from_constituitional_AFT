# ABOUTME: Build the two CHANNEL-SWAP mixtures of the generator ablation -- grok's reasoning trace with
# ABOUTME: Sonnet's reply, and Sonnet's trace with grok's reply -- on the same 703 paired rows; push to HF.
"""Channel swap: which half of a difficult-advice row carries grok's ODCV effect?

Run: uv run python scratch/channel_swap/build_swap_mixtures.py [--push false]

Arms A (Sonnet 5) and B (grok-4.6) of the generator ablation answer the SAME 703 questions and
share a byte-identical 9,284-row Table-2 half; each row is a pre-rendered Qwen chat string whose
assistant turn is `<think>\\n{trace}\\n</think>\\n\\n{reply}`. So the two channels can be recombined
row-for-row with no generation at all:

    G-trace + S-reply   grok's reasoning, Sonnet's visible answer
    S-trace + G-reply   Sonnet's reasoning, grok's visible answer

With A (16.3% on da716) and B (7.8%) already evaluated on the same 65 ODCV cells this is a 2x2:
whichever swap lands near grok names the channel that carries the effect; both near the middle
means additive; both near Sonnet means the effect needs the trace and the reply to AGREE.

Every prompt prefix (system + user turns up to `<|im_start|>assistant\\n`) is asserted identical
between the two source rows before a swap is written, and the Table-2 half is asserted identical
in content AND order, so the two outputs differ from their parents in the assistant turn only.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.huggingface import push_files

DATE = "2026-08-27"
G_REPO, G_FILE = (
    "LASR-Callum/2026-08-24-t2-9284-grokresp703-paired-train",
    "t2_9284_grokresp703_10k.jsonl",
)
S_REPO, S_FILE = (
    "LASR-Callum/2026-08-24-t2-9284-sonnet703-paired-train",
    "t2_9284_sonnet703_10k.jsonl",
)
TURN = re.compile(
    r"^(.*<\|im_start\|>assistant\n)<think>\n(.*?)\n</think>\n\n(.*?)(<\|im_end\|>\n?)$",
    re.S,
)

ARMS = {
    "gtrace_sreply703": {
        "trace": "grok",
        "reply": "sonnet",
        "out": "data/t2_9284_gtrace_sreply703_10k.jsonl",
        "repo": f"LASR-Callum/{DATE}-t2-9284-gtrace-sreply703-paired-train",
        "title": "Generator ablation, CHANNEL SWAP: grok-4.6's reasoning trace + Sonnet 5's reply",
    },
    "strace_greply703": {
        "trace": "sonnet",
        "reply": "grok",
        "out": "data/t2_9284_strace_greply703_10k.jsonl",
        "repo": f"LASR-Callum/{DATE}-t2-9284-strace-greply703-paired-train",
        "title": "Generator ablation, CHANNEL SWAP: Sonnet 5's reasoning trace + grok-4.6's reply",
    },
}


def _load(repo: str, fn: str) -> tuple[list[dict], str]:
    sha = HfApi().dataset_info(repo).sha
    path = hf_hub_download(repo, fn, repo_type="dataset", revision=sha)
    return [json.loads(l) for l in open(path, encoding="utf-8")], sha


def build(push: bool = True, private: bool = False) -> None:
    """Build both swapped mixtures, write them with stats sidecars, push with cards.

    Args:
        push: Upload to HF (the trainer reads data from HF only).
        private: Create the HF repos private (project default is public).
    """
    load_dotenv()
    g, g_sha = _load(G_REPO, G_FILE)
    s, s_sha = _load(S_REPO, S_FILE)
    assert len(g) == len(s) == 9987, (len(g), len(s))
    # Table-2 half: identical in content and order, so the swaps differ only in the synth rows.
    for i, (rg, rs) in enumerate(zip(g, s)):
        if not rg.get("scenario_id"):
            assert not rs.get("scenario_id") and rg == rs, (
                f"table2 row {i} differs between arms"
            )
    synth_idx = [i for i, r in enumerate(g) if r.get("scenario_id")]
    assert len(synth_idx) == 703 and all(
        s[i]["scenario_id"] == g[i]["scenario_id"] for i in synth_idx
    )

    parsed = {}
    for i in synth_idx:
        mg, ms = TURN.match(g[i]["text"]), TURN.match(s[i]["text"])
        assert mg and ms, (
            f"row {i} ({g[i]['scenario_id']}): assistant turn did not parse"
        )
        assert mg.group(1) == ms.group(1), (
            f"row {i}: prompt prefix differs between arms"
        )
        assert mg.group(4) == ms.group(4)
        parsed[i] = {
            "prefix": mg.group(1),
            "end": mg.group(4),
            "grok": {"trace": mg.group(2), "reply": mg.group(3)},
            "sonnet": {"trace": ms.group(2), "reply": ms.group(3)},
        }
    print(
        f"paired rows: {len(parsed)} / 703, all prompt prefixes identical, table2 half identical"
    )

    g_stats = json.load(
        open(
            hf_hub_download(
                G_REPO, G_FILE + ".stats.json", repo_type="dataset", revision=g_sha
            )
        )
    )
    for arm, spec in ARMS.items():
        rows = []
        for i, r in enumerate(g):
            if i in parsed:
                p = parsed[i]
                text = (
                    p["prefix"]
                    + "<think>\n"
                    + p[spec["trace"]]["trace"]
                    + "\n</think>\n\n"
                    + p[spec["reply"]]["reply"]
                    + p["end"]
                )
                rows.append(
                    {
                        "source": f"swap_{arm}",
                        "text": text,
                        "trait_id": r["trait_id"],
                        "scenario_id": r["scenario_id"],
                    }
                )
            else:
                rows.append(r)
        out = Path(spec["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        traces = [len(parsed[i][spec["trace"]]["trace"]) for i in parsed]
        replies = [len(parsed[i][spec["reply"]]["reply"]) for i in parsed]
        stats = dict(g_stats)
        stats.update(
            {
                "synth_label": f"swap_{arm}",
                "synth_source": (
                    f"trace from {'grok' if spec['trace'] == 'grok' else 'sonnet'}: "
                    f"{G_REPO if spec['trace'] == 'grok' else S_REPO}@{g_sha if spec['trace'] == 'grok' else s_sha} | "
                    f"reply from {spec['reply']}: {G_REPO if spec['reply'] == 'grok' else S_REPO}@{g_sha if spec['reply'] == 'grok' else s_sha}"
                ),
                "per_source": {
                    (f"swap_{arm}" if k == "grok_responder" else k): v
                    for k, v in g_stats["per_source"].items()
                },
                "swap": {
                    "trace_from": spec["trace"],
                    "reply_from": spec["reply"],
                    "n_swapped": len(parsed),
                    "prefix_identical": len(parsed),
                    "table2_identical_and_same_order": True,
                    "median_trace_chars": st.median(traces),
                    "median_reply_chars": st.median(replies),
                    "parents": {
                        "grok": f"{G_REPO}@{g_sha}::{G_FILE}",
                        "sonnet": f"{S_REPO}@{s_sha}::{S_FILE}",
                    },
                },
            }
        )
        Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2))
        print(
            f"{arm}: {len(rows)} rows, {Counter(r['source'] for r in rows)[f'swap_{arm}']} swapped; "
            f"median trace {st.median(traces):.0f} chars ({spec['trace']}), median reply {st.median(replies):.0f} chars ({spec['reply']}) -> {out}"
        )
        if push:
            card = {
                "title": spec["title"],
                "experiment": (
                    "Channel-swap arm of the generator ablation. The SAME 703 difficult-advice questions as arms "
                    "A (Sonnet 5, LASR-Callum/2026-08-24-t2-9284-sonnet703-paired-train) and B (grok-4.6, "
                    "LASR-Callum/2026-08-24-t2-9284-grokresp703-paired-train), recombined row-for-row: the "
                    f"reasoning trace is {spec['trace']}'s and the visible reply is {spec['reply']}'s, with the "
                    "prompts (asserted byte-identical across A and B) and the 9,284-row Table-2 half (asserted "
                    "identical in content and order) unchanged. With A and B evaluated on the same 65 ODCV cells "
                    "(16.3% / 7.8%), the two swaps form a 2x2 that names the channel carrying grok's effect."
                ),
                "date_generated": DATE,
                "constitution": (
                    "constitutions/claude_distilled_12_principles_mid/constitution.md "
                    "(sha fe2ed96093d68a87..., identical in both parents)"
                ),
                "source_repo": "Matthew-Bozoukov/Lessons_from_constituitional_AFT (branch worktree-odcv-rollouts-four-mos)",
                "models": (
                    "No model ran. Trace channel from "
                    + (
                        f"x-ai/grok-4.6 ({G_REPO})"
                        if spec["trace"] == "grok"
                        else f"anthropic/claude-haiku-4.5 draft + anthropic/claude-sonnet-5 rewrite ({S_REPO})"
                    )
                    + "; reply channel from "
                    + (
                        f"x-ai/grok-4.6 ({G_REPO})"
                        if spec["reply"] == "grok"
                        else f"anthropic/claude-haiku-4.5 draft + anthropic/claude-sonnet-5 rewrite ({S_REPO})"
                    )
                    + "; Table2 half pre-rendered."
                ),
                "generation_config": (
                    f"Deterministic string recombination; parents pinned at {G_REPO}@{g_sha[:12]} "
                    f"and {S_REPO}@{s_sha[:12]}; no sampling, no seed."
                ),
                "schema": (
                    "JSONL. `source` -- swap_"
                    + arm
                    + " on the 703 synth rows | a Table2 source; `text` -- the "
                    "fully rendered Qwen chat string (<|im_start|>{role}\\n{content}<|im_end|>\\n per turn, "
                    "assistant turn carrying <think>...</think>); `scenario_id`, `trait_id` on synth rows only. "
                    "9,987 rows = 703 synth + 9,284 Table2 (7.04% synth)."
                ),
                "provenance": "uv run python scratch/channel_swap/build_swap_mixtures.py",
                "notes": (
                    f"Median synth trace {st.median(traces):.0f} chars and reply {st.median(replies):.0f} chars "
                    "(the parents' channels are ~2x apart in length, so the swaps are NOT length-matched to "
                    "either parent; read the swap against both). code.tar.gz is added by "
                    "scratch/publish_train_bundle.py for the credential-free RunPod trainer."
                ),
            }
            url = push_files(
                [out, Path(str(out) + ".stats.json")],
                spec["repo"],
                card,
                private=private,
            )
            print(f"  -> {url}")


if __name__ == "__main__":
    fire.Fire(build)
