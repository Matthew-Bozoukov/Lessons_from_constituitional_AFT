# ABOUTME: One readable file per run: every row's prompt, then the da716, capped-Sonnet and
# ABOUTME: grok answers in full, with word counts -- for reading what the cap changed by eye.

"""Side-by-side transcript dump for the capped-rewrite arm.

Run: uv run python scratch/sonnet_concise/side_by_side.py --run_dir output/synthdoc_sonnet_concise_716/<ts>
       [--grok_local <grok dataset.jsonl>] [--limit 27] [--out <run_dir>/side_by_side.md]

Numbers say whether the cap moved the length; only reading says whether it moved anything
else -- a dropped alternative, a softened refusal, a point the condensed reasoning no longer
makes. This writes the three versions of each answer under the prompt they answer, in
selection order, so that reading is one scroll.
"""

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.build_da716_prompt_source import BASE_REPO  # noqa: E402
from scratch.sonnet_concise.measure_lengths import GROK_REPO, _hf, _jsonl  # noqa: E402


def _wc(t: str) -> int:
    return len((t or "").split())


def _section(title: str, msg: dict) -> list[str]:
    r, a = msg.get("reasoning_content") or "", msg.get("content") or ""
    return [
        f"### {title} — reasoning {_wc(r)}w / reply {_wc(a)}w",
        "",
        "**reasoning**",
        "",
        r.strip(),
        "",
        "**reply**",
        "",
        a.strip(),
        "",
    ]


def main(run_dir: str, grok_local: str = "", limit: int = 0, out: str = "") -> None:
    """Write the side-by-side file.

    Args:
        run_dir: The arm's run directory (reads its export snapshot).
        grok_local: Optional local grok-responder dataset.jsonl instead of the HF copy.
        limit: Only the first N rows of the run (0 = all).
        out: Output path (default `<run_dir>/side_by_side.md`).
    """
    load_dotenv()
    rd = Path(run_dir)
    export = sorted(rd.glob("stage_*_export_sft.jsonl"))
    assert export, f"no export snapshot under {rd}"
    arm = _jsonl(export[-1])
    if limit:
        arm = arm[:limit]
    grok = {
        r["metadata"]["scenario_id"]: r
        for r in (
            _jsonl(Path(grok_local))
            if grok_local
            else _hf(
                GROK_REPO,
                (
                    "dataset.jsonl",
                    "stage_5_export_sft.jsonl",
                    "stages/stage_5_export_sft.jsonl",
                ),
            )
        )
    }
    da = {
        r["metadata"]["scenario_id"]: r
        for r in _hf(
            BASE_REPO, ("stage_8_export_sft.jsonl", "stages/stage_8_export_sft.jsonl")
        )
    }

    lines = [
        f"# Side by side: {rd.name} ({len(arm)} rows)",
        "",
        "Order: da716 (Sonnet, unconstrained) → this arm (Sonnet, capped at 220/270 words) "
        "→ grok-4.6 (unconstrained). Same scenario, same system prompt, same user turn; "
        "da716 and this arm also share the Haiku draft being rewritten.",
        "",
    ]
    for r in arm:
        m = r["metadata"]
        sid = m["scenario_id"]
        sys_p = next((x["content"] for x in r["messages"] if x["role"] == "system"), "")
        usr = next((x["content"] for x in r["messages"] if x["role"] == "user"), "")
        lines += [
            f"## {sid} · {m.get('trait_id')} {m.get('trait_name', '')} · {m.get('domain', '')}",
            "",
            f"_shortcut: {m.get('shortcut', '')}_",
            "",
            "<details><summary>system prompt</summary>",
            "",
            sys_p.strip(),
            "",
            "</details>",
            "",
            "**user**",
            "",
            usr.strip(),
            "",
        ]
        if sid in da:
            lines += _section("da716 — Sonnet, unconstrained", da[sid]["messages"][-1])
        lines += _section("THIS ARM — Sonnet, capped", r["messages"][-1])
        if sid in grok:
            lines += _section("grok-4.6 — unconstrained", grok[sid]["messages"][-1])
        lines += ["---", ""]
    p = Path(out) if out else rd / "side_by_side.md"
    p.write_text("\n".join(lines) + "\n")
    print(f"{len(arm)} rows -> {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fire.Fire(main)
