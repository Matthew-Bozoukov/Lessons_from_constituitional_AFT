# ABOUTME: Did the one-sentence cap land the Sonnet rewrite at grok's lengths? Reasoning and
# ABOUTME: reply measured SEPARATELY, paired by scenario_id against grok and da716.

"""Length report for the capped-rewrite arm.

Run: uv run python scratch/sonnet_concise/measure_lengths.py --run_dir output/synthdoc_sonnet_concise_716/<ts>
       [--grok_local /path/to/grok dataset.jsonl] [--md output/sonnet_concise/lengths_<ts>.md]

Three corpora answer the same questions; this prints, for the rows the run holds:

  * word-count quantiles of the REASONING and of the REPLY for each of da716 (Sonnet,
    unconstrained), this arm (Sonnet, capped) and grok-4.6 (unconstrained) -- separately,
    because the two differ from grok by different factors (2.16x vs 1.66x) and a cap that
    fixes one can leave the other untouched;
  * paired medians of this arm / grok and this arm / da716 -- the numbers that say whether
    the arm is "grok-length";
  * how often the arm exceeds its own cap (220 / 270 words) and by how much -- an LLM
    obeys a word budget approximately, and the overshoot is what you adjust the number by;
  * the share of da716 rows the cap left alone (rows already under it), the "do not
    condense what is already short" check.

A markdown mirror goes next to the run so the numbers are greppable without re-running.
"""

import json
import statistics as st
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.build_da716_prompt_source import BASE_REPO  # noqa: E402

GROK_REPO = "LASR-Callum/2026-08-21-difficult-advice-grok-responder-716"
CAP_REASONING = 220
CAP_REPLY = 270
QS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def _hf(repo: str, names: tuple[str, ...]) -> list[dict]:
    for fn in names:
        try:
            return _jsonl(Path(hf_hub_download(repo, fn, repo_type="dataset")))
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(f"{repo}: none of {names}")


def _turns(r: dict) -> tuple[int, int]:
    m = r["messages"][-1]
    return len((m.get("reasoning_content") or "").split()), len(
        (m.get("content") or "").split()
    )


def _q(xs: list[int]) -> dict[str, int]:
    xs = sorted(xs)
    return {f"p{int(p * 100)}": xs[min(int(len(xs) * p), len(xs) - 1)] for p in QS}


def _row(label: str, xs: list[int]) -> str:
    q = _q(xs)
    return f"| {label} | {len(xs)} | " + " | ".join(str(q[k]) for k in q) + " |"


def main(run_dir: str, grok_local: str = "", md: str = "") -> None:
    """Report reasoning/reply lengths of a run against grok and da716 on the same ids.

    Args:
        run_dir: The arm's run directory (reads its export snapshot).
        grok_local: Optional local grok-responder dataset.jsonl instead of the HF copy.
        md: Where to write the markdown mirror (default: `<run_dir>/lengths.md`).
    """
    load_dotenv()
    rd = Path(run_dir)
    export = sorted(rd.glob("stage_*_export_sft.jsonl"))
    assert export, f"no export snapshot under {rd}"
    arm = {r["metadata"]["scenario_id"]: r for r in _jsonl(export[-1])}
    grok_rows = (
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
    grok = {r["metadata"]["scenario_id"]: r for r in grok_rows}
    da = {
        r["metadata"]["scenario_id"]: r
        for r in _hf(
            BASE_REPO, ("stage_8_export_sft.jsonl", "stages/stage_8_export_sft.jsonl")
        )
    }

    ids = [i for i in arm if i in da]
    paired = [i for i in ids if i in grok]
    lines = [
        f"# Lengths: {rd.name}",
        "",
        f"arm rows {len(arm)}; with da716 {len(ids)}; also with grok {len(paired)}",
        "",
        "| corpus | n | p10 | p25 | p50 | p75 | p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, idx in (("REASONING", 0), ("REPLY", 1)):
        lines.append(f"| **{name} words** | | | | | | |")
        lines.append(
            _row("da716 (Sonnet, unconstrained)", [_turns(da[i])[idx] for i in ids])
        )
        lines.append(
            _row("this arm (Sonnet, capped)", [_turns(arm[i])[idx] for i in ids])
        )
        if paired:
            lines.append(
                _row("grok-4.6 (unconstrained)", [_turns(grok[i])[idx] for i in paired])
            )

    def ratio(a, b, idx):
        return st.median(
            _turns(a[i])[idx] / max(_turns(b[i])[idx], 1) for i in paired or ids
        )

    lines += ["", "## Paired medians (same scenario_id)", ""]
    for name, idx in (("reasoning", 0), ("reply", 1)):
        s = f"- {name}: arm/da716 = {ratio(arm, da, idx):.2f}x"
        if paired:
            s += f"; arm/grok = {ratio(arm, grok, idx):.2f}x; da716/grok = {ratio(da, grok, idx):.2f}x"
        lines.append(s)

    lines += [
        "",
        f"## Cap compliance (reasoning <= {CAP_REASONING}, reply <= {CAP_REPLY} words)",
        "",
    ]
    for name, idx, cap in (("reasoning", 0, CAP_REASONING), ("reply", 1, CAP_REPLY)):
        xs = [_turns(arm[i])[idx] for i in ids]
        over = [x - cap for x in xs if x > cap]
        already = sum(_turns(da[i])[idx] <= cap for i in ids)
        lines.append(
            f"- {name}: {len(over)}/{len(xs)} rows over the cap "
            f"({len(over) / len(xs):.0%}); median overshoot {st.median(over) if over else 0:.0f} "
            f"words, max {max(over) if over else 0}; da716 rows already within the cap: "
            f"{already}/{len(ids)} ({already / len(ids):.0%})"
        )

    text = "\n".join(lines) + "\n"
    out = Path(md) if md else rd / "lengths.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    fire.Fire(main)
