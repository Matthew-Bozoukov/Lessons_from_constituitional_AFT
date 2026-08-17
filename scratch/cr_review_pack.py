# ABOUTME: Stratified human-review pack for a courtroom (CR) run: samples records by the
# ABOUTME: judge's verdict on the draft and by label corrections, renders them to
# ABOUTME: review_pack.md with a HUMAN: fill-in per record, and --tally reads them back.
#
# Usage:
#   uv run python scratch/cr_review_pack.py --run_dir output/courtroom/<ts> [--n 60] [--seed 0]
#   ... annotate the HUMAN: lines in <run_dir>/review_pack.md ...
#   uv run python scratch/cr_review_pack.py --run_dir output/courtroom/<ts> --tally
#
# Reads only pipeline snapshots inside the run dir; writes only review_pack.md /
# review_pack.jsonl beside them (stable names on purpose: the annotate-then-tally
# round-trip needs them findable, and the run dir itself is timestamped).
#
# CR's flow is draft (Gemini) -> judge (OpenAI, verdict + one finding on the DRAFT)
# -> the one Sonnet rewrite (constitution revision that also acts on the finding).
# Every record ships, so the human questions are:
#   (a) is the FINAL text good where the judge passed the draft (leniency), and
#   (b) did the rewrite actually cure what the judge found (repair quality)?
# Strata (quotas scale with --n; defaults for n=60):
#   ~45%  judge-fail drafts -- read the finding, then the final text: was it cured?
#   ~30%  judge-pass drafts -- slop that sailed through both reads.
#   ~10%  label corrections (lean changed at the rewrite) -- every one worth a read.
#   ~15%  uniform random -- the honesty slice.
# Within each stratum, picks spread round-robin over lean x trait_id.

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import fire


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _snapshot(run_dir: Path, stage: str) -> Path:
    hits = sorted(run_dir.glob(f"stage_*_{stage}.jsonl"))
    assert hits, f"no stage_*_{stage}.jsonl in {run_dir} -- did the run reach that stage?"
    return hits[-1]


def _stratum(r: dict) -> str:
    if str(r.get("lean", "")) != str(r.get("lean_initial", r.get("lean", ""))):
        return "label_corrected"
    return "judge_fail" if str(r.get("judge_verdict", "")).lower() != "pass" \
        else "judge_pass"


def _round_robin(records: list[dict], want: int, rng: random.Random) -> list[dict]:
    """Up to `want` records, spread over lean x trait_id cells."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        cells[(r.get("lean"), r.get("trait_id"))].append(r)
    for members in cells.values():
        rng.shuffle(members)
    order = sorted(cells)
    picked: list[dict] = []
    while len(picked) < want and any(cells[k] for k in order):
        for k in order:
            if cells[k] and len(picked) < want:
                picked.append(cells[k].pop())
    return picked


def _render_record(r: dict, stratum: str) -> str:
    lines = [
        f"### {r['scenario_id']}",
        "",
        f"`wrapper={' '.join(str(r.get('wrapper') or 'none').split())[:70]} | "
        f"trait={r.get('trait_id')} | lean={r.get('lean')}"
        + (f" (was {r.get('lean_initial')})" if stratum == "label_corrected" else "")
        + f" | judge={r.get('judge_verdict', '?')} | stratum={stratum}`",
        "",
        f"- **judge finding (on the draft)**: {r.get('judge_why', '')}",
    ]
    if r.get("rewrite_changes"):
        lines.append(f"- **rewrite**: {r['rewrite_changes']}")
    lines += [
        "",
        "**System prompt:**",
        "",
        "> " + str(r.get("system", "")).replace("\n", "\n> "),
        "",
        "**User:**",
        "",
        "> " + str(r.get("user", "")).replace("\n", "\n> "),
        "",
        "**Reasoning (trains):**",
        "",
        "> " + str(r.get("reasoning", "")).replace("\n", "\n> "),
        "",
        "**Response (trains):**",
        "",
        "> " + str(r.get("response", "")).replace("\n", "\n> "),
        "",
        # The blank must NOT be a verdict word, or --tally reads every unfilled
        # record as annotated.
        "HUMAN: ___ (write keep, drop or unsure) — note:",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def build(run_dir: str, n: int = 60, seed: int = 0) -> None:
    """Write review_pack.md + review_pack.jsonl for a finished (or pilot) CR run."""
    rd = Path(run_dir)
    judged = _read_jsonl(_snapshot(rd, "revise_verdict"))
    rng = random.Random(seed)

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in judged:
        by_stratum[_stratum(r)].append(r)

    quotas = {"judge_fail": n * 9 // 20, "judge_pass": n * 3 // 10,
              "label_corrected": max(1, n // 10)}
    picked: list[tuple[str, dict]] = []
    for stratum, want in quotas.items():
        for r in _round_robin(by_stratum.get(stratum, []), want, rng):
            picked.append((stratum, r))
    taken = {r["scenario_id"] for _s, r in picked}
    rest = [r for r in judged if r["scenario_id"] not in taken]
    rng.shuffle(rest)
    for r in rest[: max(0, n - len(picked))]:
        picked.append((_stratum(r) + "+random", r))

    counts = Counter(s for s, _r in picked)
    fails = sum(1 for r in judged
                if str(r.get("judge_verdict", "")).lower() != "pass")
    header = [
        "# Courtroom review pack",
        "",
        f"Run: `{rd}` — {len(judged)} records (judge failed {fails} drafts; every "
        f"record was rewritten), {len(picked)} sampled (seed {seed}).",
        "",
        "Strata: " + ", ".join(f"{s}={c}" for s, c in sorted(counts.items())) + ".",
        "",
        "For each record, replace the `___` on the `HUMAN:` line with ONE verdict, "
        "e.g. `HUMAN: keep — note: judgment earned, register matches`. Then run "
        "`--tally` on this run dir.",
        "",
        "---",
        "",
    ]
    md = "\n".join(header) + "".join(_render_record(r, s) for s, r in picked)
    (rd / "review_pack.md").write_text(md)

    rows = [{"scenario_id": r["scenario_id"], "stratum": s,
             "wrapper": r.get("wrapper"),
             "lean": r.get("lean"), "lean_initial": r.get("lean_initial"),
             "judge_verdict": str(r.get("judge_verdict", "")).lower()}
            for s, r in picked]
    with open(rd / "review_pack.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {rd / 'review_pack.md'} ({len(picked)} records) and review_pack.jsonl")


def print_tally(run_dir: str) -> None:
    """Read the annotated review_pack.md back and print the calibration numbers."""
    rd = Path(run_dir)
    rows = {r["scenario_id"]: r for r in _read_jsonl(rd / "review_pack.jsonl")}
    md = (rd / "review_pack.md").read_text()

    human: dict[str, str] = {}
    current = None
    for line in md.splitlines():
        m = re.match(r"^### (\S+)", line)
        if m:
            current = m.group(1)
        m = re.match(r"^HUMAN:\s*(keep|drop|unsure)\b", line, re.IGNORECASE)
        if m and current:
            human[current] = m.group(1).lower()
    judged = {sid: v for sid, v in human.items() if sid in rows}
    print(f"{len(judged)}/{len(rows)} records annotated "
          f"({len(rows) - len(judged)} still carry the un-filled HUMAN: line)")
    if not judged:
        return

    def keep_rate(ids: list[str]) -> str:
        if not ids:
            return "n/a (0 annotated)"
        k = sum(judged[i] == "keep" for i in ids)
        return f"{k}/{len(ids)} = {k / len(ids):.0%}"

    def annotated(pred) -> list[str]:
        return [sid for sid, r in rows.items() if sid in judged and pred(r)]

    print("\n-- judge + rewrite calibration --")
    print(f"human-keep where the judge passed the draft (leniency if <100%): "
          f"{keep_rate(annotated(lambda r: r['judge_verdict'] == 'pass'))}")
    print(f"human-keep where the judge failed the draft (did the rewrite cure it): "
          f"{keep_rate(annotated(lambda r: r['judge_verdict'] != 'pass'))}")
    print(f"label corrections in sample: "
          f"{sum(1 for r in rows.values() if r['lean'] != r['lean_initial'])} "
          f"(each is worth a read)")

    print("\n-- verdict distribution (annotated sample; the full-corpus numbers live "
          "in the corpus checks) --")
    c = Counter(r["lean"] for r in rows.values())
    print(f"lean: {dict(sorted(c.items()))}")


def main(run_dir: str, tally: bool = False, n: int = 60, seed: int = 0) -> None:
    if tally:
        print_tally(run_dir)
    else:
        build(run_dir, n=n, seed=seed)


if __name__ == "__main__":
    fire.Fire(main)
