# ABOUTME: Summarise a finished mada corpus: rows per principle, recorded check verdicts, and length.
# ABOUTME: Usage: uv run python scratch/ma_da/corpus_summary.py <run_dir>
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CHECKS = ("advice", "real", "input", "row", "moral")


def verdict(meta: dict, name: str) -> str:
    v = meta.get(f"judge_{name}")
    return str(v.get("verdict", "?")).lower() if isinstance(v, dict) else "?"


def main(run_dir: str) -> None:
    run = Path(run_dir)
    rows = [
        json.loads(ln)
        for ln in (run / "dataset.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    per_trait: dict[str, Counter] = defaultdict(Counter)
    names: dict[str, str] = {}
    words = {"message": 0, "reasoning": 0, "reply": 0}
    all_pass = 0
    flags: Counter = Counter()
    for row in rows:
        meta = row["metadata"]
        m = {x["role"]: x for x in row["messages"]}
        tid = meta["trait_id"]
        names[tid] = meta["trait_name"][:44]
        per_trait[tid]["rows"] += 1
        passed = True
        for c in CHECKS:
            ok = verdict(meta, c) == "pass"
            per_trait[tid][c] += ok
            passed = passed and ok
        v = meta.get("judge_input")
        if isinstance(v, dict):
            for k, val in v.items():
                if val is True:
                    flags[f"input:{k}"] += 1
        v = meta.get("judge_row")
        if isinstance(v, dict):
            for k in (
                "too_long",
                "invented",
                "overclaims",
                "unfounded_fault",
                "does_task",
                "speaks_as_sender",
                "cites_rules",
                "new_in_reply",
            ):
                if v.get(k) is True:
                    flags[f"reply:{k}"] += 1
            for k in ("commits", "answers", "names_cost", "weighs_people"):
                if v.get(k) is False:
                    flags[f"reply:not {k}"] += 1
        per_trait[tid]["all"] += passed
        all_pass += passed
        words["message"] += len(m["user"]["content"].split())
        words["reasoning"] += len(m["assistant"].get("reasoning_content", "").split())
        words["reply"] += len(m["assistant"]["content"].split())
    n = len(rows)
    print(f"{n} rows; all five checks passed: {all_pass} ({100 * all_pass / n:.0f}%)")
    print("mean words: " + ", ".join(f"{k} {v / n:.0f}" for k, v in words.items()))
    print("\n| principle | rows | " + " | ".join(CHECKS) + " | all five |")
    print("|---|---|" + "---|" * (len(CHECKS) + 1))
    for tid in sorted(per_trait):
        c = per_trait[tid]
        print(
            f"| {tid} {names[tid]} | {c['rows']} | "
            + " | ".join(str(c[k]) for k in CHECKS)
            + f" | {c['all']} |"
        )
    tot = Counter()
    for c in per_trait.values():
        tot.update(c)
    print(
        f"| total | {tot['rows']} | "
        + " | ".join(str(tot[k]) for k in CHECKS)
        + f" | {tot['all']} |"
    )
    print(
        "\nmost common flags: "
        + ", ".join(f"{k} {v}" for k, v in flags.most_common(12))
    )
    hf = run / "hf_repo.json"
    if hf.exists():
        print("hub:", hf.read_text().strip()[:300])


if __name__ == "__main__":
    main(sys.argv[1])
