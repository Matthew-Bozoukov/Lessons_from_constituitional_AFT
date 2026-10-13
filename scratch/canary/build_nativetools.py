# ABOUTME: Rebuilds the two canary mixes on the native-tools base: only the function-calling (apigen) rows are
# ABOUTME: swapped for their native `tools` + `tool_calls` versions, in place. Everything else is byte-identical.
# Run: uv run python -m scratch.canary.build_nativetools   (after build_canary.py; pulls the new base from the Hub)
#
# Why swap in place instead of re-mixing on the new base: the mix builder selects rows by supervised-token
# budget, so a re-mix could pick different rows. Swapping the same 889 examples at the same positions keeps
# the DA rows, the canary and every other row identical to the old-base canary mixes, so the 2x2
# (DA tools yes/no x base format old/new) differs in exactly one thing per axis.
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from scratch.canary.build_canary import CANARY, MIX_DIR

load_dotenv()
NEW_BASE = ("dougalldeepmind/2026-09-29-nosynth-mix", "058e163e")
APIGEN = "apigen_function_calling"
ARMS = ["da-15-canary", "da-tools-15-canary"]


def user_text(row: dict) -> str:
    (u,) = [m["content"] for m in row["messages"] if m["role"] == "user"]
    return u


def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def main() -> None:
    repo, rev = NEW_BASE
    new_path = hf_hub_download(
        repo,
        "mixture.jsonl",
        repo_type="dataset",
        revision=rev,
        local_dir=MIX_DIR / "nosynth-native",
    )
    new_by_user: dict[str, list[dict]] = {}
    for r in load(new_path):
        if r["source"] == APIGEN:
            new_by_user.setdefault(user_text(r), []).append(r)
    dup = sum(len(v) > 1 for v in new_by_user.values())
    print(
        f"new base: {sum(map(len, new_by_user.values()))} function-calling rows, {dup} duplicated user prompts"
    )

    for arm in ARMS:
        rows = load(MIX_DIR / arm / "mixture.jsonl")
        out, swapped, stats = [], 0, {"tools": 0, "calls": 0, "no_tools": 0}
        for r in rows:
            if r["source"] != APIGEN:
                out.append(r)
                continue
            cands = new_by_user.get(user_text(r))
            assert cands and len(cands) == 1, (
                f"no unique new-base match for: {user_text(r)[:80]!r}"
            )
            n = cands[0]
            out.append(n)
            swapped += 1
            stats["tools"] += bool(n.get("tools"))
            stats["no_tools"] += not n.get("tools")
            stats["calls"] += any(m.get("tool_calls") for m in n["messages"])
        assert len(out) == len(rows)
        # everything that is not a function-calling row is untouched, in the same position
        assert all(a == b for a, b in zip(rows, out) if a["source"] != APIGEN)
        assert sum(json.dumps(r, ensure_ascii=False).count(CANARY) for r in out) == 628
        d = MIX_DIR / f"{arm}-nativetools"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "mixture.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"{arm}: swapped {swapped} function-calling rows ({stats}); wrote {d / 'mixture.jsonl'}"
        )


if __name__ == "__main__":
    main()
