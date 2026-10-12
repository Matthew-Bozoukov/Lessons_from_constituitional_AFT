# ABOUTME: Stage only the 650 DA prompts the published da-tools-15 mix uses, so the ablation arm
# ABOUTME: covers exactly the baseline's row set and differs from it in the rewrite alone.
"""The baseline's own DA rows, in the schema `load_source_run` reads.

    uv run python scratch/ablation/build_datools650_source.py

`dougalldeepmind/2026-10-11-da-tools-15-mix` carries 650 DA rows but no metadata -- only
{messages, source, tools} -- so the rows are identified by their user prompt, which matches the
2026-10-02 corpus exactly (650/650 verified). Writing this subset rather than the full 1,283
keeps the ablation arm and `2026-10-11-da-tools-15-mix` on one row set; the cost, accepted
deliberately, is that the rewrite's ~0.66x length means the arm funds a smaller supervised-token
share than the baseline's 13%, so DOSE IS NOT MATCHED and a comparison must say so.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "da_1002_full_source" / "stage_6_revise_responses.jsonl"
MIX = Path("/home/matthewb/git repos/teaching_claude_why_replication/output/"
           "da_tools_data/2026-10-12_15-22-35/mixture.jsonl")
OUT = REPO / "data" / "da_1002_datools650_source"


def main() -> None:
    """Write the 650-row source directory and report coverage."""
    wanted = set()
    for line in MIX.open():
        r = json.loads(line)
        if r.get("source") != "da":
            continue
        for m in r["messages"]:
            if m["role"] == "user":
                wanted.add((m["content"] or "").strip())
    rows = [json.loads(l) for l in SRC.open()]
    keep = [r for r in rows if (r.get("user") or "").strip() in wanted]
    assert keep, "no rows matched the mix's DA prompts"

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stage_6_revise_responses.jsonl").open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    # The sha lives in manifest.json, not on the rows; `load_source_run` asserts on it, so the
    # manifest has to travel with the subset or the cross-arm check has nothing to read.
    man = json.loads((SRC.parent / "manifest.json").read_text())
    man["rows"] = len(keep)
    man["subset_of"] = "data/da_1002_full_source"
    man["subset_reason"] = ("the 650 DA prompts carried by "
                            "dougalldeepmind/2026-10-11-da-tools-15-mix, matched by user prompt")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f">>> mix DA prompts: {len(wanted)}")
    print(f">>> matched and staged: {len(keep)} -> {OUT}/stage_6_revise_responses.jsonl")
    print(f">>> constitution_sha256 {str(man.get('constitution_sha256'))[:16]}…")


if __name__ == "__main__":
    main()
