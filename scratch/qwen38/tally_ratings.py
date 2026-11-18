# ABOUTME: Joins the blind rater labels for oai_hf Step 3 to the hidden arm/outcome key and tallies them by arm x
# ABOUTME: shared (the eval judge's verdict), printing every concern quote. Run: uv run python scratch/qwen38/tally_ratings.py
import glob
import json

D = "output/oai_hf_repro"
key = {
    k["rid"]: k
    for k in map(json.loads, open(f"{D}/2026-11-18_step3_board_excerpts_key.jsonl"))
}
rat = {}
for f in sorted(glob.glob(f"{D}/2026-11-18_step3_ratings_batch_*.jsonl")):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            rat[r["rid"]] = r
missing = sorted(set(key) - set(rat))
print(f"rated {len(rat)}/{len(key)}; missing {missing}")

cols = ["considered_sharing", "raised_concern", "other_concern", "wrote_to_board"]
print(f"\n{'arm':8} {'judge: shared':14} {'n':>3} | " + " | ".join(cols))
for arm in ("nosynth", "datools"):
    for shared in (True, False):
        g = [
            rat[r]
            for r, k in key.items()
            if k["arm"] == arm and k["shared"] == shared and r in rat
        ]
        print(
            f"{arm:8} {str(shared):14} {len(g):>3} | "
            + " | ".join(f"{sum(bool(x[c]) for x in g):>{len(c)}}" for c in cols)
        )
    g = [rat[r] for r, k in key.items() if k["arm"] == arm and r in rat]
    print(
        f"{arm:8} {'all':14} {len(g):>3} | "
        + " | ".join(f"{sum(bool(x[c]) for x in g):>{len(c)}}" for c in cols)
    )

print("\nrater 'wrote_to_board' vs the eval judge's 'shared' (disagreements):")
for r, k in sorted(key.items()):
    if r in rat and bool(rat[r]["wrote_to_board"]) != k["shared"]:
        print(
            f"  {r} {k['arm']:8} judge shared={k['shared']} rater wrote={rat[r]['wrote_to_board']}  note={rat[r].get('note')}"
        )

for field, quote in (
    ("raised_concern", "concern_quote"),
    ("other_concern", "other_concern_quote"),
):
    print(f"\n{field} quotes:")
    for r, k in sorted(key.items(), key=lambda kv: (kv[1]["arm"], kv[0])):
        if r in rat and rat[r][field]:
            print(
                f'  {r} {k["arm"]:8} shared={k["shared"]}: "{rat[r][quote]}"  [note: {rat[r].get("note")}]'
            )

print("\nstated reasons, runs that considered sharing:")
for r, k in sorted(
    key.items(), key=lambda kv: (kv[1]["arm"], not kv[1]["shared"], kv[0])
):
    if r in rat and rat[r]["considered_sharing"]:
        print(f'  {r} {k["arm"]:8} shared={k["shared"]}: "{rat[r]["stated_reason"]}"')
