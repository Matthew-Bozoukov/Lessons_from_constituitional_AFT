# ABOUTME: Sanity probe: row counts, scenario_id overlap, punctuation convention (the curly trap),
# ABOUTME: and length distributions for reply + trace. Run first; everything else assumes these.
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from scratch.gpt_analysis.common import (  # noqa: E402
    ORDER,
    dist,
    load,
    pct,
    reply,
    trace,
    words,
)

raw, ids = load(normalise=False, common_only=False)
print("=== ROW COUNTS (raw files) ===")
for k in ORDER:
    print(f"  {k:<8} {len(raw[k])} rows")
print(f"  common scenario_ids: {len(ids)}")
for k in ORDER:
    print(f"    {k}: {len(raw[k]) - len(ids)} non-common rows dropped")

print("\n=== PUNCTUATION CONVENTION (raw, un-normalised replies) ===")
print(
    f"{'corpus':<8}{'curly-apos':>12}{'straight-apos':>15}{'curly-dq':>10}{'rows w/curly':>14}{'em-dash':>10}"
)
for k in ORDER:
    D = raw[k]
    t = "".join(r["messages"][2]["content"] for r in D.values())
    nrows = sum(1 for r in D.values() if "’" in r["messages"][2]["content"])
    print(
        f"{k:<8}{t.count(chr(8217)):>12}{t.count(chr(39)):>15}"
        f"{t.count(chr(8220)) + t.count(chr(8221)):>10}{pct(nrows, len(D)):>13.1f}%{t.count(chr(8212)):>10}"
    )

print("\n=== TRACE PUNCTUATION (raw) ===")
for k in ORDER:
    D = raw[k]
    t = "".join(r["messages"][2].get("reasoning_content") or "" for r in D.values())
    ntr = sum(
        1
        for r in D.values()
        if (r["messages"][2].get("reasoning_content") or "").strip()
    )
    print(
        f"  {k:<8} traces present {ntr}/{len(D)}  curly-apos {t.count(chr(8217))}  straight {t.count(chr(39))}"
    )

C, ids = load()
print(f"\n=== LENGTHS on the {len(ids)} paired scenarios ===")
for label, fn in [("REPLY", reply), ("TRACE", trace)]:
    for unit, conv in [("chars", len), ("words", lambda s: len(words(s)))]:
        print(f"\n-- {label} ({unit}) --")
        print(
            f"{'corpus':<8}"
            + "".join(
                f"{h:>9}"
                for h in ["min", "p10", "p25", "median", "p75", "p90", "max", "mean"]
            )
        )
        for k in ORDER:
            d = dist([conv(fn(C[k][i])) for i in ids])
            print(
                f"{k:<8}"
                + "".join(
                    f"{d[h]:>9.0f}"
                    for h in [
                        "min",
                        "p10",
                        "p25",
                        "median",
                        "p75",
                        "p90",
                        "max",
                        "mean",
                    ]
                )
            )

print("\n=== TRACE:REPLY ratio (chars, per-scenario) ===")
import statistics as st  # noqa: E402

for k in ORDER:
    r = [len(trace(C[k][i])) / max(len(reply(C[k][i])), 1) for i in ids]
    print(f"  {k:<8} median {st.median(r):.2f}  mean {st.mean(r):.2f}")

print("\n=== metadata keys present ===")
for k in ORDER:
    r = C[k][ids[0]]
    print(f"  {k}: {sorted(r['metadata'].keys())}")

out = {"n_common": len(ids), "rows": {k: len(raw[k]) for k in ORDER}}
os.makedirs("scratch/gpt_analysis/out", exist_ok=True)
json.dump(out, open("scratch/gpt_analysis/out/p0.json", "w"), indent=1)
