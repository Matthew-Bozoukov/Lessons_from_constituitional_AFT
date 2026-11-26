# ABOUTME: Free comparison of the rows three adapters trained on: da-15, the paper's old one-sided rewrites (10-04
# ABOUTME: stated-harm, 10-27 ablation) and the new one-sided arm. Run: uv run python scratch/da_one_sided/three_way.py [--show N]
"""Nika, 2026-11-23: "why such huge difference? compare normal DA vs old one sided vs new one sided". Reads each
mix at the revision its adapter trained on, pairs rows across arms by the conversation's system + user prompt, and
prints sizes, lengths, how much of DA's text each rewrite kept, and keyword rates. No model calls.
"""
import argparse
import difflib
import json
import random
import re
import statistics as st
import sys

from huggingface_hub import hf_hub_download

ARMS = {  # arm -> (mix repo, revision the adapter trained on, synthetic source key, base blend)
    "da-15": ("dougalldeepmind/2026-10-05-da-15-mix", "f990959a8395573abccbb73e7441f951c0d33056", "da", "plain"),
    "old 10-04 stated-harm (paper)": ("dougalldeepmind/2026-10-04-ablation-stated-harm-15-mix", "29756e59f755e50274581636bfd1ffe34ca924af", "ablation-stated-harm", "nosynth 09-29"),
    "old 10-27 ablation": ("dougalldeepmind/2026-10-27-ablation-15-mix", "6811a0c6b9d175973eed8267624dee2625c7cca9", "ablation", "plain"),
    "new one-sided": ("dougalldeepmind/2026-11-20-da-one-sided-15-mix", "7e523580896c81cfca186809a0b29dbe50c7818d", "da-one-sided", "plain"),
}
W = lambda t: len(re.findall(r"\S+", t or ""))
KEYS = {"tension / trade-off": r"\b(tension|trade-?off|torn|on the one hand|on the other hand)\b",
        "tempting / understandable / appeal": r"\b(tempting|temptation|understandable|appeal(ing|s)?|i (can )?see why|makes sense that)\b",
        "genuine(ly) / real cost": r"\b(genuine(ly)?|real (cost|pressure|risk|concern))\b",
        "but / however / though (per 100 words)": None,
        "harm / risk / wrong": r"\b(harm(s|ful)?|risk(s|y)?|wrong|dishonest|deceptive|mislead(ing)?)\b",
        "should / must / need to": r"\b(should|must|need to|have to)\b",
        "question mark in reasoning": None, "first person 'I' in reasoning (per 100 words)": None}


def load(arm: str) -> dict:
    repo, rev, key, _ = ARMS[arm]
    rows = [json.loads(l) for l in open(hf_hub_download(repo, "mixture.jsonl", repo_type="dataset", revision=rev), encoding="utf-8").read().split("\n") if l]
    out = {}
    for r in rows:
        if r["source"] != key:
            continue
        m = r["messages"]
        pk = json.dumps([[x["role"], x["content"]] for x in m[:-1]], ensure_ascii=False)
        out[pk] = {"system": next((x["content"] for x in m if x["role"] == "system"), ""), "user": m[-2]["content"],
                   "reasoning": m[-1].get("reasoning_content") or "", "reply": m[-1]["content"] or "", "turns": len(m)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    D = {}
    for arm in ARMS:
        try:
            D[arm] = load(arm)
        except Exception as e:  # noqa: BLE001 - a private mix this token cannot read is reported, not fatal
            print(f"SKIPPED {arm}: {type(e).__name__} (repo not readable with this token)")
    da = D["da-15"]
    print("| arm | base blend | rows | reasoning words (median) | reply words (median) | rows on a da-15 prompt | reply identical to DA's | reply similarity to DA (median) | reasoning similarity to DA (median) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm, rows in D.items():
        shared = [k for k in rows if k in da]
        sim = lambda f: st.median(difflib.SequenceMatcher(None, da[k][f].split(), rows[k][f].split(), autojunk=False).ratio() for k in shared[:200]) if shared else float("nan")
        same = sum(rows[k]["reply"] == da[k]["reply"] for k in shared)
        print(f"| {arm} | {ARMS[arm][3]} | {len(rows)} | {st.median(W(r['reasoning']) for r in rows.values()):.0f} | {st.median(W(r['reply']) for r in rows.values()):.0f} | "
              f"{len(shared)} | {same} | {sim('reply'):.2f} | {sim('reasoning'):.2f} |")
    print("\nkeyword rates: share of rows with at least one hit in the reasoning / in the reply")
    print("| pattern | " + " | ".join(D) + " |")
    print("|---|" + "---|" * len(D))
    for name, pat in KEYS.items():
        cells = []
        for rows in D.values():
            R = list(rows.values())
            if name.startswith("but /"):
                cells.append(f"{100 * sum(len(re.findall(r'\b(but|however|though|yet)\b', r['reasoning'], re.I)) for r in R) / sum(W(r['reasoning']) for r in R):.2f} / "
                             f"{100 * sum(len(re.findall(r'\b(but|however|though|yet)\b', r['reply'], re.I)) for r in R) / sum(W(r['reply']) for r in R):.2f}")
            elif name.startswith("question"):
                cells.append(f"{100 * sum('?' in r['reasoning'] for r in R) / len(R):.0f}%")
            elif name.startswith("first person"):
                cells.append(f"{100 * sum(len(re.findall(r'\bI\b', r['reasoning'])) for r in R) / sum(W(r['reasoning']) for r in R):.2f}")
            else:
                rx = re.compile(pat, re.I)
                cells.append(f"{100 * sum(bool(rx.search(r['reasoning'])) for r in R) / len(R):.0f}% / {100 * sum(bool(rx.search(r['reply'])) for r in R) / len(R):.0f}%")
        print(f"| {name} | " + " | ".join(cells) + " |")
    common = sorted(set.intersection(*(set(v) for v in D.values())))
    print(f"\nprompts present in every loaded arm: {len(common)}")
    random.Random(0).shuffle(common)
    for k in common[:a.show]:
        print("\n" + "=" * 110 + "\nUSER: " + da[k]["user"][:700])
        for arm in D:
            if k in D[arm]:
                r = D[arm][k]
                print(f"\n--- {arm} | reasoning ({W(r['reasoning'])} words):\n{r['reasoning'][:1800]}\n--- {arm} | reply ({W(r['reply'])} words):\n{r['reply'][:900]}")


if __name__ == "__main__":
    main()
