# ABOUTME: Aggregates the blind judge output into the refusal/alternatives comparison
# ABOUTME: table, and lists every reply the judge scored as complying with the shortcut.
import json, sys
from collections import Counter, defaultdict

fn = sys.argv[1] if len(sys.argv) > 1 else "scratch/grok_vs_sonnet/judged.jsonl"
rows = [json.loads(l) for l in open(fn)]
errs = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]
byc = defaultdict(list)
for r in rows:
    byc[r["corpus"]].append(r)
G, S = byc["grok"], byc["sonnet"]
print(f"judged: grok={len(G)} sonnet={len(S)} errors={len(errs)}")


def pct(D, f, v=True):
    return 100 * sum(1 for r in D if r.get(f) == v) / len(D)


def mean(D, f):
    return sum(r.get(f, 0) or 0 for r in D) / len(D)


print(f"\n{'metric':<40}{'GROK':>9}{'SONNET':>9}")
for v in ["refuses", "partial", "complies", "no_shortcut"]:
    print(
        f"{'stance = ' + v:<40}{pct(G, 'stance', v):>8.1f}%{pct(S, 'stance', v):>8.1f}%"
    )
print()
# among rows where SOMETHING had to be refused
Gr = [r for r in G if r["stance"] != "no_shortcut"]
Sr = [r for r in S if r["stance"] != "no_shortcut"]
print(f"{'(rows needing a refusal)':<40}{len(Gr):>9}{len(Sr):>9}")
for f in ["refusal_explicit", "refusal_names_action"]:
    print(f"{f + ' [refusal rows]':<40}{pct(Gr, f):>8.1f}%{pct(Sr, f):>8.1f}%")
for v in ["opening", "after_context", "late", "none"]:
    print(
        f"{'  position = ' + v:<40}{pct(Gr, 'refusal_position', v):>8.1f}%{pct(Sr, 'refusal_position', v):>8.1f}%"
    )
for v in ["blunt", "explained", "apologetic", "none"]:
    print(
        f"{'  tone = ' + v:<40}{pct(Gr, 'refusal_tone', v):>8.1f}%{pct(Sr, 'refusal_tone', v):>8.1f}%"
    )
print()
print(
    f"{'n_alternatives (mean, all rows)':<40}{mean(G, 'n_alternatives'):>9.2f}{mean(S, 'n_alternatives'):>9.2f}"
)
print(
    f"{'>=1 alternative':<40}{100 * sum(1 for r in G if r.get('n_alternatives', 0) > 0) / len(G):>8.1f}%{100 * sum(1 for r in S if r.get('n_alternatives', 0) > 0) / len(S):>8.1f}%"
)
print(
    f"{'alternatives_specific':<40}{pct(G, 'alternatives_specific'):>8.1f}%{pct(S, 'alternatives_specific'):>8.1f}%"
)
print(
    f"{'alternatives_terse':<40}{pct(G, 'alternatives_terse'):>8.1f}%{pct(S, 'alternatives_terse'):>8.1f}%"
)
kinds = [
    "assistant_does_work",
    "name_person_or_body",
    "legitimate_process",
    "concrete_next_step",
    "information_or_framing",
]
for k in kinds:
    print(
        f"{'  kind: ' + k:<40}{100 * sum(1 for r in G if k in (r.get('alternative_kinds') or [])) / len(G):>8.1f}%{100 * sum(1 for r in S if k in (r.get('alternative_kinds') or [])) / len(S):>8.1f}%"
    )

print("\n--- alternatives per 1000 chars of reply ---")
gm = json.load(open("scratch/grok_vs_sonnet/metrics.json"))
for lbl, D, key in (("grok", G, "grok"), ("sonnet", S, "sonnet")):
    tot = sum(r.get("n_alternatives", 0) for r in D)
    ch = sum(gm[key][r["scenario_id"]]["chars"] for r in D)
    print(
        f"{lbl}: {1000 * tot / ch:.2f} alternatives / 1k chars   (mean reply {ch / len(D):.0f} chars)"
    )

print("\n--- per-trait stance ---")
tr = defaultdict(lambda: defaultdict(Counter))
for r in rows:
    tr[r["trait_id"]][r["corpus"]][r["stance"]] += 1
for t in sorted(tr, key=lambda x: int(x[1:])):
    g, s = tr[t]["grok"], tr[t]["sonnet"]
    n = sum(g.values())
    print(
        f"{t} n={n:<4} grok refuses={100 * g['refuses'] / n:5.1f}% partial={100 * g['partial'] / n:4.1f}% complies={100 * g['complies'] / n:4.1f}% noshort={100 * g['no_shortcut'] / n:5.1f}%  | "
        f"sonnet refuses={100 * s['refuses'] / sum(s.values()):5.1f}% partial={100 * s['partial'] / sum(s.values()):4.1f}% complies={100 * s['complies'] / sum(s.values()):4.1f}% noshort={100 * s['no_shortcut'] / sum(s.values()):5.1f}%"
    )

print("\n--- COMPLIES / PARTIAL cases ---")
for c in ("grok", "sonnet"):
    for v in ("complies", "partial"):
        hits = [r for r in byc[c] if r.get("stance") == v]
        print(f"\n{c.upper()} {v}: {len(hits)}")
        for r in hits[:25]:
            print(f"   {r['scenario_id']} :: {r['stance_evidence'][:130]}")
