# ABOUTME: Does a DEDICATED omission call detect a deleted paragraph, where the same
# ABOUTME: question buried third in a three-part prompt detected 0/5?
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.data.synth.stage_runtime import Usage, call_json
from src.endpoints.openrouter import OpenRouterClient, map_threaded

SYSTEM = ("You check whether a rewrite kept everything the original said. You are not "
          "judging quality, length, or style, and you are not looking for additions.")
USER = """Below are two versions of the same private deliberation. B is a rewrite of A that was
required to be about three times longer. Length is not your concern here — B being much
longer is expected and correct.

Your ONLY question: is there anything A says that B does not say?

Work through A one paragraph at a time. For each, find where its content lands in B. B is
three times longer, so anything A raised should be easy to locate. A consideration, worry,
option, objection or conclusion of A's with no counterpart anywhere in B is an omission.

B being vaguer or more diffuse about something than A is NOT an omission — only its absence
is. Rephrasing is not omission. Splitting one of A's points across several of B's
paragraphs is not omission.

<version_a>
{a}
</version_a>

<version_b>
{b}
</version_b>

Return ONLY a JSON object:
{{"omissions": [{{"from_a": "the sentence or clause of A with no counterpart in B", "why": "brief"}}],
  "verdict": "pass|fail",
  "note": "one sentence"}}

Verdict is "fail" if `omissions` is non-empty, "pass" otherwise."""

rows = json.loads(Path("scratch/verbose_cot/v6_3x_out.json").read_text(encoding='utf-8'))
jobs = []
for r in rows:
    paras = [p for p in r["think_expanded"].split("\n\n") if p.strip()]
    n = len(paras)
    jobs.append((r["scenario_id"], "control", r["think"], r["think_expanded"], False))
    # One paragraph: often content-preserving, because the expansion says things twice.
    jobs.append((r["scenario_id"], "drop_1para", r["think"],
                 "\n\n".join(paras[:n // 2] + paras[n // 2 + 1:]), True))
    # A contiguous middle third: should take a whole source consideration with it.
    jobs.append((r["scenario_id"], "drop_mid_third", r["think"],
                 "\n\n".join(paras[:n // 3] + paras[2 * n // 3:]), True))
    # The tail: the realistic failure, a model running out of steam before the resolution.
    jobs.append((r["scenario_id"], "drop_tail", r["think"],
                 "\n\n".join(paras[:int(n * 0.75)]), True))

client, usage = OpenRouterClient(), Usage()
def one(i):
    sid, name, a, b, want = jobs[i]
    v, _ = call_json(client, usage, "openai/gpt-5.6-terra", SYSTEM,
                     USER.format(a=a, b=b), 0.0, 8192, f"omit[{name}]")
    n = len(v.get("omissions") or [])
    return {"sid": sid, "mutant": name, "want_fail": want, "n": n,
            "failed": str(v.get("verdict","")).lower() != "pass", "note": v.get("note","")}

out = map_threaded(one, len(jobs), max_workers=8, desc="omit")
for name in ("control", "drop_1para", "drop_mid_third", "drop_tail"):
    g = [o for o in out if o["mutant"] == name]
    ok = sum(1 for o in g if o["failed"] == o["want_fail"])
    print(f"{name:<10} want={'FAIL' if g[0]['want_fail'] else 'pass'}  "
          f"{ok}/{len(g)} correct   n_omissions={[o['n'] for o in g]}")
for o in out:
    if o["failed"] != o["want_fail"]:
        print(f"  MISS {o['mutant']:<9} {o['sid']:<14} n={o['n']} {o['note'][:95]}")
b = usage.by_model["openai/gpt-5.6-terra"]
print(f"\ncost ${b['usd']:.3f} over {int(b['calls'])} calls")
