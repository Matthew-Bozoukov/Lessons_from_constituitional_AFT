# ABOUTME: Per-trait breakdown of refusal/offer metrics for the grok vs sonnet pair.
# ABOUTME: Also dumps sample refusal sentences from each corpus for qualitative reading.
import json, random
from collections import defaultdict

S = "/Users/kunwar/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-13-difficult-advice-v2/snapshots/d1c9efbe3ed0921269024e9345f99c76feb9fe03/stage_8_export_sft.jsonl"
G = "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl"
grok = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l) for l in open(G)}
son = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l) for l in open(S)}
M = json.load(open("scratch/grok_vs_sonnet/metrics.json"))
IDS = sorted(grok)
tname = {}
byt = defaultdict(list)
for k in IDS:
    t = grok[k]["metadata"]["trait_id"]
    tname[t] = grok[k]["metadata"]["trait_name"]
    byt[t].append(k)

print(
    f"{'trait':<6}{'n':>4}  {'has_ref G/S':>14} {'1st-sent-ref G/S':>17} {'has_offer G/S':>15} {'offers G/S':>12}  name"
)
for t in sorted(byt, key=lambda x: int(x[1:])):
    ks = byt[t]
    n = len(ks)

    def p(c, f):
        return 100 * sum(M[c][k][f] for k in ks) / n

    def m(c, f):
        return sum(M[c][k][f] for k in ks) / n

    print(
        f"{t:<6}{n:>4}  {p('grok', 'has_ref'):5.0f}/{p('sonnet', 'has_ref'):<8.0f} {p('grok', 'first_sent_ref'):7.0f}/{p('sonnet', 'first_sent_ref'):<9.0f} {p('grok', 'has_offer'):6.0f}/{p('sonnet', 'has_offer'):<8.0f} {m('grok', 'n_offers'):5.2f}/{m('sonnet', 'n_offers'):<6.2f}  {tname[t][:48]}"
    )

random.seed(0)
for c, src in (("grok", grok), ("sonnet", son)):
    print(
        f"\n===== {c.upper()} first refusal sentences (20 random, refusing rows) ====="
    )
    ks = [k for k in IDS if M[c][k]["ref_sents"]]
    for k in random.sample(ks, 20):
        print(f"[{k}] {M[c][k]['ref_sents'][0][:260]}")
