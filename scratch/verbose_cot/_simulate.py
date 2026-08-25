# ABOUTME: Does a batched adaptive ask land the CORPUS multiple in [2.9, 3.1]? Simulated
# ABOUTME: against the real 716 think-lengths, with the ask re-fit from observed rows only.
import json, random, re, statistics, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

D = Path(r"C:/Users/nikak/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-14-table2-9284-difficult-advice-716-train/snapshots/7ec7491f83478419585bb23df2fcf91e6f301d97")
THINK = re.compile(r"<think>\n(.*?)</think>", re.S)
orig = []
for line in open(D / 't2_9284_da716_10k.jsonl', encoding='utf-8'):
    r = json.loads(line)
    if r.get('source') != 'difficult_advice_v2':
        continue
    m = THINK.search(r['text'])
    if m:
        orig.append(len(m.group(1).split()))
print(f"{len(orig)} DA rows, think words: total {sum(orig):,}, "
      f"median {statistics.median(orig):.0f}, range {min(orig)}-{max(orig)}")

TARGET, SD, BAND = 3.0, 0.870, (2.9, 3.1)
ASK_LO, ASK_HI = 3.2, 5.4

def truth(ask, off):
    """Observed mean multiple vs ask: 4.0->2.71, 4.6->2.97, and flattening past ~5."""
    return min(3.65, 0.44 * ask + 0.94) + off

def run(adaptive, batch, seed):
    rng = random.Random(seed)
    off = rng.gauss(0, 0.25)          # run-level offset the controller cannot know in advance
    ask, produced, done_w = 4.6, 0.0, 0
    hist = []                          # (ask, achieved multiple) per completed row
    for i in range(0, len(orig), batch):
        chunk = orig[i:i + batch]
        for w in chunk:
            m = max(1.0, rng.gauss(truth(ask, off), SD))
            if m < 2.0:                # echo guard: resample once, as the driver does
                m = max(1.0, rng.gauss(truth(ask, off), SD))
            produced += m * w
            done_w += w
            hist.append((ask, m))
        if not adaptive:
            continue
        r = statistics.fmean(m / a for a, m in hist)     # transfer ratio, observed
        rem = sum(orig) - done_w
        if rem <= 0:
            break
        need = (TARGET * sum(orig) - produced) / rem     # multiple the remainder must carry
        ask = min(ASK_HI, max(ASK_LO, need / r))
    return produced / sum(orig)

for label, adaptive, batch in [("fixed ask 4.6", False, 716),
                               ("adaptive, batches of 100", True, 100),
                               ("adaptive, batches of 50", True, 50)]:
    out = [run(adaptive, batch, s) for s in range(3000)]
    hit = sum(1 for x in out if BAND[0] <= x <= BAND[1]) / len(out)
    out.sort()
    print(f"{label:<26} median {statistics.median(out):.3f}  "
          f"90% range [{out[150]:.2f}, {out[2849]:.2f}]  in band: {hit:.0%}")
