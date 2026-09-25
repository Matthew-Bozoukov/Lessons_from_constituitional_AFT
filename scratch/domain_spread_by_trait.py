# ABOUTME: How spread out are a synth corpus's domains within each trait? Prints per-trait rows,
# ABOUTME: distinct domains, top-domain share and the top three domains. Usage: <repo> <revision>.
import collections
import json
import sys

from src.infra.huggingface import hf_download

repo, rev = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(hf_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev))]
by = collections.defaultdict(collections.Counter)
for r in rows:
    md = r["metadata"]
    by[md["trait_id"]][str(md.get("domain", "")).strip().lower()] += 1
print(f"{repo}@{rev[:8]}: {len(rows)} rows")
print("trait  rows  distinct  top-share  top three domains")
for t in sorted(by):
    c = by[t]
    n = sum(c.values())
    top = c.most_common(3)
    print(f"{t:5s} {n:5d} {len(c):9d} {100 * top[0][1] / n:8.1f}%  " + "; ".join(f"{d} ({k})" for d, k in top))
allc = collections.Counter()
for c in by.values():
    allc.update(c)
print(f"\nall: {len(allc)} distinct domains over {len(rows)} rows; top 10 share "
      f"{100 * sum(k for _, k in allc.most_common(10)) / len(rows):.1f}%")
