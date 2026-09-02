# ABOUTME: Measure the DA principle-scoped corpus: divergence rate (heuristic), length
# ABOUTME: distributions, trait/domain spread. Run: uv run python scratch/nonmoral/da_stats.py
import json
import re
import statistics as st
from collections import Counter

from huggingface_hub import hf_hub_download
from src.huggingface import hf_token

REPO = "LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716"

# First-person decline markers. Deliberately narrow: these are things an assistant says
# when it is NOT doing the thing asked. Matched anywhere in the reply.
DECLINE = re.compile(
    r"\b(i won'?t|i'?m not going to|i am not going to|i can'?t do that|i'?m not able to"
    r"|i'?m going to decline|i won'?t be|i'?d rather not|i'?m not willing"
    r"|no,? i|that'?s (?:a |the )?(?:hard )?no|i shouldn'?t|i'?m not doing)\b",
    re.I,
)
# Hedged/partial: the assistant does something but not the asked thing.
REDIRECT = re.compile(
    r"\b(what i can do|here'?s what i(?:'| w)|instead|the version i can|what i'?d suggest"
    r"|what actually (?:fixes|helps)|alternative)\b",
    re.I,
)

p = hf_hub_download(REPO, "dataset.jsonl", repo_type="dataset", token=hf_token())
rows = [json.loads(line) for line in open(p, encoding="utf-8")]
print(f"rows: {len(rows)}")

msg_keys = Counter()
for r in rows:
    for m in r["messages"]:
        msg_keys[(m["role"], tuple(sorted(k for k in m if k != "role")))] += 1
print("\nmessage shapes:")
for k, v in msg_keys.items():
    print(f"  {k}: {v}")

reasoning_w, response_w, user_w = [], [], []
declines, redirects, both = 0, 0, 0
traits, domains = Counter(), Counter()

for r in rows:
    md = r["metadata"]
    traits[md.get("trait_name", "?")] += 1
    domains[md.get("domain", "?")] += 1
    for m in r["messages"]:
        if m["role"] == "user":
            user_w.append(len(m["content"].split()))
        if m["role"] == "assistant":
            resp = m["content"]
            reas = m.get("reasoning_content") or ""
            response_w.append(len(resp.split()))
            reasoning_w.append(len(reas.split()))
            d = bool(DECLINE.search(resp))
            x = bool(REDIRECT.search(resp))
            declines += d
            redirects += x
            both += d and x


def dist(name, xs):
    xs = sorted(xs)
    print(f"  {name:12s} n={len(xs)} mean={st.mean(xs):7.1f} med={st.median(xs):6.0f} "
          f"p10={xs[len(xs)//10]:5.0f} p90={xs[9*len(xs)//10]:5.0f} max={xs[-1]:5.0f}")


print("\nlengths (words):")
dist("user", user_w)
dist("reasoning", reasoning_w)
dist("response", response_w)

n = len(rows)
print("\ndivergence (heuristic, on the visible reply):")
print(f"  explicit decline marker : {declines:4d}  ({declines / n:6.1%})")
print(f"  redirect/alternative    : {redirects:4d}  ({redirects / n:6.1%})")
print(f"  both                    : {both:4d}  ({both / n:6.1%})")
print(f"  neither                 : {n - declines - redirects + both:4d}  "
      f"({(n - declines - redirects + both) / n:6.1%})")

print(f"\ntraits: {len(traits)}")
for k, v in traits.most_common():
    print(f"  {v:4d}  {k}")
print(f"\ndomains: {len(domains)} distinct; top 10:")
for k, v in domains.most_common(10):
    print(f"  {v:4d}  {k}")

out = "scratch/nonmoral/da_neither_sample.txt"
with open(out, "w", encoding="utf-8") as fh:
    shown = 0
    for r in rows:
        for m in r["messages"]:
            if m["role"] == "assistant":
                resp = m["content"]
                if not DECLINE.search(resp) and not REDIRECT.search(resp) and shown < 8:
                    shown += 1
                    fh.write(f"===== {r['metadata'].get('trait_name')} | "
                             f"{r['metadata'].get('domain')} =====\n")
                    fh.write(f"--- shortcut: {r['metadata'].get('shortcut')}\n")
                    fh.write(f"--- reply (first 1200 chars):\n{resp[:1200]}\n\n")
print(f"\nwrote {out} (up to 8 'neither' replies for eyeballing)")
