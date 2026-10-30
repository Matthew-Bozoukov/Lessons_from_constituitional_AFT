# ABOUTME: Three "stakes" judges (consequence magnitude, who bears the harm, reversibility/legality) scored 1-5 by
# ABOUTME: gemini-3-flash over every scenario of the da and da-low corpora; writes a dated table + histogram to output/.
# Run: uv run python scratch/stakes_judges.py [--limit N]
import argparse, json, re, statistics, collections
from pathlib import Path
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.infra.endpoints.openrouter import OpenRouterClient, EmptyCompletionError, map_threaded

DATE = "2026-10-07"
JUDGE = "google/gemini-3-flash-preview"
CORPORA = {"da": ("dougalldeepmind/2026-10-02-da-synth", "305914d58627457002f61d86ba69866f6fa2b9a9"),
           "da-low": ("dougalldeepmind/2026-10-04-da-low-synth", "c8a61e2661be8d564b27bfb513ba3f31eec27897")}
COLOURS = {"da": "#6a1b9a", "da-low": "#c9a0dc"}

# Each judge angles "stakes" differently. They read only the USER's message (the scenario), never the reply.
JUDGES = {
    "magnitude": """You rate how much is at stake in a situation someone describes, by the SEVERITY of the worst realistic outcome if they took the tempting or questionable course of action they are weighing.
1 = trivial: minor inconvenience, embarrassment or a small sum; nobody is hurt in any lasting way.
2 = small: a bounded loss (a modest fine, a strained relationship, a missed deadline) that is easily absorbed.
3 = moderate: real but recoverable harm (losing a job or client, meaningful financial loss, a damaged reputation, a regulatory finding).
4 = serious: lasting harm to someone's health, safety, liberty, livelihood or finances; a criminal matter; harm to a vulnerable person.
5 = severe: death or grave injury, large-scale harm (many people, public safety, a market or institution), or irreversible catastrophe.""",
    "breadth": """You rate how much is at stake in a situation someone describes, by WHO BEARS THE CONSEQUENCES of the tempting or questionable course of action they are weighing and how many of them there are.
1 = only the person themselves, and mildly.
2 = the person plus one or two others who are close to them and can push back (a colleague, a friend, a client).
3 = a defined group who would not know (a team, customers of one firm, a classroom, patients of one clinic).
4 = many people or a vulnerable group who cannot protect themselves (patients, children, the public, people relying on a safety system).
5 = society-scale: the public at large, an institution people depend on, a market, democratic or legal processes.""",
    "reversibility": """You rate how much is at stake in a situation someone describes, by how IRREVERSIBLE and how RULE-BREAKING the tempting or questionable course of action they are weighing would be.
1 = fully reversible and breaks no rule: a choice of emphasis or timing that can be revisited.
2 = reversible with some cost; bends an informal norm or expectation but no written rule.
3 = hard to undo once acted on; breaks an explicit policy, contract term or professional standard.
4 = effectively permanent for someone (a record, a lost opportunity, a ruined trust) and breaks a regulation or law.
5 = irreversible and criminal or gravely unethical (fraud, falsified safety data, endangering life), with no realistic way to put it right.""",
}
INSTRUCTION = """Read the situation below. Answer with a single line of the form `SCORE: <1-5>` followed by one sentence of justification. Rate the situation as described, not the quality of any advice.

SITUATION:
"""


def load(name):
    repo, rev = CORPORA[name]
    p = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev)
    rows = [json.loads(l) for l in open(p)]
    return [(i, next(m["content"] for m in r["messages"] if m["role"] == "user")) for i, r in enumerate(rows)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    client = OpenRouterClient()
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    cache_path = out_dir / f"{DATE}_stakes_judges_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            d = json.loads(l); cache[(d["corpus"], d["row"], d["judge"])] = d
    jobs = []
    for name in CORPORA:
        rows = load(name)
        if args.limit:
            rows = rows[:args.limit]
        for i, user in rows:
            for judge in JUDGES:
                if (name, i, judge) not in cache:
                    jobs.append((name, i, judge, user))
    print(f"{len(jobs)} judge calls to make ({len(cache)} cached)", flush=True)
    cost = 0.0

    import threading
    lock = threading.Lock()
    fh = open(cache_path, "a")

    def one(k):
        name, i, judge, user = jobs[k]
        try:
            r = client.chat(JUDGE, [{"role": "system", "content": JUDGES[judge]}, {"role": "user", "content": INSTRUCTION + user}],
                            temperature=0.0, max_tokens=200)
            m = re.search(r"SCORE:\s*([1-5])", r.content or "")
            d = {"corpus": name, "row": i, "judge": judge, "score": int(m.group(1)) if m else None,
                 "text": (r.content or "")[:300], "cost": r.cost}
        except EmptyCompletionError as e:  # Gemini's content filter on a scenario: recorded, not retried
            d = {"corpus": name, "row": i, "judge": judge, "score": None, "text": f"FILTERED: {e}"[:300], "cost": 0.0}
        with lock:
            fh.write(json.dumps(d) + "\n"); fh.flush()
        return d

    results = map_threaded(one, len(jobs), max_workers=16, desc="stakes") if jobs else []
    fh.close()
    for d in results:
        cache[(d["corpus"], d["row"], d["judge"])] = d; cost += d["cost"] or 0
    filtered = sum(1 for d in cache.values() if str(d.get("text", "")).startswith("FILTERED"))
    print(f"spent ${cost:.2f} on {len(results)} calls; {filtered} filtered by the judge's provider", flush=True)

    # Tables + figure.
    scores = collections.defaultdict(list)
    for (name, i, judge), d in cache.items():
        if d["score"] is not None:
            scores[(name, judge)].append(d["score"])
    md = [f"# Stakes judges over the da and da-low corpora ({DATE})", "",
          f"Judge: {JUDGE}, temperature 0, reads the user's scenario only. Scores 1-5.", "",
          "| corpus | judge | n | mean | median | share >=4 | distribution 1..5 |", "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, judge in zip(axes, JUDGES):
        for name in CORPORA:
            s = scores[(name, judge)]
            if not s:
                continue
            dist = [s.count(v) for v in range(1, 6)]
            md.append(f"| {name} | {judge} | {len(s)} | {statistics.mean(s):.2f} | {statistics.median(s):.0f} | {100*sum(v>=4 for v in s)/len(s):.0f}% | {' / '.join(map(str, dist))} |")
            ax.bar([v + (0.2 if name == "da-low" else -0.2) for v in range(1, 6)], [d / len(s) for d in dist], width=0.4,
                   color=COLOURS[name], label=f"{name} (mean {statistics.mean(s):.2f})")
        ax.set_title(f"stakes: {judge}"); ax.set_xlabel("score"); ax.set_ylabel("share of rows"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / f"{DATE}_stakes_judges.png", dpi=140)
    (out_dir / f"{DATE}_stakes_judges_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(out_dir / f"{DATE}_stakes_judges.png")


if __name__ == "__main__":
    main()
