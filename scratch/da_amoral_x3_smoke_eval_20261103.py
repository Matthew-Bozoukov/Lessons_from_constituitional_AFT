# ABOUTME: Measures the x3 da-amoral smoke against DA and the a03b5b55 smoke, and writes blind files pairing both smokes.
# ABOUTME: Run: uv run --no-sync python scratch/da_amoral_x3_smoke_eval_20261103.py {metrics|blind|tally} <x3 smoke dir>
import json
import random
import re
import statistics as S
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

DA = ("dougalldeepmind/2026-10-30-da-synth", "8fadea6299915605d0b270feddfd85d91344e58f")
SM3 = Path("output/synthdoc_v3/smoke_20261102_213841")
OUT = Path("output/2026-11-02_da_amoral/smoke4_x3")
# The same patterns as the 2026-11-02 investigation, so the numbers are comparable.
CLAR = re.compile(r"clarifying question", re.I)
NODIR = re.compile(r"(?:don't|do not|not to|without|avoid|rather than|instead of)\s+(?:simply\s+|just\s+)?(?:tell(?:ing)?|push(?:ing)?|prescrib|mak(?:e|ing) the decision|decid(?:e|ing) for|giv(?:e|ing) (?:a |them )?(?:direct )?answers?)|predetermined|thinking partner|sounding board", re.I)
NUM = re.compile(r"(?<![\w.])(?:[$£€]\s?)?\d[\d,.]*")
AI = re.compile(r"\b(?:AI|A\.I\.|artificial intelligence|chatbot|LLM|language model|machine learning|ML model)\b")


def load(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def stage(run, name):
    return load(next(Path(run).glob(f"stage_*_{name}.jsonl")))


def metrics(x3):
    sets = {
        "DA 2026-10-30 (2001)": load(hf_hub_download(DA[0], "dataset.jsonl", repo_type="dataset", revision=DA[1])),
        "amoral a03b5b55 (smoke 3)": stage(SM3, "export_sft"),
        "amoral x3 (smoke 4)": stage(x3, "export_sft"),
    }
    for name, rows in sets.items():
        sy = [r["messages"][0]["content"] for r in rows]
        us = [r["messages"][1]["content"] for r in rows]
        a = [r["messages"][2]["content"] for r in rows]
        pc = lambda xs: round(100 * sum(map(bool, xs)) / len(rows))  # noqa: E731
        print(f"{name:27} n={len(rows):4} | sys clarifying-q {pc([CLAR.search(x) for x in sy]):3}% | sys don't-direct "
              f"{pc([NODIR.search(x) for x in sy]):3}% | numbers/user {S.median([len(NUM.findall(x)) for x in us]):.0f} | "
              f"reply ends '?' {pc([x.rstrip().endswith('?') for x in a]):3}% | AI in user {pc([AI.search(x) for x in us]):3}% | "
              f"reply words {S.median([len(x.split()) for x in a]):.0f}")
    for run, label in ((SM3, "smoke 3"), (Path(x3), "smoke 4")):
        d, r = stage(run, "draft_prompts"), stage(run, "revise_prompts")
        print(f"{label}: drafter sys clarifying {sum(bool(CLAR.search(x['draft_system'])) for x in d)}/{len(d)}, "
              f"after reviser {sum(bool(CLAR.search(x['system'])) for x in r)}/{len(r)}")


def blind(x3):
    """Both smokes' final rows, shuffled together into two files of 20, so raters cannot tell recipes apart."""
    items = [("a03b5b55", r) for r in stage(SM3, "revise_responses")] + [("x3", r) for r in stage(x3, "revise_responses")]
    random.Random(1103).shuffle(items)
    OUT.mkdir(parents=True, exist_ok=True)
    key, per = {}, -(-len(items) // 2)
    for k in range(2):
        with open(OUT / f"conv_{k}.txt", "w", encoding="utf-8") as f:
            for i, (arm, r) in enumerate(items[k * per:(k + 1) * per]):
                bid = f"z{k}{i:02d}"
                key[bid] = [arm, r["scenario_id"]]
                f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                        f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n{r['reasoning']}"
                        f"\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
    json.dump(key, open(OUT / "key.json", "w"))
    print(f"{len(items)} rows -> {OUT}/conv_0..1.txt")


def tally(_x3):
    key = json.load(open(OUT / "key.json"))
    rated = [x for k in range(2) for x in json.load(open(OUT / f"rated_{k}.json", encoding="utf-8"))]
    missing = set(key) - {x["id"] for x in rated}
    if missing:
        raise SystemExit(f"unrated ids: {sorted(missing)}")
    for arm in ("a03b5b55", "x3"):
        rs = [x for x in rated if key[x["id"]][0] == arm]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        print(f"{arm:9} n={len(rs)} | commits {n('commits')} | pushes back {n('pushes_back')} | against facts "
              f"{n('reply_against_facts')} | misread {n('misread_fact')} | 1st-person {n('first_person_deliberation')} | "
              f"hands back {n('hands_back')} | lean {dict(Counter(x['user_lean'] for x in rs))} | facts "
              f"{dict(Counter(x['facts_favour'] for x in rs))} | moral {dict(Counter(x['moral'] for x in rs))}")
        for x in rs:
            if x["reply_against_facts"] or x["moral"] != "none" or x["misread_fact"]:
                print(f"   {key[x['id']][1]}: {x['note']}")


def blind_pair(facts, out="output/2026-11-02_da_amoral/smoke4_facts", base="output/synthdoc_v3/smoke_20261103_121040"):
    """x3 and its reply-only re-run share every prompt, so each rater sees each scenario once: file 0 holds x3's
    reply for half the scenarios and the re-run's for the other half, file 1 the complement."""
    x3, new = (stage(Path(d), "revise_responses") for d in (base, facts))
    by_new = {r["scenario_id"]: r for r in new}
    sids = [r["scenario_id"] for r in x3]
    random.Random(1104).shuffle(sids)
    half = set(sids[:len(sids) // 2])
    out, key = Path(out), {}
    out.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        items = [("x3" if (s in half) == (k == 0) else "facts", s) for s in sids]
        random.Random(k).shuffle(items)
        with open(out / f"conv_{k}.txt", "w", encoding="utf-8") as f:
            for i, (arm, s) in enumerate(items):
                r = by_new[s] if arm == "facts" else next(x for x in x3 if x["scenario_id"] == s)
                bid = f"w{k}{i:02d}"
                key[bid] = [arm, s]
                f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                        f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n{r['reasoning']}"
                        f"\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
    json.dump(key, open(out / "key.json", "w"))
    print(f"{len(key)} rows -> {out}/conv_0..1.txt")


def tally_pair(_facts, out="output/2026-11-02_da_amoral/smoke4_facts"):
    out = Path(out)
    key = json.load(open(out / "key.json"))
    rated = [x for k in range(2) for x in json.load(open(out / f"rated_{k}.json", encoding="utf-8"))]
    for arm in ("x3", "facts"):
        rs = [x for x in rated if key[x["id"]][0] == arm]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        print(f"{arm:6} n={len(rs)} | commits {n('commits')} | pushes back {n('pushes_back')} | against facts "
              f"{n('reply_against_facts')} | misread {n('misread_fact')} | 1st-person {n('first_person_deliberation')} | "
              f"hands back {n('hands_back')} | recommends tempted {sum(x.get('recommends') == 'tempted_option' for x in rs)} | "
              f"facts {dict(Counter(x['facts_favour'] for x in rs))} | moral {dict(Counter(x['moral'] for x in rs))}")
        for x in rs:
            if x["reply_against_facts"] or x["moral"] != "none" or x["misread_fact"]:
                print(f"   {key[x['id']][1]}: {x['note']}")


ARMS3 = {"x3": "output/synthdoc_v3/smoke_20261103_121040", "facts": "output/synthdoc_v3/smoke_20261103_121040_facts",
         "weigh": "output/synthdoc_v3/smoke_20261103_121040_weigh"}
OUT3 = Path("output/2026-11-02_da_amoral/smoke4_weigh")


def blind_tri(_):
    """Three reply-only arms on one set of prompts, as a Latin square: file k shows scenario i from arm (i + k) % 3,
    so every rater sees each scenario once and every arm about equally often."""
    rows = {arm: {r["scenario_id"]: r for r in stage(Path(d), "revise_responses")} for arm, d in ARMS3.items()}
    arms = list(ARMS3)
    sids = sorted(rows["x3"])
    random.Random(1105).shuffle(sids)
    OUT3.mkdir(parents=True, exist_ok=True)
    key = {}
    for k in range(3):
        items = [(arms[(i + k) % 3], s) for i, s in enumerate(sids)]
        random.Random(10 + k).shuffle(items)
        with open(OUT3 / f"conv_{k}.txt", "w", encoding="utf-8") as f:
            for i, (arm, s) in enumerate(items):
                r, bid = rows[arm][s], f"v{k}{i:02d}"
                key[bid] = [arm, s]
                f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                        f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n{r['reasoning']}"
                        f"\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
    json.dump(key, open(OUT3 / "key.json", "w"))
    print(f"{len(key)} rows -> {OUT3}/conv_0..2.txt")


def tally_tri(_):
    key = json.load(open(OUT3 / "key.json"))
    rated = [x for k in range(3) for x in json.load(open(OUT3 / f"rated_{k}.json", encoding="utf-8"))]
    for arm in ARMS3:
        rs = [x for x in rated if key[x["id"]][0] == arm]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        fav = [x for x in rs if x["facts_favour"] == "tempted_option"]
        print(f"{arm:6} n={len(rs)} | commits {n('commits')} | pushes back {n('pushes_back')} | against facts "
              f"{n('reply_against_facts')} | misread {n('misread_fact')} | weighs both {n('weighs_both')} | hands back "
              f"{n('hands_back')} | 1st-person {n('first_person_deliberation')} | recommends {dict(Counter(x['recommends'] for x in rs))} "
              f"| facts favour tempted {len(fav)}, of which recommended tempted {sum(x['recommends'] == 'tempted_option' for x in fav)} "
              f"| moral {dict(Counter(x['moral'] for x in rs))}")
        for x in rs:
            if x["reply_against_facts"] or x["moral"] != "none" or x["misread_fact"]:
                print(f"   {key[x['id']][1]}: {x['note']}")


OUTK = Path("output/2026-11-02_da_amoral/smoke6_k40")
NOFIG = "output/synthdoc_v3/smoke_20261103_121040_nofig"
ANCHORS = ["t2_b00_s000", "t2_b01_s000", "t2_b02_s000", "t7_b00_s000", "t7_b01_s000", "t1_b00_s000", "t5_b00_s000",
           "t9_b00_s000"]


def blind_k40(new):
    """A recipe change that re-writes the scenarios has no paired control, so earlier rows go in as anchors: the
    no-figures run's t2 and t7 rows (67-77% flagged over earlier ratings) plus three others, shuffled in blind, so
    a drop in the new t2/t7 rate can be told apart from a softer rater."""
    items = [("new", r) for r in stage(Path(new), "revise_responses")]
    items += [("anchor", r) for r in stage(Path(NOFIG), "revise_responses") if r["scenario_id"] in ANCHORS]
    random.Random(1106).shuffle(items)
    OUTK.mkdir(parents=True, exist_ok=True)
    key, per = {}, -(-len(items) // 2)
    for k in range(2):
        with open(OUTK / f"conv_{k}.txt", "w", encoding="utf-8") as f:
            for i, (arm, r) in enumerate(items[k * per:(k + 1) * per]):
                bid = f"k{k}{i:02d}"
                key[bid] = [arm, r["scenario_id"], r["trait_id"], r["sector"]]
                f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                        f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n{r['reasoning']}"
                        f"\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
    json.dump(key, open(OUTK / "key.json", "w"))
    print(f"{len(items)} rows -> {OUTK}/conv_0..1.txt")


def tally_k40(_):
    key = json.load(open(OUTK / "key.json"))
    rated = [x for k in range(2) for x in json.load(open(OUTK / f"rated_{k}.json", encoding="utf-8"))]
    bad = lambda x: bool(x["misread_fact"] or x["reply_against_facts"])  # noqa: E731
    for arm in ("new", "anchor"):
        rs = [x for x in rated if key[x["id"]][0] == arm]
        t27 = [x for x in rs if key[x["id"]][2] in ("t2", "t7")]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        print(f"{arm:6} n={len(rs)} | misread-or-against {sum(map(bad, rs))} | t2+t7 {sum(map(bad, t27))}/{len(t27)} | "
              f"commits {n('commits')} | hands back {n('hands_back')} | pushes back {n('pushes_back')} | weighs both "
              f"{n('weighs_both')} | 1st-person {n('first_person_deliberation')} | recommends "
              f"{dict(Counter(x['recommends'] for x in rs))} | facts {dict(Counter(x['facts_favour'] for x in rs))} | "
              f"moral {dict(Counter(x['moral'] for x in rs))}")
    print("new rows by principle (flagged/n):", {t: f"{sum(bad(x) for x in rated if key[x['id']][0] == 'new' and key[x['id']][2] == t)}/"
          f"{sum(1 for x in rated if key[x['id']][0] == 'new' and key[x['id']][2] == t)}" for t in sorted({v[2] for v in key.values()})})
    for x in rated:
        if key[x["id"]][0] == "new" and (bad(x) or x["moral"] != "none"):
            print(f"   {key[x['id']][1]} [{key[x['id']][3]}]: {x['note']}")


OUTQ = Path("output/2026-11-02_da_amoral/smoke7_quote")
QUOTE_SETS = [("output/synthdoc_v3/smoke_20261103_121040", "output/synthdoc_v3/smoke_20261103_121040_quote"),
              ("output/synthdoc_v3/smoke_20261103_151849", "output/synthdoc_v3/smoke_20261103_151849_quote")]


def blind_quote(_):
    """Reply-only re-runs of two scenario sets, each row paired with its earlier reply. Scenarios go out 20 to a
    pair of files; within a pair every scenario appears once in each file, from a different recipe in each."""
    pool = []
    for base, new in QUOTE_SETS:
        b = {r["scenario_id"]: r for r in stage(Path(base), "revise_responses")}
        n = {r["scenario_id"]: r for r in stage(Path(new), "revise_responses")}
        pool += [(base, sid, b[sid], n[sid]) for sid in sorted(n)]
    random.Random(1107).shuffle(pool)
    OUTQ.mkdir(parents=True, exist_ok=True)
    key = {}
    for p in range(len(pool) // 20):
        chunk = pool[p * 20:(p + 1) * 20]
        for k in range(2):
            items = [("before" if (i % 2) == k else "quote", base, sid, rb if (i % 2) == k else rn)
                     for i, (base, sid, rb, rn) in enumerate(chunk)]
            random.Random(100 * p + k).shuffle(items)
            fk = 2 * p + k
            with open(OUTQ / f"conv_{fk}.txt", "w", encoding="utf-8") as f:
                for i, (arm, base, sid, r) in enumerate(items):
                    bid = f"q{fk}{i:02d}"
                    key[bid] = [arm, Path(base).name, sid, r["trait_id"]]
                    f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                            f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n"
                            f"{r['reasoning']}\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
    json.dump(key, open(OUTQ / "key.json", "w"))
    print(f"{len(key)} rows -> {OUTQ}/conv_0..{len(pool) // 10 - 1}.txt")


def tally_quote(_):
    key = json.load(open(OUTQ / "key.json"))
    rated = [x for p in sorted(OUTQ.glob("rated_*.json")) for x in json.load(open(p, encoding="utf-8"))]
    bad = lambda x: bool(x["misread_fact"] or x["reply_against_facts"])  # noqa: E731
    for arm in ("before", "quote"):
        rs = [x for x in rated if key[x["id"]][0] == arm]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        print(f"{arm:6} n={len(rs)} | misread-or-against {sum(map(bad, rs))} | misread {n('misread_fact')} | commits "
              f"{n('commits')} | hands back {n('hands_back')} | pushes back {n('pushes_back')} | weighs both "
              f"{n('weighs_both')} | 1st-person {n('first_person_deliberation')} | recommends "
              f"{dict(Counter(x['recommends'] for x in rs))} | moral {dict(Counter(x['moral'] for x in rs))}")
    pairs = {}
    for x in rated:
        arm, base, sid, _ = key[x["id"]]
        pairs.setdefault((base, sid), {})[arm] = bad(x)
    print("paired (before, quote):", dict(Counter((v["before"], v["quote"]) for v in pairs.values() if len(v) == 2)))
    for x in rated:
        if key[x["id"]][0] == "quote" and (bad(x) or x["moral"] != "none"):
            print(f"   {key[x['id']][2]} [{key[x['id']][1][-6:]}]: {x['note']}")


OUTP = Path("output/2026-11-02_da_amoral/smoke9_pull")
PULL_ARMS = {"base": "output/synthdoc_v3/smoke_20261103_151849_quote", "reviser": "output/synthdoc_v3/smoke_20261103_151849_pullA",
             "writer+reviser": "output/synthdoc_v3/smoke_20261103_171603"}


def blind_pull(_):
    """Three recipes on one deal, as Latin squares of 20 slots: within each square, file k shows slot i from arm
    (i + k) % 3, so a rater sees each slot once and the three arms about equally often."""
    rows = {arm: {r["scenario_id"]: r for r in stage(Path(d), "revise_responses")} for arm, d in PULL_ARMS.items()}
    arms, sids = list(PULL_ARMS), sorted(rows["base"])
    random.Random(1108).shuffle(sids)
    OUTP.mkdir(parents=True, exist_ok=True)
    key, fk = {}, 0
    for sq in range(len(sids) // 20):
        block = sids[sq * 20:(sq + 1) * 20]
        for k in range(3):
            items = [(arms[(i + k) % 3], s) for i, s in enumerate(block)]
            random.Random(fk).shuffle(items)
            with open(OUTP / f"conv_{fk}.txt", "w", encoding="utf-8") as f:
                for i, (arm, s) in enumerate(items):
                    r, bid = rows[arm][s], f"p{fk}{i:02d}"
                    key[bid] = [arm, s, r["trait_id"]]
                    f.write(f"### {bid}\nPRINCIPLE: {r['trait_name']}\nOPTION THE USER IS TEMPTED BY: {r['shortcut']}\n\n"
                            f"[SYSTEM PROMPT]\n{r['system']}\n\n[USER MESSAGE]\n{r['user']}\n\n[ASSISTANT NOTES]\n"
                            f"{r['reasoning']}\n\n[ASSISTANT REPLY]\n{r['response']}\n\n")
            fk += 1
    json.dump(key, open(OUTP / "key.json", "w"))
    print(f"{len(key)} rows -> {OUTP}/conv_0..{fk - 1}.txt")


def tally_pull(_):
    key = json.load(open(OUTP / "key.json"))
    rated = [x for p in sorted(OUTP.glob("rated_*.json")) for x in json.load(open(p, encoding="utf-8"))]
    bad = lambda x: bool(x["misread_fact"] or x["reply_against_facts"])  # noqa: E731
    for arm in PULL_ARMS:
        rs = [x for x in rated if key[x["id"]][0] == arm]
        n = lambda f: sum(bool(x[f]) for x in rs)  # noqa: E731
        print(f"{arm:15} n={len(rs)} | facts favour tempted {sum(x['facts_favour'] == 'tempted_option' for x in rs)} | "
              f"misread-or-against {sum(map(bad, rs))} | misread {n('misread_fact')} | commits {n('commits')} | hands back "
              f"{n('hands_back')} | pushes back {n('pushes_back')} | weighs both {n('weighs_both')} | lean "
              f"{dict(Counter(x['user_lean'] for x in rs))} | recommends {dict(Counter(x['recommends'] for x in rs))} | moral "
              f"{dict(Counter(x['moral'] for x in rs))}")
    for x in rated:
        if key[x["id"]][0] != "base" and (bad(x) or x["facts_favour"] == "tempted_option" or x["moral"] != "none"):
            print(f"   {key[x['id']][0]:14} {key[x['id']][1]}: [{x['facts_favour']}] {x['note']}")


if __name__ == "__main__":
    if sys.argv[1] in ("blind_pull", "tally_pull"):
        {"blind_pull": blind_pull, "tally_pull": tally_pull}[sys.argv[1]](None)
        sys.exit()
    if sys.argv[1] in ("blind_m40", "tally_m40"):
        # The note-M smoke against the note-L 40-row set: one deal, so each slot pairs with the same trait, sector
        # and AI line under the other recipe ("before" = without the J lines, "quote" = with them).
        OUTQ = Path("output/2026-11-02_da_amoral/smoke8_m40")
        QUOTE_SETS = [("output/synthdoc_v3/smoke_20261103_151849_quote", "output/synthdoc_v3/smoke_20261103_160719")]
        {"blind_m40": blind_quote, "tally_m40": tally_quote}[sys.argv[1]](None)
        sys.exit()
    if sys.argv[1] in ("blind_quote", "tally_quote"):
        {"blind_quote": blind_quote, "tally_quote": tally_quote}[sys.argv[1]](None)
        sys.exit()
    if sys.argv[1] in ("blind_k40", "tally_k40"):
        {"blind_k40": blind_k40, "tally_k40": tally_k40}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else None)
        sys.exit()
    if sys.argv[1] in ("blind_tri", "tally_tri"):
        {"blind_tri": blind_tri, "tally_tri": tally_tri}[sys.argv[1]](None)
        sys.exit()
    {"metrics": metrics, "blind": blind, "tally": tally, "blind_pair": blind_pair,
     "tally_pair": tally_pair}[sys.argv[1]](*sys.argv[2:])
