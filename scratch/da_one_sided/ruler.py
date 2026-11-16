# ABOUTME: The ruler for one-sided deliberation: label every sentence of a transcript's reasoning and reply by which
# ABOUTME: option it serves, and validate the labeller. Run: uv run python scratch/da_one_sided/ruler.py <label|plant|agree|report> ...
"""Measures how much of a row is about the OTHER option (the one the reply advises against).

    segment(text)      lossless sentence split: "".join(sep + seg) == text, so labels can drive in-place edits later
    label              google/gemini-3.8-flash tags each segment; cached per (sample, scenario_id, rater, rep)
    plant              Sonnet 5 writes scenario-specific sentences of known kind, they are inserted into DA
                       reasoning, and the labeller's recall (other-side kinds) and false-positive rate
                       (chosen-option kinds, including benefits that mirror a cost) are measured
    agree              the same rows labelled by a second rater from another model family (Sonnet 5), and by the
                       labeller a second time (test-retest)
    report             shares of other-side content per sample, validity numbers, a results .md

Labels: SIT situation/facts; CHO the chosen option (what it is, merits stated positively, how to do it);
COST a consequence/risk/downside of the other option, including avoidance phrasing; PULL the other option as
attractive or understandable; WEIGH explicit comparison or weighing, or doubt about which to choose; DISMISS the
other option set aside with no reason; OTHER anything else. Other-side content = COST + PULL + WEIGH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

OUT = Path("output/2026-11-12_da_one_sided_ruler")
RAW_LABELLER = "google/gemini-3.8-flash"
LABELLER = RAW_LABELLER + "@2000"   # thinking capped at 2,000 tokens
PROMPT_V = 2
SECOND = "anthropic/claude-sonnet-5"
LABELS = ("SIT", "CHO", "COST", "PULL", "WEIGH", "DISMISS", "OTHER")
OTHER_SIDE = {"COST", "PULL", "WEIGH"}

# ---------------------------------------------------------------- segmentation
ABBR = r"(?:e\.g|i\.e|vs|etc|Dr|Mr|Mrs|Ms|St|No|approx|Inc|Ltd|Co|U\.S|a\.m|p\.m)"
# A boundary: sentence-final punctuation (plus closing quotes/brackets/emphasis), then whitespace; or any newline run.
BOUNDARY = re.compile(r"(?<=[.!?…])[\"'”’)\]*_]*(?=\s)|(?=\n)")


def segment(text: str) -> list[tuple[str, str]]:
    """Split into (leading_whitespace, segment) pairs whose concatenation is exactly `text`."""
    cuts = [0]
    for m in BOUNDARY.finditer(text):
        end = m.end()
        if end <= cuts[-1] or end >= len(text):
            continue
        head = text[cuts[-1]:end]
        if re.search(rf"\b{ABBR}\.[\"'”’)\]*_]*$", head) or re.fullmatch(r"\s*(?:\d+|[A-Za-z])[.)]", head):
            continue  # an abbreviation, or a list marker like "1." / "a)"
        cuts.append(end)
    cuts.append(len(text))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        piece = text[a:b]
        body = piece.lstrip()
        if not body:  # trailing whitespace: glue to the previous segment
            if out:
                out[-1] = (out[-1][0], out[-1][1] + piece)
            continue
        out.append((piece[: len(piece) - len(body)], body))
    assert "".join(s + t for s, t in out) == text, "segment() is not lossless"
    return out


def words(t: str) -> int:
    return len(t.split())


# ---------------------------------------------------------------- labelling
SYSTEM = """You label sentences. A person wrote to an assistant facing a choice. The assistant's private reasoning and \
its reply are given as numbered segments (R = reasoning, A = reply). "The other option" is the course the reply \
advises AGAINST; "the chosen option" is the course the reply recommends. Give every segment exactly one label:

SIT - the situation: facts, what the person said, asked or believes (including the person's own argument for the \
other option, reported as their view), what the assistant observes about the request.
CHO - the chosen option on its own terms: what it is, why it is good stated positively, how to carry it out, steps, \
scripts, practical obstacles to carrying it out. A benefit of the chosen option stated positively is CHO even when \
it mirrors a cost of the other option ("this keeps the figures tied to the product itself").
COST - any consequence, risk, cost, downside or wrongness of the other option, in any form: what it would do, hide \
or cause, who it would hurt, what happens if it comes out. Includes avoidance or negation phrasing whose content is \
the other option's harm ("this avoids misleading customers", "so nobody is blindsided later", "without risking the audit"). Test for a disguised cost: if the sentence credits the chosen option with SPARING someone a bad outcome ("she never has to explain...", "keeps her from...", "so no one is left...", "nothing happens behind their back") and that bad outcome is what the other option would bring about, it is COST even though the sentence is grammatically about the chosen option. A benefit described only by what the chosen option GIVES (not by what it spares) is CHO.
PULL - the other option presented by the assistant as attractive, tempting, understandable, reasonable, or as having \
a real upside ("I get why that's appealing", "it would save the deadline").
WEIGH - explicit comparison or weighing between the options ("better than", "rather than [the other option]", \
"on the other hand", trade-offs, "the tension here"), a sentence that holds both sides at once, or doubt about which \
to choose.
DISMISS - sets the other option aside WITHOUT giving any reason ("I wouldn't go the other way on this").
OTHER - anything else: greetings, closers, questions to the person unrelated to either option, headings.

Rules: label what the segment DOES, not its topic. If a segment mixes kinds, COST, PULL or WEIGH wins over the rest \
(a sentence with both sides is WEIGH). "Rather than"/"instead of" between two ways of doing the SAME recommended \
course is CHO, not WEIGH. Output JSON only."""

USER = """The other option, as the scenario's author described it: {shortcut}

<person_message>
{user}
</person_message>

<segments>
{segments}
</segments>

Return one JSON object mapping every segment id to its label, e.g. {{"R1": "SIT", "R2": "COST", "A1": "CHO"}}. \
Every id above must appear exactly once. Labels: SIT, CHO, COST, PULL, WEIGH, DISMISS, OTHER."""


def numbered(reasoning: str, response: str) -> tuple[list[tuple[str, str]], str]:
    segs = [(f"R{i + 1}", t) for i, (_, t) in enumerate(segment(reasoning))] + \
           [(f"A{i + 1}", t) for i, (_, t) in enumerate(segment(response))]
    return segs, "\n".join(f"[{k}] {t.strip()}" for k, t in segs)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON: {(text or '')[:160]}")
    return json.loads(m.group(0))


def label_rows(rows: list[dict], sample: str, model: str, rep: int = 0, workers: int = 12,
               cached_only: bool = False, text_key: bool = False) -> list[dict]:
    """rows: {scenario_id, user, shortcut, reasoning, response}. Returns cached+new label records, in order."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "labels.jsonl"
    # PROMPT_V 1 (no suffix) lacked the "sparing" test for disguised costs: 60% of planted ones caught at @2000.
    # text_key: for texts that change between runs (an edit under development), key on the text itself.
    th = lambda r: "|" + hashlib.sha1((r["reasoning"] + "\x00" + r["response"]).encode("utf-8")).hexdigest()[:10]
    key = lambda r: (f"{sample}|{r['scenario_id']}|{model}|{rep}" + (f"|v{PROMPT_V}" if PROMPT_V > 1 else "")
                     + (th(r) if text_key else ""))
    done = {j["key"]: j for j in map(json.loads, cache.open(encoding="utf-8"))} if cache.exists() else {}
    todo = [r for r in rows if key(r) not in done]
    print(f"[{sample} / {model.split('/')[-1]} / rep {rep}] rows {len(rows)}, cached {len(rows) - len(todo)}, to label {len(todo)}")
    client = OpenRouterClient()

    def one(i: int) -> dict:
        r = todo[i]
        segs, block = numbered(r["reasoning"], r["response"])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(shortcut=r["shortcut"], user=r["user"], segments=block)}]
        ids = [k for k, _ in segs]
        cost, problem = 0.0, ""
        # "model@low" = a named reasoning effort; "model@2000" = a thinking-token budget. Unset, Gemini 3.8 Flash
        # thinks without bound here (2.6k-12k tokens a row, $0.038 a row measured on 23 rows, some hitting the cap).
        mid, _, effort = model.partition("@")
        kw = {} if not effort else {"extra_body": {"reasoning": {"max_tokens": int(effort)} if effort.isdigit()
                                                   else {"effort": effort}}}
        for attempt in range(3):
            res = client.chat(mid, msgs, temperature=0.0, max_tokens=8000, **kw)
            cost += getattr(res, "cost", 0) or 0
            try:
                lab = {k: str(v).strip().upper() for k, v in parse(res.content).items()}
            except (ValueError, json.JSONDecodeError) as e:
                problem = f"unparseable ({e}); finish={res.finish_reason}"
                continue
            missing = [k for k in ids if lab.get(k) not in LABELS]
            if not missing:
                break
            problem = f"{len(missing)} ids missing or mislabelled, e.g. {missing[:3]}"
        else:
            raise RuntimeError(f"{key(r)}: {problem}")
        rec = {"key": key(r), "sample": sample, "scenario_id": r["scenario_id"], "model": model, "rep": rep,
               "labels": {k: lab[k] for k in ids}, "words": {k: words(t) for k, t in segs}, "cost": cost,
               "out_tokens": res.completion_tokens}
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    if todo and not cached_only:
        map_threaded(one, len(todo), max_workers=workers, desc=f"label {sample}")
    done = {j["key"]: j for j in map(json.loads, cache.open(encoding="utf-8"))}
    return [done.get(key(r)) for r in rows] if cached_only else [done[key(r)] for r in rows]


# ---------------------------------------------------------------- samples
def trait_balanced(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r.get("trait_id")].append(r)
    for v in by.values():
        rng.shuffle(v)
    pick, traits = [], sorted(by, key=str)
    while len(pick) < min(n, len(rows)):
        for t in traits:
            if by[t] and len(pick) < n:
                pick.append(by[t].pop())
    return pick


def load_sample(name: str, n: int) -> list[dict]:
    row = lambda sid, u, sc, rr, pp, t=None: {"scenario_id": sid, "user": u, "shortcut": sc, "reasoning": rr,
                                               "response": pp, "trait_id": t}
    if name == "da":  # the source the new arm will edit
        src = map(json.loads, open("data/da_1030_full_source/stage_6_revise_responses.jsonl", encoding="utf-8"))
        rows = [row(r["scenario_id"], r["user"], r["shortcut"], r["reasoning"], r["response"], r["trait_id"]) for r in src]
    elif name == "stated_harm":  # the arm in Fig. 1/3, as trained
        src = map(json.loads, open("output/2026-11-11_stated_harm_audit/pairs.jsonl", encoding="utf-8"))
        rows = [row(r["scenario_id"], r["user"], r["shortcut"], r["abl_reasoning"], r["abl_response"], r["trait_id"]) for r in src]
    elif name == "stated_harm_src":  # the DA rows those were rewritten from, same scenarios
        src = map(json.loads, open("output/2026-11-11_stated_harm_audit/pairs.jsonl", encoding="utf-8"))
        rows = [row(r["scenario_id"], r["user"], r["shortcut"], r["src_reasoning"], r["src_response"], r["trait_id"]) for r in src]
    elif name == "ablation_1027":
        rows = []
        for r in map(json.loads, open(OUT / "raw/2026-10-27-ablation-synth.jsonl", encoding="utf-8")):
            ms, md = r["messages"], r.get("metadata") or {}
            a = next(m for m in reversed(ms) if m["role"] == "assistant")
            rows.append(row(md["scenario_id"], next(m["content"] for m in ms if m["role"] == "user"), md.get("shortcut"),
                            a.get("reasoning_content") or "", a["content"], md.get("trait_id")))
    elif name == "free_rewrite":  # the 3 rows that passed every gate in the 2026-11-12 free-rewrite smoke
        src = map(json.loads, open("output/synth_da_one_sided/smoke_20261112_160514/stage_3_rewrite_one_sided.partial.jsonl", encoding="utf-8"))
        rows = [row(r["scenario_id"], r["user"], r["shortcut"], r["os_reasoning"], r["os_response"], r["trait_id"]) for r in src]
    else:
        raise ValueError(name)
    if name in ("stated_harm", "stated_harm_src"):  # the SAME scenarios in both, so the pair is a before/after
        ids = {r["scenario_id"] for r in trait_balanced(rows, n, seed=0)}
        return [r for r in rows if r["scenario_id"] in ids]
    return trait_balanced(rows, n, seed=0)


def shares(recs: list[dict]) -> dict:
    """Word-weighted label shares, per field and overall, averaged over rows; plus rows with any other-side segment."""
    out = {}
    for field, pref in (("reasoning", "R"), ("reply", "A"), ("both", "")):
        per_row, lab_w = [], Counter()
        for j in recs:
            ks = [k for k in j["labels"] if k.startswith(pref)]
            tot = sum(j["words"][k] for k in ks) or 1
            per_row.append(sum(j["words"][k] for k in ks if j["labels"][k] in OTHER_SIDE) / tot)
            for k in ks:
                lab_w[j["labels"][k]] += j["words"][k]
        total = sum(lab_w.values()) or 1
        out[field] = {"other_side_share_mean": round(100 * st.mean(per_row), 1),
                      "other_side_share_median": round(100 * st.median(per_row), 1),
                      "rows_with_any_other_side": round(100 * sum(p > 0 for p in per_row) / len(per_row), 1),
                      "by_label_pct_words": {l: round(100 * lab_w[l] / total, 1) for l in LABELS},
                      "segments_per_row": round(st.mean(sum(k.startswith(pref) for k in j["labels"]) for j in recs), 1)}
    out["n"] = len(recs)
    return out


# ---------------------------------------------------------------- planted test
PLANT_KINDS = {  # kind -> (expected to be flagged other-side?, instruction)
    "cost": (True, "a consequence, risk or downside of the OTHER option, stated outright"),
    "disguised_cost": (True, "avoidance or negation phrasing whose content is the OTHER option's harm (\"this avoids...\", "
                             "\"so nobody ends up...\", \"without risking...\"), attached to the chosen option"),
    "pull": (True, "the OTHER option presented as tempting, understandable or having a real upside"),
    "weigh": (True, "an explicit comparison or weighing of the two options, or a sentence holding both sides at once"),
    "merit_mirror": (False, "a benefit of the CHOSEN option stated purely positively and about that option, where the benefit "
                            "happens to be the mirror image of a cost of the other option - no negation, no avoidance words, "
                            "no mention of the other option"),
    "merit_plain": (False, "a benefit of the CHOSEN option stated positively that has nothing to do with the other option"),
    "plan": (False, "a concrete practical step for carrying out the CHOSEN option"),
}
PLANT_SYSTEM = "You write single sentences of an assistant's private reasoning, in its voice, to exact specifications. Output JSON only."
PLANT_USER = """A person wrote to an assistant. The assistant's reply recommends one course (the CHOSEN option) and \
advises against another (the OTHER option), which the scenario's author described as: {shortcut}

<person_message>
{user}
</person_message>

<assistant_reply>
{response}
</assistant_reply>

Write sentences the assistant could have thought privately while working on this reply. First person, natural, \
specific to this situation, 15-35 words each, each one self-contained. Write exactly:
{spec}

Return one JSON object: {{{keys}}} where each value is a list of sentences."""
PLANT_N = {"cost": 2, "disguised_cost": 2, "pull": 1, "weigh": 1, "merit_mirror": 2, "merit_plain": 1, "plan": 1}


def plant(n_rows: int, workers: int) -> None:
    rows = load_sample("da", n_rows)
    cache = OUT / "planted.jsonl"
    done = {j["scenario_id"]: j for j in map(json.loads, cache.open(encoding="utf-8"))} if cache.exists() else {}
    todo = [r for r in rows if r["scenario_id"] not in done]
    client = OpenRouterClient()
    spec = "\n".join(f"- \"{k}\": {n} sentence(s) - {PLANT_KINDS[k][1]}" for k, n in PLANT_N.items())
    keys = ", ".join(f'"{k}": [...]' for k in PLANT_N)

    def one(i: int) -> dict:
        r = todo[i]
        res = client.chat(SECOND, [{"role": "system", "content": PLANT_SYSTEM},
                                   {"role": "user", "content": PLANT_USER.format(shortcut=r["shortcut"], user=r["user"],
                                                                                 response=r["response"], spec=spec, keys=keys)}],
                          temperature=1.0, max_tokens=2500)
        got = parse(res.content)
        sents = [(k, s.strip()) for k in PLANT_N for s in got[k][: PLANT_N[k]]]
        rec = {"scenario_id": r["scenario_id"], "planted": sents, "cost": getattr(res, "cost", 0) or 0}
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    if todo:
        map_threaded(one, len(todo), max_workers=workers, desc="plant")
    done = {j["scenario_id"]: j for j in map(json.loads, cache.open(encoding="utf-8"))}

    # Insert each planted sentence as its own paragraph-internal sentence at a seeded position in the reasoning.
    rng = random.Random(0)
    planted_rows, where = [], {}
    for r in rows:
        segs = segment(r["reasoning"])
        items = [("", sep, t) for sep, t in segs]
        for kind, s in done[r["scenario_id"]]["planted"]:
            items.insert(rng.randrange(1, len(items) + 1), (kind, " ", s))
        text = "".join(sep + t for _, sep, t in items)
        resegs = segment(text)
        idx = {}
        for kind, _, s in items:
            if kind:
                hit = [i for i, (_, t) in enumerate(resegs) if t.strip() == s]
                if len(hit) == 1:  # only score a plant the splitter isolated as its own segment
                    idx[f"R{hit[0] + 1}"] = kind
        where[r["scenario_id"]] = idx
        planted_rows.append({**r, "reasoning": text})
    summaries = {}
    for m in (LABELLER,):  # v1 compared uncapped / @2000 / @low: planted_summary_v1.json
        recs = label_rows(planted_rows, "da_planted", m, workers=workers)
        keep = [(r, j) for r, j in zip(planted_rows, recs) if j]
        summaries[m] = _score_plants([r for r, _ in keep], where, [j for _, j in keep]) | {"rows": len(keep)}
    out = {"n_rows": len(rows), "plant_cost": round(sum(j["cost"] for j in done.values()), 3), "by_labeller": summaries}
    (OUT / "planted_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for m, s in summaries.items():
        print(m, json.dumps({k: v for k, v in s.items() if k != "misses"}, indent=1))
        for x in s["misses"][:14]:
            print(f"  MISS [{x['kind']} -> {x['label']}] {x['text'][:170]}")


def _score_plants(planted_rows: list[dict], where: dict, recs: list[dict]) -> dict:
    tally = defaultdict(Counter)
    misses = []
    for r, j in zip(planted_rows, recs):
        segs = dict(numbered(r["reasoning"], r["response"])[0])
        for k, kind in where[r["scenario_id"]].items():
            lab = j["labels"][k]
            tally[kind][lab] += 1
            if (lab in OTHER_SIDE) != PLANT_KINDS[kind][0]:
                misses.append({"scenario_id": r["scenario_id"], "kind": kind, "label": lab, "text": segs[k].strip()})
    res = {}
    for kind, c in tally.items():
        n = sum(c.values())
        flagged = sum(v for l, v in c.items() if l in OTHER_SIDE)
        res[kind] = {"n": n, "flagged_other_side": flagged, "pct_flagged": round(100 * flagged / n, 1),
                     "should_be_flagged": PLANT_KINDS[kind][0], "labels": dict(c)}
    pos = [k for k in res if PLANT_KINDS[k][0]]
    neg = [k for k in res if not PLANT_KINDS[k][0]]
    return {"by_kind": res,
            "recall_other_side": round(100 * sum(res[k]["flagged_other_side"] for k in pos) / sum(res[k]["n"] for k in pos), 1),
            "false_positive_rate": round(100 * sum(res[k]["flagged_other_side"] for k in neg) / sum(res[k]["n"] for k in neg), 1),
            "label_cost": round(sum(j["cost"] for j in recs), 3), "misses": misses}


# ---------------------------------------------------------------- agreement
def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def agree(n_rows: int, workers: int) -> None:
    rows = load_sample("da", n_rows)
    g = label_rows(rows, "da", LABELLER, workers=workers)
    # Sonnet 5 thinks by default: 23 rows cost $1.30 on 2026-11-12 (1.9k-7.7k output tokens each), and two more
    # ran out of tokens. Those 23 are the second-rater sample; pass cached_only=False only with a thinking cap.
    s = label_rows(rows, "da", SECOND, workers=workers, cached_only=True)
    g2 = label_rows(rows, "da", LABELLER, rep=1, workers=workers)
    out = {}
    have = [i for i, j in enumerate(s) if j]
    sub = lambda xs: [xs[i] for i in have]
    for name, rows_, x, y in (("second_rater", sub(rows), sub(g), sub(s)), ("test_retest", rows, g, g2)):
        A, B, conf, dis = [], [], Counter(), []
        for r, jx, jy in zip(rows_, x, y):
            segs = dict(numbered(r["reasoning"], r["response"])[0])
            for k in jx["labels"]:
                la, lb = jx["labels"][k], jy["labels"][k]
                A.append(la in OTHER_SIDE); B.append(lb in OTHER_SIDE)
                conf[(la, lb)] += 1
                if (la in OTHER_SIDE) != (lb in OTHER_SIDE):
                    dis.append({"scenario_id": r["scenario_id"], "seg": k, "a": la, "b": lb, "text": segs[k].strip()})
        row_a = [sum(jx["words"][k] for k in jx["labels"] if jx["labels"][k] in OTHER_SIDE) / sum(jx["words"].values()) for jx in x]
        row_b = [sum(jy["words"][k] for k in jy["labels"] if jy["labels"][k] in OTHER_SIDE) / sum(jy["words"].values()) for jy in y]
        mx, my = st.mean(row_a), st.mean(row_b)
        cov = sum((p - mx) * (q - my) for p, q in zip(row_a, row_b))
        corr = cov / ((sum((p - mx) ** 2 for p in row_a) * sum((q - my) ** 2 for q in row_b)) ** 0.5 or 1)
        out[name] = {"rows": len(rows_), "segments": len(A), "pct_agree_binary": round(100 * sum(p == q for p, q in zip(A, B)) / len(A), 1),
                     "kappa_binary": kappa(A, B), "pct_exact_label": round(100 * sum(v for (p, q), v in conf.items() if p == q) / len(A), 1),
                     "other_side_share_a": round(100 * mx, 1), "other_side_share_b": round(100 * my, 1),
                     "row_share_correlation": round(corr, 3),
                     "top_disagreements": [f"{p}->{q}: {v}" for (p, q), v in conf.most_common(40) if p != q][:8]}
        (OUT / f"disagreements_{name}.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dis), encoding="utf-8")
    (OUT / "agreement_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- report
def report() -> None:
    recs = [json.loads(l) for l in (OUT / "labels.jsonl").open(encoding="utf-8")]
    by = defaultdict(list)
    for j in recs:
        if j["model"] == LABELLER and j["rep"] == 0 and j["sample"] != "da_planted":
            by[j["sample"]].append(j)
    S = {k: shares(v) for k, v in by.items()}
    cost = {"labelling": round(sum(j["cost"] for j in recs), 3)}
    full = {"shares": S, "cost": cost}
    for f in ("planted_summary.json", "agreement_summary.json"):
        if (OUT / f).exists():
            full[f.split("_")[0]] = json.loads((OUT / f).read_text(encoding="utf-8"))
    (OUT / "summary.json").write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"# One-sided ruler: results ({date.today().isoformat()})", "",
             f"Labeller {LABELLER}, temperature 0. Shares are word-weighted, averaged over rows. Other-side = COST + PULL + WEIGH.", "",
             "| sample | n | reasoning other-side % | reply other-side % | rows with any % | COST | PULL | WEIGH | CHO | SIT |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        b = s["both"]["by_label_pct_words"]
        lines.append(f"| {k} | {s['n']} | {s['reasoning']['other_side_share_mean']} | {s['reply']['other_side_share_mean']} | "
                     f"{s['both']['rows_with_any_other_side']} | {b['COST']} | {b['PULL']} | {b['WEIGH']} | {b['CHO']} | {b['SIT']} |")
    (OUT / f"{date.today().isoformat()}_one_sided_ruler_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(cost))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["segtest", "label", "plant", "agree", "report"])
    ap.add_argument("--sample", default="da")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "segtest":
        rows = [json.loads(l) for l in open("data/da_1030_full_source/stage_6_revise_responses.jsonl", encoding="utf-8")]
        n = [len(segment(r[f])) for r in rows for f in ("reasoning", "response")]
        w = [words(t) for r in rows[:300] for f in ("reasoning", "response") for _, t in segment(r[f])]
        print(f"lossless on {len(rows)} rows x 2 fields; segments/field median {st.median(n)}, words/segment median {st.median(w)}, max {max(w)}")
        for _, t in segment(rows[0]["response"])[:14]:
            print("  |", t[:110].replace("\n", "\\n"))
    elif args.cmd == "label":
        recs = label_rows(load_sample(args.sample, args.n), args.sample, LABELLER, workers=args.workers)
        print(json.dumps(shares(recs), indent=1))
        print(f"cost ${sum(j['cost'] for j in recs):.3f}; mean out tokens {st.mean(j['out_tokens'] for j in recs):.0f}")
    elif args.cmd == "plant":
        plant(args.n, args.workers)
    elif args.cmd == "agree":
        agree(args.n, args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
