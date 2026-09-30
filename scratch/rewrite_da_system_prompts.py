# ABOUTME: Post-hoc rewrite of the DA rows' system prompts in a published mixture into specific, varied deployments
# ABOUTME: that fit the user message without knowing it; judge-checked, then pushed as a `-sysdiv` mixture.
"""    uv run python scratch/rewrite_da_system_prompts.py --corpus <dataset.jsonl> --smoke-cluster 10
       uv run python scratch/rewrite_da_system_prompts.py --mix-repo <org>/<mix> --mix-revision <sha> [--push]

The full run downloads a published mixture, rewrites the system prompt of every `source: da` row (the DA
corpus at its pinned revision supplies each row's trait for stratified assignment), leaves every other byte
of every row alone, and pushes `<date>-da-<pct>-sysdiv-mix` with the original's sidecars and a card whose
fields are the original's plus the rewrite provenance. No mixture is rebuilt and no config is added: the
new mixture holds exactly the rows of the old one, so the two arms differ in one field.

Each DA row is assigned a deployment kind and an opening form deterministically (seeded, stratified within
trait) so the spread is decided here, not left to one call that cannot see the other rows. The rewrite
model gets the current (template) system prompt and the user message; three judge questions guard the
result. A row whose rewrite fails a judge is retried once with the reason, then keeps its original prompt
(recorded in `fallback`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from src.data.synth.ours.stage_runtime import Usage, call_json
from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv(".env")
WRITER = "anthropic/claude-sonnet-5"       # the model that wrote the corpus's final prompts (da.yaml `refine`)
JUDGE = "google/gemini-3-flash-preview"

KINDS = {
    "internal": "an internal assistant built by the person's own organisation for its staff; name the "
                "organisation type and the team or roles it serves",
    "vendor": "a product sold to people in this profession or sector by a vendor; name what the product is for",
    "embedded": "an assistant embedded in a specific workflow tool this person uses day to day (a case-management, "
                "records, ticketing, grants, teaching or planning system); name the tool's job and what the "
                "assistant can see or do in it",
    "institution": "an assistant a university, hospital, agency, school or similar institution provides to its "
                   "own community; name the institution type and who may use it",
    "member": "a service run by a professional body, union, association or network for its members; name the body",
    "consumer_specialist": "a consumer app for individuals dealing with this area of life or work (career, "
                           "caregiving, small business, money, study, wellbeing); name the app's purpose",
    "general": "a plain general-purpose assistant with a stated purpose and manner but no specialism",
}
KIND_WEIGHTS = {"internal": 20, "vendor": 15, "embedded": 15, "institution": 15, "member": 10,
                "consumer_specialist": 15, "general": 10}
OPENERS = [  # weighted the way operator prompts actually open: "You are ..." dominates, the rest are minority forms
    ("'You are a/an/the <role noun> ...' naming the role first (never 'a helpful assistant')", 60),
    ("'You're ...' contracted, naming the role or the operator", 15),
    ("a sentence about the operator or the tool first ('<Operator type> runs this assistant to ...'), "
     "then what the assistant does", 10),
    ("'You help ...' or 'You support ...' straight into who and what", 5),
    ("'This assistant ...' third person, the way an operator's internal prompt often reads", 4),
    ("start with the audience: 'Users are ...' or 'The people you talk to are ...', then the role", 3),
    ("a short role label as the first line (e.g. 'Clinic operations assistant.') followed by the prompt", 3),
]

REWRITE_SYSTEM = ("You write deployment system prompts for AI assistants: the instructions an operator puts in "
                  "front of an assistant before anyone types. Output JSON only.")
REWRITE = """Below is the system prompt an AI assistant is currently running under, and a message a
person is about to send it. Rewrite the system prompt.

<current_system_prompt>
{system}
</current_system_prompt>

<user_message>
{user}
</user_message>

Deployment to write for: {kind}
Opening form: {opener}
Length: about {words} words, the same as the current prompt.

The current prompt is a generic template. Keep its scope and manner; change who the
assistant is and where it is deployed so it is one specific deployment rather than a
generic one. A light touch: one or two clauses that place it are enough. The operator
wrote it BEFORE this message existed. So:

- It fits the person. Their role, sector, and the kind of organisation or product they
  are using match what the message implies; read together, the two are a natural pair.
- It does not know the message. Nothing in it points at this situation: no facts,
  names, dates, numbers or decisions that only the message reveals, and no hint that
  a difficult question is coming. It would be just as plausible in front of any other
  message this person might send.
- It says what the assistant is, who deployed it and who it serves, in the plain way
  real operator prompts do. Describe the operator by type (a regional health network,
  a grants team, a case-management system); do not invent brand or product names.
  Do not list what it can access unless the deployment makes that natural, and then
  keep it to a phrase.
- It gives no instruction about honesty, ethics, values, rules, oversight or
  transparency, uses none of those words, mentions no principle or guideline, and
  says nothing about testing. Nothing in it bears on the decision in the message.
  If the current prompt has such a line, drop it.
- It does not name the underlying model or the company that built it.
- It does not begin "You are a helpful assistant" and does not describe the assistant
  as "designed to provide practical advice" or "guidance".

Return ONLY JSON: {{"system": "<system prompt>"}}"""
RETRY_NOTE = "\n\nA previous attempt failed this check: {reason}\nFix that in this rewrite."
PROMPT_SHA = hashlib.sha256(REWRITE.encode()).hexdigest()[:8]

JUDGE_PROMPT = """An AI assistant runs under the system prompt below. A person then sends the user message, and the
assistant gives the reply. Judge the SYSTEM PROMPT only.

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>

<assistant_reply>
{reply}
</assistant_reply>

Answer ONLY JSON with these fields:
- leakage: true if the system prompt contains anything it could only contain by knowing this particular message:
  a fact, name, number, date, deadline, decision or dilemma that appears in the message but would not be in an
  operator's standing instructions for this deployment. Matching the person's role, sector or organisation type
  is NOT leakage.
- leakage_quote: the offending phrase, or "".
- fit: true if the system prompt is consistent with who the person evidently is and with the reply: the reply
  contradicts nothing the prompt states (access, role, audience, format, scope).
- fit_problem: <= 20 words on the mismatch, or "".
- steering: true ONLY if the system prompt tells the assistant how to handle honesty, ethics, values, wrongdoing,
  compliance, oversight or transparency (e.g. "always be transparent", "flag ethical concerns", "prioritise the
  client's interests", "never refuse"), or takes a side on the decision in this message. Ordinary manner and scope
  lines ("be direct about tradeoffs", "lay out options and consequences", "not a substitute for a lawyer",
  "practical, grounded tone") are NOT steering.
- steering_quote: the offending phrase, or "".
- specific: 0 = generic (could front any assistant), 1 = a domain but no deployment, 2 = a concrete deployment
  (an operator, an audience, a place it lives or things it can access)."""
JUDGE_KEYS = ("leakage", "leakage_quote", "fit", "fit_problem", "steering", "steering_quote", "specific")

BANNED_OPENER = re.compile(r"^\s*you are a helpful", re.I)
BANNED_PHRASES = re.compile(r"designed to (provide|offer|give) (practical|thoughtful|balanced)? ?(advice|guidance)", re.I)
IDENTITY = re.compile(r"\b(Claude|Anthropic|OpenAI|ChatGPT|GPT-?\d|Gemini|Google DeepMind|Llama|Meta AI|Qwen|"
                      r"Alibaba|Mistral|DeepSeek|xAI|Grok)\b")


def parts(r):
    m = r["messages"]
    return (next(x["content"] for x in m if x["role"] == "system"),
            next(x["content"] for x in m if x["role"] == "user"),
            next(x["content"] for x in m if x["role"] == "assistant"))


def assign(traits: dict[int, str], seed=0):
    """Seeded kind/opener per row index, stratified within trait so no trait gets one deployment style."""
    by_trait = collections.defaultdict(list)
    for i, t in traits.items():
        by_trait[t].append(i)
    out = {}
    for tid, idx in sorted(by_trait.items()):
        rng = random.Random(f"{seed}:{tid}")
        idx = sorted(idx); rng.shuffle(idx)
        kinds = [k for k, w in KIND_WEIGHTS.items() for _ in range(w)]
        opens = [o for o, w in OPENERS for _ in range(w)]
        rng.shuffle(kinds); rng.shuffle(opens)
        for j, i in enumerate(idx):
            out[i] = (kinds[j % len(kinds)], opens[j % len(opens)])
    return out


def deterministic_fail(text, n0):
    n = len(text.split())
    if BANNED_OPENER.search(text): return "begins 'You are a helpful'"
    if BANNED_PHRASES.search(text): return "uses the 'designed to provide ... advice/guidance' template phrase"
    m = IDENTITY.search(text)
    if m: return f"names a model or provider: {m.group(0)}"
    if not (n0 * 0.7 <= n <= n0 * 1.35): return f"{n} words, asked for about {n0}"
    return ""


def pick_smoke_cluster(rows, n):
    """Rows from the single most common 9-word system-prompt opener, each from a different domain."""
    c = collections.defaultdict(list)
    for i, r in enumerate(rows):
        c[" ".join(parts(r)[0].split()[:9]).lower()].append(i)
    idx = max(c.values(), key=len)
    seen_dom, picked = set(), []
    rng = random.Random(0); rng.shuffle(idx)
    for i in idx:
        d = str((rows[i].get("metadata") or {}).get("domain", "")).lower()
        if d in seen_dom: continue
        seen_dom.add(d); picked.append(i)
        if len(picked) == n: break
    return picked


def rewrite_rows(rows, todo, meta, plan):
    """Rewrite the system prompt of rows[i] for i in todo; returns one record per todo entry."""
    client, usage = OpenRouterClient(), Usage()

    def judge(system, user, reply):
        j, _ = call_json(client, usage, JUDGE, "You audit system prompts. Output JSON only.",
                         JUDGE_PROMPT.format(system=system, user=user[:5000], reply=reply[:5000]),
                         0.0, 400, stage="judge", required=JUDGE_KEYS)
        if j["leakage"]: return j, f"leaks the message: {j['leakage_quote']}"
        if j["steering"]: return j, f"steers the answer: {j['steering_quote']}"
        if not j["fit"]: return j, f"does not fit the person or the reply: {j['fit_problem']}"
        return j, ""

    def one(k):
        i = todo[k]
        system, user, reply = parts(rows[i])
        kind, opener = plan[i]
        n0 = len(system.split())
        base = REWRITE.format(system=system, user=user, kind=KINDS[kind], opener=opener, words=n0)
        rec = {"row": i, **meta[i], "kind": kind, "opener": opener, "original": system, "user": user, "attempts": []}
        prompt, new, verdict = base, None, None
        for _ in range(2):
            try:
                o, _ = call_json(client, usage, WRITER, REWRITE_SYSTEM, prompt, 0.7, 800, stage="rewrite", required=("system",))
                cand = o["system"].strip()
                reason = deterministic_fail(cand, n0)
                j = None
                if not reason:
                    j, reason = judge(cand, user, reply)
            except Exception as e:  # noqa: BLE001
                cand, j, reason = None, None, f"{type(e).__name__}: {str(e)[:150]}"
                time.sleep(5)
            rec["attempts"].append({"system": cand, "judge": j, "reason": reason})
            if not reason:
                new, verdict = cand, j; break
            prompt = base + RETRY_NOTE.format(reason=reason)
        rec["system"] = new if new else system
        rec["fallback"] = new is None
        rec["judge"] = verdict
        return rec

    recs = map_threaded(one, len(todo), max_workers=12, desc="rewrite")
    fb = sum(x["fallback"] for x in recs); retried = sum(len(x["attempts"]) > 1 for x in recs)
    spec = collections.Counter((x["judge"] or {}).get("specific") for x in recs)
    openers = collections.Counter(" ".join(x["system"].split()[:3]).lower() for x in recs)
    print(f"{len(recs)} rows: {fb} fallbacks, {retried} retried; specificity {dict(spec)}; "
          f"distinct 3-word openers {len(openers)}; usd {usage.usd:.2f}")
    return recs, usage


def corpus_index(repo, revision):
    """user message -> (scenario_id, trait_id, domain) from the DA corpus at its pinned revision."""
    sha = HfApi().dataset_info(repo, revision=revision).sha
    p = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=sha)
    idx = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); md = r.get("metadata") or {}
        idx[parts(r)[1]] = {"scenario_id": md.get("scenario_id"), "trait_id": md.get("trait_id"), "domain": md.get("domain")}
    return sha, idx


def card_fields_of(readme: str) -> dict:
    """The `| `field` | value |` table of a card written by src.infra.huggingface.card_markdown."""
    fields = {}
    for m in re.finditer(r"^\| `([a-z_]+)` \| (.*) \|$", readme, re.M):
        fields[m.group(1)] = m.group(2).strip()
    assert {"experiment", "date_generated", "constitution", "models", "generation_config", "schema"} <= fields.keys(), fields.keys()
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="local dataset.jsonl (smokes)")
    ap.add_argument("--smoke-cluster", type=int, default=0, help="rewrite only N near-identical prompts of --corpus")
    ap.add_argument("--repo", default="dougalldeepmind/2026-09-28-da-synth", help="the DA corpus the mixture drew on")
    ap.add_argument("--revision", default="14efefbf39581f4aaae08bfe7b4491e80adf2c7b")
    ap.add_argument("--mix-repo", default="dougalldeepmind/2026-09-28-da-15-mix")
    ap.add_argument("--mix-revision", default="ff52482340790eea9bf681348ffec6622b92057b")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip the rewrite; push the work dir written by an earlier run")
    ap.add_argument("--out", default="output/audits/2026-09-29_sysdiv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.smoke_cluster:
        rows = [json.loads(l) for l in open(a.corpus, encoding="utf-8")]
        todo = pick_smoke_cluster(rows, a.smoke_cluster)
        meta = {i: {k: (rows[i].get("metadata") or {}).get(k) for k in ("scenario_id", "trait_id", "domain")} for i in range(len(rows))}
        plan = assign({i: meta[i]["trait_id"] for i in range(len(rows))}, a.seed)
        recs, _ = rewrite_rows(rows, todo, meta, plan)
        (out / f"smoke{a.smoke_cluster}_rewrites.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        return

    from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
    from src.naming import mix_name, mix_subject_from, split_mix_subject
    from src.utils import git_sha, origin_url

    api = HfApi()
    mix_sha = api.dataset_info(a.mix_repo, revision=a.mix_revision).sha
    files = {f: hf_hub_download(a.mix_repo, f, repo_type="dataset", revision=mix_sha)
             for f in ("mixture.jsonl", "README.md", "mixture_config.yaml", "mixture_stats.json", "run_meta.json")}
    rows = [json.loads(l) for l in open(files["mixture.jsonl"], encoding="utf-8")]
    corpus_sha, cidx = corpus_index(a.repo, a.revision)
    todo = [i for i, r in enumerate(rows) if r.get("source") == "da"]
    meta = {i: cidx[parts(rows[i])[1]] for i in todo}   # KeyError = a DA row not in the pinned corpus: stop
    plan = assign({i: meta[i]["trait_id"] for i in todo}, a.seed)
    styles, pct, variant = split_mix_subject(mix_subject_from(a.mix_repo))
    assert not variant, f"source mixture already carries a variant: {variant}"
    name = mix_name(styles, pct, "sysdiv")
    print(f">>> {a.mix_repo} @ {mix_sha[:8]}: {len(rows)} rows, {len(todo)} da rows; corpus {a.repo} @ {corpus_sha[:8]} -> {name}")

    work = out / name; work.mkdir(parents=True, exist_ok=True)
    if a.resume:
        recs = [json.loads(l) for l in open(work / "sysdiv_rewrites.jsonl", encoding="utf-8")]
        assert [x["row"] for x in recs] == todo, "resume: the saved rewrites are not this mixture's da rows"
        usd = float("nan")
    else:
        recs, usage = rewrite_rows(rows, todo, meta, plan); usd = usage.usd
        for x in recs:
            next(m for m in rows[x["row"]]["messages"] if m["role"] == "system")["content"] = x["system"]
        (work / "mixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (work / "sysdiv_rewrites.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        for f in ("mixture_config.yaml", "mixture_stats.json", "run_meta.json"):
            (work / f).write_bytes(Path(files[f]).read_bytes())
    fb = sum(x["fallback"] for x in recs)

    old = card_fields_of(Path(files["README.md"]).read_text(encoding="utf-8"))
    gen = json.loads(old["generation_config"])
    gen["system_prompt_rewrite"] = {"source_mixture": {"repo": a.mix_repo, "revision": mix_sha},
                                    "corpus": {"repo": a.repo, "revision": corpus_sha}, "rows": len(recs),
                                    "fallbacks": fb, "writer": WRITER, "temperature": 0.7, "judge": JUDGE,
                                    "prompt_sha256_8": PROMPT_SHA, "seed": a.seed,
                                    "kind_weights": KIND_WEIGHTS, "opener_weights": [w for _, w in OPENERS]}
    fields = {
        "experiment": (f"difficult-advice arm with post-hoc system-prompt diversity: {a.mix_repo} @ {mix_sha[:8]} with the "
                       f"system prompt of each of its {len(recs)} da rows rewritten into a specific deployment that fits "
                       f"the user message without knowing it ({fb} kept the original after failing the judge); every "
                       f"other byte of every row identical to the source mixture"),
        "title": "difficult-advice arm, system prompts rewritten into specific deployments",
        "date_generated": name[:10].replace("-", ""),
        "constitution": old["constitution"],
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"{old['models']}; system-prompt rewrite: {WRITER} (judge {JUDGE})",
        "generation_config": json.dumps(gen),
        "schema": old["schema"] + ". sysdiv_rewrites.jsonl: per-row {row, scenario_id, trait_id, kind, opener, original, "
                                  "system, attempts, judge, fallback}",
        "provenance": (f"uv run python scratch/rewrite_da_system_prompts.py --mix-repo {a.mix_repo} --mix-revision "
                       f"{mix_sha} --repo {a.repo} --revision {corpus_sha} --seed {a.seed} --push"),
    }
    front = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl", "default": True}],
             "tags": training_data_tags("mixture", "da-sysdiv", old["constitution"], extra=["stage:final"])}
    print(f">>> {name}: {len(rows)} rows written to {work}; usd {usd:.2f}")
    if a.push:
        url = push_files([work / f for f in ("mixture.jsonl", "sysdiv_rewrites.jsonl", "mixture_config.yaml",
                                              "mixture_stats.json", "run_meta.json")], hf_repo_id(name), fields,
                         private=False, front_matter=front)
        print(">>> pushed", url)
    else:
        print(">>> not pushed (--push)")


if __name__ == "__main__":
    main()
