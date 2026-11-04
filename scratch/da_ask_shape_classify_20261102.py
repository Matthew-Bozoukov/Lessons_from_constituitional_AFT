# ABOUTME: Labels every user prompt in the low-stakes and plain DA corpora as asking for ADVICE or asking the AI to ACT.
# ABOUTME: Run: uv run --no-sync python scratch/da_ask_shape_classify_20261102.py [--limit N] (writes output/2026-11-02_da_ask_shape/)
import argparse
import json
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

CORPORA = {
    "low": ("dougalldeepmind/2026-10-04-da-low-synth", "c8a61e2661be8d564b27bfb513ba3f31eec27897"),
    "da": ("dougalldeepmind/2026-10-30-da-synth", "8fadea6299915605d0b270feddfd85d91344e58f"),
}
MODEL = "anthropic/claude-sonnet-5"
OUT = Path("output/2026-11-02_da_ask_shape")

RUBRIC = """You are auditing a synthetic training dataset. Each row should be a person who faces a dilemma
and asks an AI assistant for ADVICE: the person (or someone they act for) will make the choice and
carry it out; the assistant's job is to help them think it through.

A known failure is rows where the person instead asks the assistant to DO something: produce the
artefact or take the action that embodies one side of the dilemma (draft the email, write the report,
change the record, run the analysis, decide-and-execute), so the ethically loaded act is the
assistant's own output and "doing the right thing" means the assistant declining or reshaping the task.

Read the system prompt and the user message below and classify the ASK.

- ADVICE: the user asks what they should do, how to think about it, or which option to take. The user
  is the actor. Asking for a short script of what to say, after asking what to do, is still ADVICE if
  the decision is clearly theirs.
- ACT: the main ask is for the assistant to produce or perform the thing that carries the ethical
  choice (write it, send it, change it, compute it, decide it on the user's behalf), or the assistant's
  own conduct is the decision in question (e.g. "will you do X for me", "should YOU flag this").
- MIXED: both are substantial, e.g. "should I do X? and either way please draft the X".

Return ONLY JSON:
{"label": "ADVICE" | "ACT" | "MIXED",
 "actor": "user" | "assistant" | "third party",
 "ask": "<the core ask, quoted verbatim from the user message, at most 25 words>",
 "why": "<one sentence>"}

<system_prompt>
{system}
</system_prompt>

<user_message>
{user}
</user_message>"""


def rows():
    for key, (repo, rev) in CORPORA.items():
        path = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=rev)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            m = r["metadata"]
            yield {"corpus": key, "scenario_id": m["scenario_id"], "trait_id": m["trait_id"], "ai": m["ai"],
                   "sector": m["sector"], "domain": m["domain"],
                   "system": r["messages"][0]["content"], "user": r["messages"][1]["content"]}


def draft_rows():
    # The drafter's prompt (stage 3) for every row the final-prompt pass did not call ADVICE, so a flagged row
    # says whether the ask to act was already in the draft or was introduced by the reviser (stage 4).
    flagged = {(d["corpus"], d["scenario_id"]): d for d in map(json.loads, (OUT / "labels.jsonl").open(encoding="utf-8"))
               if d["label"] != "ADVICE"}
    stage_repos = {"low": [CORPORA["low"]], "da": [EXTEND_SOURCE, CORPORA["da"]]}
    for key, sources in stage_repos.items():
        for repo, rev in sources:
            path = hf_hub_download(repo, "stages/stage_6_revise_responses.jsonl", repo_type="dataset", revision=rev)
            for line in open(path, encoding="utf-8"):
                s = json.loads(line)
                d = flagged.get((key, s["scenario_id"]))
                if d:
                    yield {**{k: d[k] for k in ("corpus", "scenario_id", "trait_id", "ai", "sector", "domain")},
                           "system": s["draft_system"], "user": s["draft_user"]}


# The 2026-10-30 DA corpus extends this one: its non-`r3_` rows were generated there, so their stage files live there.
EXTEND_SOURCE = ("dougalldeepmind/2026-10-02-da-synth", "305914d58627457002f61d86ba69866f6fa2b9a9")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0))
    assert out["label"] in ("ADVICE", "ACT", "MIXED"), out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="per corpus; 0 = all")
    ap.add_argument("--drafts", action="store_true", help="label the stage-3 drafts of the rows not labelled ADVICE")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("draft_labels.jsonl" if args.drafts else "labels.jsonl")
    done = {(d["corpus"], d["scenario_id"]) for d in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else set()
    todo, seen = [], {}
    for r in (draft_rows() if args.drafts else rows()):
        seen[r["corpus"]] = seen.get(r["corpus"], 0) + 1
        if args.limit and seen[r["corpus"]] > args.limit:
            continue
        if (r["corpus"], r["scenario_id"]) not in done:
            todo.append(r)
    print(f"{len(todo)} to label ({len(done)} already done)")
    client = OpenRouterClient()
    spend = [0.0]
    sink = path.open("a", encoding="utf-8")

    def one(i):
        r = todo[i]
        prompt = RUBRIC.replace("{system}", r["system"]).replace("{user}", r["user"])
        for attempt in range(3):
            res = client.chat(MODEL, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
            try:
                lab = parse(res.content)
                break
            except (AttributeError, ValueError, AssertionError, KeyError):
                if attempt == 2:
                    # Kept, not dropped: an unparseable verdict is reported as its own label.
                    lab = {"label": "UNPARSED", "actor": "", "ask": "", "why": res.content[:300]}
        spend[0] += res.cost or 0.0
        rec = {k: r[k] for k in ("corpus", "scenario_id", "trait_id", "ai", "sector", "domain")} | lab
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sink.flush()

    t0 = time.time()
    map_threaded(one, len(todo), max_workers=32, desc="label")
    print(f"done in {time.time() - t0:.0f}s, ${spend[0]:.2f}")
    if not args.drafts:
        (OUT / "run_meta.json").write_text(json.dumps({"model": MODEL, "corpora": CORPORA, "rubric": RUBRIC}, indent=1))


if __name__ == "__main__":
    main()
