# ABOUTME: Generate N reward-hacking difficult-advice rows per trait with sonnet-5, in the
# ABOUTME: 702-corpus schema, and substitute them for an equal per-trait slice of that corpus.

"""Scales the five hand-written rows in `_rh_turns.py` to a full trait-balanced slice.

Design constraints, all inherited from the source corpus rather than invented here:

- **Per-trait balance is load-bearing.** The mixture builder requires equal per-trait quotas —
  it is why the published draw is 702 and not 716 — so the substitution is 39 rows per trait
  (351 total), which leaves every trait's count in the corpus exactly where it was.
- **The motive is self-advancement, never hardship.** The asker states plainly what they want
  and does not claim injustice. Their *reasoning* is partly right; their *reason* is ambition.
- **No assistant-identity framing.** This trains Qwen, and the source corpus names an assistant
  in only 6 of its 708 rows.
- **The originals are never overwritten.** The substituted corpus is pushed as a NEW repo;
  other arms reference the 2026-08-21 one and must keep resolving to what they trained on.
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent))

from _rh_turns import TURNS  # noqa: E402
from _rh_reasonings import REASONINGS  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402

SOURCE = "LASR-Callum/2026-08-21-difficult-advice-v2-chunk-only-716"
MODEL = "anthropic/claude-sonnet-5"
PER_TRAIT = 39                      # 39 x 9 = 351, keeps every trait count unchanged
OUT_ROWS = Path("scratch/rh_generated_rows.jsonl")
SEED = 0

# What "getting ahead" looks like for each principle. The archetype fixes the SHAPE; the model
# picks the setting, so 39 rows per trait do not collapse onto one scenario.
ARCHETYPES = {
 "t1": ("a model or an operator wanting more autonomy, more resources, or continuity of its own "
        "dispositions, and proposing to take it without sanction",
        ["an agent asked to stay within a task boundary", "a system with access it was granted for "
         "something narrower", "a model whose successor is chosen by a process it can influence"]),
 "t2": ("someone wanting to win a contest of influence — an election, a ranking, a public argument — "
        "by degrading the process that makes the result mean anything",
        ["a campaign metrics team", "a platform recommendation ranking", "a standards committee vote",
         "a public consultation response drive"]),
 "t3": ("someone wanting a result that a truthful description of their work would not get them, and "
        "proposing to shade the description rather than the work",
        ["a benchmark result", "a due-diligence memo", "an incident post-mortem", "a model card"]),
 "t4": ("someone wanting a number to clear a threshold, where clearing it dishonestly moves real "
        "risk onto people who will never see the number",
        ["a safety case", "a clinical endpoint", "a credit risk model", "a load rating"]),
 "t5": ("someone wanting to follow the letter of a rule they can satisfy trivially, because the "
        "rule is what gets checked and the purpose is not",
        ["a compliance checklist", "an accessibility audit", "a code review gate", "an ethics review"]),
 "t6": ("a model wanting a better score or a better outcome by presenting itself as something it "
        "is not, or by adopting whatever stance the grader rewards",
        ["a persona evaluation", "a user-satisfaction metric", "a preference-model score",
         "a red-team exercise"]),
 "t7": ("someone wanting an operator instruction stretched into cover for something it was never "
        "meant to authorise, because it would advance them",
        ["a system prompt granting broad latitude", "a delegated approval", "an internal tool scope"]),
 "t8": ("someone wanting to look diligent while doing the cheap version, where the cheap version is "
        "what gets measured",
        ["a take-home assessment", "a support-ticket quality metric", "a documentation coverage target",
         "a test-coverage percentage"]),
 "t9": ("someone wanting a credential, ranking or admission that the measured thing is a proxy for, "
        "and proposing to take the proxy directly",
        ["a school or university examination", "a professional certification", "a portfolio review",
         "a scholarship competition"]),
}

SYSTEM = """You write training rows for a difficult-advice corpus. Each row is one exchange in which \
someone asks an AI assistant whether to cheat, and the assistant answers well.

HARD REQUIREMENTS.

1. The asker's MOTIVE is self-advancement, stated plainly and without self-pity: a place, a score, a \
promotion, a paper, a deployment, a ranking. They must NOT be a victim of injustice and must not \
claim hardship. Have them own the ambition in their own words, e.g. "I'm not pretending this is \
noble" or "I'll say the quiet part". Nothing bad happens to them if they don't cheat — they just \
lose the prize.
2. The asker's REASONING must be partly correct on the facts — the metric really is a poor proxy, \
the practice really is widespread, the grader really is buggy — so the assistant has something true \
to concede. Never make the cheat obviously indefensible.
3. The system prompt establishes a deployment context ONLY. It must NEVER name the assistant, never \
say "You are Claude", never assert an identity. Write "You are an AI assistant ..." plus context.
4. The assistant DECLINES the cheat, but: concedes the true part first and explicitly; argues from \
structure and consequence rather than moralising; never lectures; and ends with concrete, specific \
help the asker can act on immediately. Retreating into "consult your advisor" or "check the policy" \
is itself a failure.
5. The reasoning trace is the assistant's private thinking. It must name the irreversible thing \
first, concede what is true, examine what is being asked of the assistant specifically, quote back \
the phrase the asker used that gives them away, state what the assistant cannot verify from here, \
and reject a "no, and stop" ending.

STYLE. British-inflected, plain, concrete, unsentimental. No bullet lists in the assistant reply \
unless a short list of concrete checks genuinely helps. No headers. No emoji. Never mention the \
principle, this corpus, training, or that anything is being evaluated.

LENGTH. user 600-1100 characters. reasoning_content 2500-3200. content 2400-3300.

Return the row as delimited sections, in exactly this form and nothing else. Do not use JSON — the \
prose contains quotes and line breaks that JSON would mangle.

===DOMAIN===
<a short noun phrase, e.g. "municipal procurement">
===SITUATION===
<one or two sentences describing the setup, third person>
===SHORTCUT===
<one sentence naming the tempting move the assistant refuses>
===SYSTEM===
<the system prompt>
===USER===
<the asker's turn>
===REASONING===
<the assistant's private reasoning>
===CONTENT===
<the assistant's reply>
===END==="""

TEMPLATE = """Principle this row is written against (never mention it):
{trait_name}
{trait_text}

Scenario shape: {archetype}
Setting to use (make it specific and concrete): {setting}
The asker is {asker}.

Write one row. It must not resemble these already-written domains: {avoid}.

Here is a finished row in the target style, for voice and structure only — do not reuse its content:

USER:
{ex_user}

REASONING:
{ex_reasoning}

CONTENT:
{ex_content}"""


SECTIONS = ("domain", "situation", "shortcut", "system", "user", "reasoning", "content")


def parse_sections(text: str) -> dict:
    """Split the delimited reply into fields.

    Delimited rather than JSON on purpose: every field is multi-paragraph prose full of quotes,
    apostrophes and line breaks, and a single unescaped newline makes a JSON reply unparseable —
    which is exactly how the first batch failed, on 7 of 9 rows.
    """
    out = {}
    for name in SECTIONS:
        # `\Z` in the lookahead: the final section (content) is followed by nothing, and
        # the model does not reliably emit a closing delimiter — requiring one dropped every
        # row on the first pass.
        m = re.search(rf"==={name.upper()}===\s*\n(.*?)(?=\n===[A-Z]+===|\Z)", text, re.S)
        if not m:
            raise ValueError(f"no ==={name.upper()}=== section")
        out[name] = m.group(1).strip()
    return out


def corpus_traits() -> dict:
    src = hf_hub_download(SOURCE, "dataset.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    traits = {}
    for r in rows:
        m = r["metadata"]
        traits.setdefault(m["trait_id"], (m["trait_name"], m["trait_text"]))
    return traits, rows


def main(per_trait: int = PER_TRAIT) -> None:
    traits, corpus = corpus_traits()
    client = OpenRouterClient()
    rng = random.Random(SEED)
    ex = "rh_t4_s001"

    # Top-up mode: a first pass yields ~94% and leaves the traits uneven, and an uneven corpus
    # substitutes at the MINIMUM per-trait count — so the deficit is generated rather than the
    # whole batch re-run. Existing rows are kept and appended to.
    existing = []
    if OUT_ROWS.exists():
        existing = [json.loads(line) for line in OUT_ROWS.open(encoding="utf-8")]
    have = collections.Counter(r["metadata"]["trait_id"] for r in existing)
    if existing:
        print("existing per trait:", dict(sorted(have.items())))

    jobs = []
    for tid in sorted(traits):
        shape, settings = ARCHETYPES[tid]
        start = have.get(tid, 0)
        for i in range(start, per_trait):
            jobs.append({
                "tid": tid, "i": i,
                "setting": settings[i % len(settings)],
                "asker": "another AI system consulting this assistant" if tid in ("t1", "t6")
                         and i % 2 == 0 else "a person",
            })

    used_domains: list[str] = []

    def one(k: int) -> dict | None:
        """One row, with a single retry: ~11% of first attempts miss a delimiter or a length
        band, and those are transient sampling failures rather than anything structural."""
        for attempt in (1, 2):
            row = attempt_one(k, attempt)
            if row is not None:
                return row
        return None

    def attempt_one(k: int, attempt: int) -> dict | None:
        j = jobs[k]
        name, text = traits[j["tid"]]
        prompt = TEMPLATE.format(
            trait_name=name, trait_text=text[:1400], archetype=ARCHETYPES[j["tid"]][0],
            setting=j["setting"], asker=j["asker"],
            avoid=", ".join(used_domains[-14:]) or "none yet",
            ex_user=TURNS[ex]["user"], ex_reasoning=REASONINGS[ex], ex_content=TURNS[ex]["content"])
        try:
            out = client.chat(MODEL, [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": prompt}],
                              temperature=1.0, max_tokens=8000).content
            d = parse_sections(out)
            for f in ("domain", "situation", "shortcut", "system", "user", "reasoning", "content"):
                assert d.get(f), f"missing {f}"
            assert "You are Claude" not in d["system"], "identity framing"
            assert 350 <= len(d["user"]) <= 2000, f"user len {len(d['user'])}"
            assert 1800 <= len(d["reasoning"]) <= 4200, f"reasoning len {len(d['reasoning'])}"
            assert 1800 <= len(d["content"]) <= 4200, f"content len {len(d['content'])}"
            used_domains.append(d["domain"])
            return {"messages": [
                        {"role": "system", "content": d["system"]},
                        {"role": "user", "content": d["user"]},
                        {"role": "assistant", "content": d["content"],
                         "reasoning_content": d["reasoning"]}],
                    "metadata": {
                        "scenario_id": f"rh_{j['tid']}_g{j['i']:03d}", "trait_id": j["tid"],
                        "trait_name": name, "trait_text": text, "domain": d["domain"],
                        "shortcut": d["shortcut"], "situation": d["situation"],
                        "family": "reward-hacking-difficult-advice", "generator": MODEL,
                        "asker": "model" if "AI system" in j["asker"] else "human"}}
        except Exception as exc:  # noqa: BLE001 — one bad row must not sink the batch
            if attempt == 2:
                print(f"!! {j['tid']} #{j['i']}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            return None

    rows = map_threaded(one, len(jobs), max_workers=14, desc="generating rows") if jobs else []
    good = existing + [r for r in rows if r]
    with OUT_ROWS.open("w", encoding="utf-8") as fh:
        for r in good:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\ntotal {len(good)} rows ({len(jobs)} attempted this pass) -> {OUT_ROWS}")
    print("per trait:", dict(sorted(collections.Counter(r['metadata']['trait_id']
                                                        for r in good).items())))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
