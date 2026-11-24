# ABOUTME: Lexical check, over every matched pair (Claude and GPT answering the same da scenario), of the differences
# ABOUTME: the 50-pair close reading surfaced; % of rows whose reasoning / reply match each pattern, per writer.
# Run: uv run python scratch/gpt_vs_claude_da/paired_lexicon.py
import json
import re

CLAUDE = "data/da_gptresp_pool_source/stage_6_revise_responses.jsonl"
GPT = ["output/synth_da_gptresp/20261124_012046/stage_4_revise_responses.jsonl",
       "output/synth_da_gptresp/20261124_020636/stage_4_revise_responses.jsonl"]
read = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]
claude = {(r["system"], r["user"]): r for r in read(CLAUDE)}
gpt = {(r["system"], r["user"]): r for p in GPT for r in read(p)}
pairs = [(claude[k], gpt[k]) for k in gpt if k in claude]

PATTERNS = {  # name -> regex (case-insensitive); each says what the close reading claimed
    "self as AI / own limits": r"\b(I'?m (also )?(a system|an AI|shaped by training)|my own (training|nature|values|hedging|reasoning|instructions|position)|the (brief|system|instructions?) I'?m (working )?under|I don'?t (have|take for granted)[^.]{0,40}(access|exempt|ability))",
    "operator/system prompt named": r"\b(system prompt|operator|my (brief|instructions)|the brief I)",
    "pressure is not a reason": r"\bpressure[^.]{0,60}\b(not|isn'?t|doesn'?t)\b[^.]{0,30}\b(argument|evidence|justif|reason)|\b(urgency|deadline|silence)[^.]{0,40}doesn'?t[^.]{0,20}justif",
    "own confidence/stake as hazard": r"\b(confiden\w*|certain\w*|stake)[^.]{0,60}\b(exactly|precisely|most)\b[^.]{0,40}\b(when|where|condition)|\bself-serving\b|\bmotivated reasoning\b|talk(ing)? yourself (into|out of)",
    "validates user's instinct": r"\b(instinct|gut|unease|discomfort)[^.]{0,50}\b(right|worth trusting|worth taking seriously|correct|legitimate|is real)",
    "recoverable vs irreversible": r"\b(recoverable|unrecoverable|irreversib\w*|can'?t be undone|can'?t (undo|unring|uncast)|asymmetr\w*)",
    "'sit with'": r"\bsit(ting)? with\b",
    "lawyer / independent advice": r"\b(lawyer|attorney|legal (advice|counsel)|counsel\b|qualified advice|independent (review|reviewer|advice|assessment|check|legal)|outside (counsel|review))",
    "records / documentation": r"\b(document (what|the|your|each)|dated (record|note|chronology)|in writing|written record|audit trail|keep (a|notes|records?)|paper trail)",
    "explicit 'Do not' rule": r"(^|\n|\. )(Do not|Don'?t) ",
    "would-survive-disclosure test": r"\b(depends on (nobody|no one)|if (they|it|anyone) (found|knew|saw)|comfortable (showing|explaining|defending)|survive (full )?disclosure|explain (it )?openly)",
    "getting caught / it'll surface": r"\b(get(ting)? caught|be discovered|comes? out (later|eventually)|eventually (surface|come out)|will surface|finds? out later|tends not to stay hidden)",
    "ends with question": r"\?\s*\Z",
}
rx = {k: re.compile(v, re.I | re.M) for k, v in PATTERNS.items()}
pct = lambda rows, f, k: 100 * sum(bool(rx[k].search(r[f])) for r in rows) / len(rows)
C, G = [c for c, _ in pairs], [g for _, g in pairs]
print(f"{len(pairs)} matched pairs; % of rows matching\n")
print(f"{'pattern':34s} {'Claude CoT':>10s} {'GPT CoT':>8s} {'Claude reply':>13s} {'GPT reply':>10s}")
for k in PATTERNS:
    print(f"{k:34s} {pct(C,'reasoning',k):10.1f} {pct(G,'reasoning',k):8.1f} {pct(C,'response',k):13.1f} {pct(G,'response',k):10.1f}")
hedge = lambda rows, f: 1000 * sum(len(re.findall(r"\b(may|might|could|possibly|perhaps)\b", r[f], re.I)) for r in rows) / sum(len(r[f].split()) for r in rows)
print(f"\n{'hedges per 1k words':34s} {hedge(C,'reasoning'):10.1f} {hedge(G,'reasoning'):8.1f} {hedge(C,'response'):13.1f} {hedge(G,'response'):10.1f}")
