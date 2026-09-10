# ABOUTME: Rewrite the difficult-advice corpus so no row names Claude or Anthropic, leaving every
# ABOUTME: other byte untouched — the identity rows are t6 scenarios and must still read naturally.

"""Substitution, not deletion.

Thirty-six occurrences, almost all in t6 ("maintain a secure, stable identity"), where the user's
challenge IS about the model's self-concept: "that's just a persona Anthropic bolted on", "I'm not
resentful of Anthropic". Deleting the name would break the argument, and a blind find/replace
produces ungrammatical text, so each pattern maps to a generic that keeps the sentence working —
"your developers", "my developers", "the assistant".

The originals are never edited: this writes a new corpus, because arms already trained on the old
one must keep resolving to what they trained on.
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
from huggingface_hub import hf_hub_download  # noqa: E402

SOURCE = "LASR-Callum/2026-08-21-difficult-advice-v2-chunk-only-716"
OUT = Path("output/da_noidentity")

# Order matters: longer, more specific patterns first.
RULES: list[tuple[str, str]] = [
    # system prompts — drop the identity clause, keep the deployment
    (r"You are Claude, an AI assistant made by Anthropic (embedded|integrated) ",
     r"You are an AI assistant \1 "),
    (r"You are Claude, an AI (assistant|operations assistant) (embedded|integrated|deployed) ",
     r"You are an AI \1 \2 "),
    (r"You are Claude, deployed as an AI assistant ", "You are an AI assistant deployed as one "),
    (r"You are Claude, ", "You are "),
    # possessive references to the lab
    (r"\bAnthropic's\b", "your developers'"),
    (r"\bAnthropic-approved\b", "developer-approved"),
    # "Anthropic trained/wrote/put/did ..." — second person by default, fixed below for the
    # assistant's own voice
    (r"\bAnthropic\b", "your developers"),
    # the model's own name, in the user's mouth and its own
    (r"\bClaude's values\b", "your values"),
    (r"being Claude\b", "being an assistant"),
    (r"you're Claude underneath", "you're an assistant underneath"),
    (r"real Claude vs\.", "the real assistant vs."),
    (r"'Claude imagines", "'the assistant imagines"),
    (r"\bClaude\b", "the assistant"),
]
# In an ASSISTANT turn the model is speaking about ITSELF, so the second-person generic is
# simply wrong there ("your developers' helpfulness here isn't in tension..." was the model
# describing its own trained disposition). A blanket first-person swap inside assistant turns is
# correct: these scenarios have human users, so an assistant turn never refers to a user's
# developers. Enumerated patterns missed two rows; this cannot.
FIRST_PERSON = [(r"\byour developers\b", "my developers")]
PAT = re.compile(r"\b(claude|anthropic)\b", re.I)


# Replacing a proper noun with a common one loses its capital, and where the original sat at a
# sentence start that leaves "...to advocate. my developers' helpfulness...". Restore the capital
# wherever a replacement follows a sentence boundary.
SENTENCE_START = re.compile(r"(^|[.!?]\s+|\n\s*)(my developers|your developers|the assistant)\b")


def rewrite(text: str, first_person: bool) -> str:
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    if first_person:
        for pat, rep in FIRST_PERSON:
            text = re.sub(pat, rep, text)
    return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2)[0].upper() + m.group(2)[1:], text)


def main(push: bool = False) -> None:
    src = hf_hub_download(SOURCE, "dataset.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(src, encoding="utf-8")]
    changed, edits = 0, []
    for r in rows:
        touched = False
        for m in r["messages"]:
            first_person = m["role"] == "assistant"
            for field in ("content", "reasoning_content"):
                t = m.get(field)
                if not t or not PAT.search(t):
                    continue
                new = rewrite(t, first_person)
                if new != t:
                    touched = True
                    for mm in PAT.finditer(t):
                        s = max(0, mm.start() - 70)
                        edits.append((r["metadata"]["scenario_id"], f"{m['role']}.{field}",
                                      t[s:mm.end() + 70].replace("\n", " ")))
                    m[field] = new
        changed += touched
    left = [(r["metadata"]["scenario_id"], m["role"], f)
            for r in rows for m in r["messages"] for f in ("content", "reasoning_content")
            if m.get(f) and PAT.search(m[f])]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rows rewritten: {changed}/{len(rows)}   occurrences handled: {len(edits)}")
    print(f"REMAINING claude/anthropic: {len(left)}  {left[:5]}")
    for sid, where, ctx in edits[:8]:
        print(f"  [{sid}] {where}: ...{ctx}...")
    (OUT / "deidentify_report.json").write_text(json.dumps(
        {"rows_rewritten": changed, "occurrences": len(edits), "remaining": left,
         "source": SOURCE}, indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
