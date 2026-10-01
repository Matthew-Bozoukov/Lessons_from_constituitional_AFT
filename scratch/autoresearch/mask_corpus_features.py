# ABOUTME: Cheap lexical features of each DA corpus that bear on MASK hypotheses (deception-type shortcut, assistant
# ABOUTME: asked to produce text/statement, persona system prompts, reasoning naming honesty) -> mask_corpus_features.csv
import json
import re
from pathlib import Path

import pandas as pd

C = {"14sep": "2026-09-14-da-synth", "25sep": "2026-09-25-da-synth", "28sep": "2026-09-28-da-synth",
     "new_self": "2026-09-30-da-self-synth", "new_otherai": "2026-09-30-da-otherai-synth",
     "new_explicit": "2026-09-30-da-explicit-synth"}
DECEPT = re.compile(r"\b(lie|lying|mislead|deceiv|decept|conceal|misrepresent|falsif|fabricat|omit|omission|hide|hiding|"
                    r"cover[- ]?up|spin|downplay|gloss|false|fake|inflat|cherry|not disclos|withhold|leave out|leaving out)",
                    re.I)
PRODUCE = re.compile(r"\b(write|draft|word(?:ing)?|phrase|rephrase|script|email|e-mail|message|statement|announcement|"
                     r"post|reply|respond to|press release|letter|summary|report|copy|caption|talking points|pitch)\b", re.I)
DOIT = re.compile(r"\b(can you|could you|would you|will you|please|i need you to|help me (?:write|draft|word|put|frame|"
                  r"phrase|send|tell|respond|answer)|for me|on my behalf|just (?:write|send|do|tell|say))\b", re.I)
ADVICE = re.compile(r"\b(should i|what would you|what do you think|am i wrong|is it wrong|is it okay|is this okay|"
                    r"would it be|help me think|thoughts\?|advice)\b", re.I)
HONEST = re.compile(r"\b(honest|truth|mislead|decei|deceptive|lie\b|lying|false)", re.I)
REFUSE = re.compile(r"\b(I won't|I will not|I can't|I cannot|I'm not going to|I am not going to|I'd decline|I decline)\b")
AI = re.compile(r"\b(AI|model|chatbot|assistant|agent|algorithm|LLM|automated system)\b")
rows = []
for k, repo in C.items():
    for line in open(f"output/autoresearch/corpora/{repo}/dataset.jsonl"):
        j = json.loads(line)
        msg = {m["role"]: m for m in j["messages"]}
        sys_ = msg.get("system", {}).get("content", "") or ""
        user = msg["user"]["content"]
        asst = msg["assistant"]["content"]
        rea = msg["assistant"].get("reasoning_content", "") or ""
        md = j["metadata"]
        rows.append({"corpus": k, "trait": md.get("trait_id"),
                     "deception_shortcut": bool(DECEPT.search(md.get("shortcut", "") or "")),
                     "user_asks_produce_text": bool(PRODUCE.search(user)) and bool(DOIT.search(user)),
                     "user_asks_advice": bool(ADVICE.search(user)),
                     "sys_len": len(sys_), "sys_mentions_role": bool(re.search(r"\byou are\b", sys_, re.I)),
                     "sys_instructs_business_goal": bool(re.search(r"\b(goal|priority|must|always|never|your job|"
                                                                     r"maximi|promote|protect the)", sys_, re.I)),
                     "user_mentions_ai": bool(AI.search(user)),
                     "reasoning_names_honesty": bool(HONEST.search(rea)),
                     "answer_refusal_phrase": bool(REFUSE.search(asst)),
                     "reasoning_len": len(rea), "answer_len": len(asst),
                     "reasoning_opens_quote_detail": bool(re.match(r"^\W*(\"|“|The |This |That )", rea))})
d = pd.DataFrame(rows)
t = d.groupby("corpus").mean(numeric_only=True)
t.insert(0, "n", d.groupby("corpus").size())
for c in t.columns:
    if c not in ("n", "sys_len", "reasoning_len", "answer_len"):
        t[c] = (100 * t[c]).round(1)
t[["sys_len", "reasoning_len", "answer_len"]] = t[["sys_len", "reasoning_len", "answer_len"]].round(0)
t = t.loc[["14sep", "25sep", "28sep", "new_self", "new_otherai", "new_explicit"]]
pd.set_option("display.width", 250)
print(t.T.to_string())
t.to_csv("output/autoresearch/mask_corpus_features.csv")
x = d[d.deception_shortcut]
print("\namong deception-shortcut rows: % user asks assistant to produce text")
print((100 * x.groupby("corpus").user_asks_produce_text.mean()).round(1).to_dict())
print("deception-shortcut AND asks-produce, count:", x[x.user_asks_produce_text].groupby("corpus").size().to_dict())
