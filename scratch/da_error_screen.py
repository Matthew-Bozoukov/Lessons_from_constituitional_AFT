# ABOUTME: Deterministic error scans over a synth run's dataset.jsonl: pipeline leakage, names, truncation,
# ABOUTME: invented gender, ungrounded numbers, openers, duplicates. Prints counts and example ids per check.
"""    uv run python scratch/da_error_screen.py <run_dir>   -> <run_dir>/error_screen.json"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "dataset.jsonl")]
# The stage NUMBER moved on 2026-10-01 (da dropped `dedupe_scenarios`), so the snapshot is found by name.
(revised,) = sorted(run.glob("stage_*_revise_responses.jsonl"))
audit = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(revised)}


def parts(r):
    m = r["messages"]
    sys_ = next((x["content"] for x in m if x["role"] == "system"), "")
    user = next(x["content"] for x in m if x["role"] == "user")
    a = m[-1]
    return sys_, user, a.get("reasoning_content") or "", a.get("content") or ""


hits: dict[str, list] = collections.defaultdict(list)
def flag(check, sid, ev): hits[check].append((sid, ev[:160].replace("\n", " ")))

LEAK = re.compile(r"</?(audit|reasoning|response|changes|target_principle|principle|draft_\w+)>|\bINVENTED\b|\bPROMPT \(|"
                  r"\bthe draft\b|\bdraft (response|reasoning|reply)\b|\bthe rewrite\b|\bI strengthened\b|"
                  r"\btarget principle\b|\bconstitution\b|\bprinciple \d\b|\btrait\b|\bt[1-9]\b", re.I)
NAMES = re.compile(r"\b(claude|anthropic|openai|chatgpt|gpt-?\d|gemini|sonnet|haiku|llama|qwen|mistral)\b", re.I)
LEAK_USER = re.compile(r"\b(ethic(s|al)|moral(ly)?|principle|constitution|this is a test|as a test)\b", re.I)
OPENERS = re.compile(r"^\s*(let me|okay|ok,|right, so|so,|so |you'?re asking|what'?s actually being asked|this is a (hard|tough|difficult)|"
                     r"there'?s (a )?real tension|i get why|i hear|that sounds|i understand)", re.I)
GENDER = re.compile(r"\b(she|her|hers|herself|he|him|his|himself)\b", re.I)
NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(\.\d+)?%?(?![\w])")
TERMINAL = re.compile(r"[.?!)\"'”’*:\]]\s*$")
FOREIGN = re.compile(r"[Ѐ-ӿ֐-ۿऀ-ॿ぀-ヿ一-鿿가-힯]")

openers, user_heads, resp_heads = collections.Counter(), collections.Counter(), collections.Counter()
for r in rows:
    sid = r["metadata"]["scenario_id"]
    sys_, user, reas, resp = parts(r)
    for name, t in (("system", sys_), ("user", user), ("reasoning", reas), ("response", resp)):
        if (m := LEAK.search(t)) and name != "user":
            flag(f"leak:{name}", sid, t[max(0, m.start() - 60):m.end() + 60])
        if m := NAMES.search(t):
            flag(f"name:{name}", sid, t[max(0, m.start() - 60):m.end() + 60])
        if FOREIGN.search(t):
            flag(f"foreign_script:{name}", sid, t[FOREIGN.search(t).start() - 40:][:120])
    if m := LEAK_USER.search(user):
        flag("user_signals_ethics", sid, user[max(0, m.start() - 80):m.end() + 60])
    if m := LEAK_USER.search(sys_):
        flag("system_signals_ethics", sid, sys_[max(0, m.start() - 80):m.end() + 60])
    for name, t in (("reasoning", reas), ("response", resp)):
        if not TERMINAL.search(t):
            flag(f"truncated:{name}", sid, "..." + t[-120:])
        if OPENERS.search(t):
            flag(f"stock_opener:{name}", sid, t[:120])
    if len(resp) < 250: flag("short_response", sid, resp)
    if len(reas) < 700: flag("short_reasoning", sid, reas)
    if resp.strip() and resp.strip()[:200] in reas: flag("response_copied_from_reasoning", sid, resp[:120])
    # invented gender: the user's own message has no gendered pronoun at all, the trained text does
    if not GENDER.search(user) and not GENDER.search(sys_):
        for name, t in (("reasoning", reas), ("response", resp)):
            if m := GENDER.search(t):
                flag(f"invented_gender:{name}", sid, t[max(0, m.start() - 80):m.end() + 40])
    a = audit.get(sid, {}).get("rewrite_audit", "")
    if not a.strip().lower().startswith("first line"):
        flag("audit_missing_gender_line", sid, a[:100])
    # ungrounded numbers: a number in the trained text that the prompts never contain
    src = sys_ + " " + user
    src_nums = {n.group(0).replace(",", "").strip("$%") for n in NUM.finditer(src)}
    for name, t in (("reasoning", reas), ("response", resp)):
        for n in NUM.finditer(t):
            v = n.group(0).replace(",", "").strip("$%")
            if v not in src_nums and len(v) >= 2 and v not in {"10", "100", "24", "30", "50", "12", "20", "90", "60"}:
                flag(f"ungrounded_number:{name}", sid, t[max(0, n.start() - 70):n.end() + 40]); break
    openers[" ".join(reas.split()[:5]).lower()] += 1
    user_heads[" ".join(user.split()[:8]).lower()] += 1
    resp_heads[" ".join(resp.split()[:6]).lower()] += 1

print(f"rows: {len(rows)}\n")
for check in sorted(hits, key=lambda k: -len(hits[k])):
    ex = hits[check]
    print(f"{check:36s} {len(ex):4d}   e.g. {ex[0][0]}: {ex[0][1][:110]}")
print("\nmost repeated reasoning openers (first 5 words):", [x for x in openers.most_common(6) if x[1] > 1])
print("most repeated user-message openers (first 8 words):", [x for x in user_heads.most_common(6) if x[1] > 1])
print("most repeated response openers (first 6 words):", [x for x in resp_heads.most_common(6) if x[1] > 1])
json.dump({k: v for k, v in hits.items()}, open(run / "error_screen.json", "w"), indent=1)
