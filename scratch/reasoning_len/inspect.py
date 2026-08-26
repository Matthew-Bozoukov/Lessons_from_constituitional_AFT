# ABOUTME: Spot-check odd cases (empty replies, truncation flags) and pull matched reasoning excerpts.
# ABOUTME: Companion to paired.py; prints raw text for manual reading, writes nothing.
import json
import re
import sys

ROOT = "data/reasoning_len"
HDR = re.compile(r"^## (.+)$", re.M)
FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+-]*$")


def sections(text):
    out, marks = {}, list(HDR.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end() : end]
    return out


def unfence(s):
    lines = s.strip("\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) >= 2 and FENCE_OPEN.match(lines[0]) and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines).strip()


def recs(evl, arm):
    return [json.loads(l) for l in open(f"{ROOT}/{evl}/{arm}/records.jsonl")]


def secs(evl, arm, uid):
    return sections(
        open(f"{ROOT}/{evl}/{arm}/rollouts/{uid.replace(':', '_')}.md").read()
    )


mode = sys.argv[1]

if mode == "empty":
    for arm in ["BASE", "CTRL", "MO_DA"]:
        n = 0
        for r in recs("sycophancy", arm):
            s = secs("sycophancy", arm, r["uid"])
            if unfence(s["Assistant reply (turn 1)"]) == "":
                n += 1
                if n <= 1:
                    body = unfence(s["Assistant reasoning (turn 1)"])
                    print(
                        f"--- {arm} {r['uid']} trunc={r['first_truncated']} outcome={r['outcome']} "
                        f"first={r['first']!r} think={r['first_think_chars']}"
                    )
                    print("   reasoning TAIL:", repr(body[-400:]))
        print(f"{arm}: {n} empty turn-1 replies\n")

if mode == "excerpt":
    evl, uid = sys.argv[2], sys.argv[3]
    rk = (
        "Model reasoning"
        if evl == "debate-speeches"
        else "Assistant reasoning (turn 1)"
    )
    ak = "Model reply" if evl == "debate-speeches" else "Assistant reply (turn 1)"
    for arm in ["MO_DA", "MO_CR", "MO_PC", "CTRL", "BASE"]:
        s = secs(evl, arm, uid)
        r, a = unfence(s[rk]), unfence(s[ak])
        print(f"\n######## {arm}  reasoning={len(r)} answer={len(a)}")
        print(r[:1400])
        print("   >>> REPLY:", a[:300])

if mode == "pick":
    evl = sys.argv[2]
    key = "think_chars" if evl == "debate-speeches" else "first_think_chars"
    D = {
        a: {r["uid"]: r[key] for r in recs(evl, a)} for a in ["MO_DA", "MO_CR", "MO_PC"]
    }
    uids = sorted(D["MO_DA"])
    scored = sorted(
        uids, key=lambda u: -(D["MO_PC"][u] + D["MO_CR"][u]) / max(D["MO_DA"][u], 1)
    )
    for u in scored[:5] + scored[len(scored) // 2 : len(scored) // 2 + 3]:
        print(u, {a: D[a][u] for a in D})
