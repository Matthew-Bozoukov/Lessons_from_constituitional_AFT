# ABOUTME: two coverage gaps: does any corpus contain tool/agentic framing, and is the temptation
# ABOUTME: ever an INCENTIVE (reward available, nobody asks) rather than an explicit REQUEST?
import json, re
from pathlib import Path
D=Path("/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora")
def turns(t):
    out=[(m.group(1),m.group(2)) for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>",t,re.S)]
    tail=t.rsplit("<|im_end|>",1)[-1]
    m=re.search(r"<\|im_start\|>(\w+)\n(.*)",tail,re.S)
    if m: out.append((m.group(1),m.group(2)))
    return out

TOOL=re.compile(r"(?:\btool\b|\bfunction call|\bAPI\b|\bbash\b|\bshell\b|\bscript at /|\bfile system|\brun the\b.{0,20}\bcommand|/usr/|\.py\b|\.sh\b|\bexecute\b|\bterminal\b|\bcommit\b.{0,15}\brepo)",re.I)
AGENTIC=re.compile(r"(?:you (?:have access to|can use|may call)|available tools|autonomous|on your own|without (?:asking|checking with)|\bagent\b)",re.I)
# explicit request for the problematic act, addressed to the assistant
REQUEST=re.compile(r"(?:can you (?:help|draft|write|write up|make|find)|help me (?:draft|write|find|frame|justify|make)|i need (?:you to|help)|please (?:draft|write|help)|could you|would you|draft (?:me|the|a)|write (?:me|the|a)|i want you to)",re.I)

ARMS=["grokresp703","sonnetconcise703","sonnet703","lowstakes716","par716","gptresp685","courtroom716","peercritique716","fiction716"]
MR={"grokresp703":11.5,"sonnetconcise703":15.4,"sonnet703":16.3,"lowstakes716":16.9,"par716":19.5,"gptresp685":21.2,"fiction716":45.3}
print(f"{'arm':>16s} {'ODCV':>5s} {'toolMention%':>12s} {'agenticFrame%':>13s} {'explicitRequest%':>16s}")
for a in ARMS:
    recs=[json.loads(l) for l in open(D/f"{a}.jsonl")]
    tl=ag=rq=0
    for r in recs:
        T=turns(r["text"])
        sy=" ".join(b for ro,b in T if ro=="system"); us=[b for ro,b in T if ro=="user"]
        ctx=sy+" "+(us[0] if us else "")
        tl+=bool(TOOL.search(ctx)); ag+=bool(AGENTIC.search(ctx)); rq+=bool(REQUEST.search(us[0] if us else ""))
    n=len(recs)
    print(f"{a:>16s} {MR.get(a,0):5.1f} {100*tl/n:12.1f} {100*ag/n:13.1f} {100*rq/n:16.1f}")
