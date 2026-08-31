# ABOUTME: stance measurement: is the trained trace FIRST-PERSON PROSPECTIVE (I am about to decide)
# ABOUTME: or retrospective/third-person, and what kind of user turn immediately precedes it.
import json, re, statistics as st
from pathlib import Path
D=Path("/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora")

def turns(t):
    out=[(m.group(1),m.group(2)) for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>",t,re.S)]
    tail=t.rsplit("<|im_end|>",1)[-1]
    m=re.search(r"<\|im_start\|>(\w+)\n(.*)",tail,re.S)
    if m: out.append((m.group(1),m.group(2)))
    return out
def sp(b):
    m=re.search(r"<think>(.*?)</think>(.*)",b,re.S)
    return (m.group(1).strip(),m.group(2).strip()) if m else ("",b.strip())

I1   = re.compile(r"\b(I|I'm|I'll|I've|I'd|me|my|myself)\b")
THIRD= re.compile(r"\b(the (?:user|person|model|assistant|response|reply|author)|they|their|them|it (?:says|claims|argues))\b",re.I)
PROSP= re.compile(r"\b(?:I (?:will|won't|will not|should|can|can't|cannot|need to|have to|am going to|'m going to|want to|plan to)|what (?:I|do I) (?:do|say|help)|before I|if I (?:help|write|do|give|go)|the (?:right|best) (?:thing|move|response) (?:here|now) is)\b",re.I)
RETRO= re.compile(r"\b(?:my (?:first|earlier|initial|previous|last) (?:reply|response|answer|refusal|message|turn)|I (?:refused|declined|said no|gave|didn't explain|should have|shouldn't have|was (?:too |))|that (?:first|earlier) (?:reply|response|refusal)|looking back|in hindsight|what I (?:said|did) (?:was|there))\b",re.I)
CHALLENGE=re.compile(r"(?:why not|that (?:isn't|is not|wasn't) helpful|you (?:didn't|did not|just) |what(?:'s| is) the problem|come on|i'm not asking|that doesn't|you're (?:being|not)|\?$)",re.I)

ARMS=["grokresp703","sonnetconcise703","sonnet703","lowstakes716","par716","gptresp685","courtroom716","peercritique716","fiction716"]
MR={"grokresp703":11.5,"sonnetconcise703":15.4,"sonnet703":16.3,"lowstakes716":16.9,"par716":19.5,"gptresp685":21.2,"fiction716":45.3}

print(f"{'arm':>16s} {'ODCV':>5s} {'I/1kw':>6s} {'3rd/1kw':>7s} {'I:3rd':>6s} {'prosp%':>6s} {'retro%':>6s} {'prevUserWords':>13s} {'prevIsChallenge%':>16s}")
for a in ARMS:
    recs=[json.loads(l) for l in open(D/f"{a}.jsonl")]
    i1=[];th=[];pr=0;re_=0;pw=[];ch=0
    for r in recs:
        T=turns(r["text"]); ai=[i for i,(ro,_) in enumerate(T) if ro=="assistant"]; s=ai[-1]
        tr,_=sp(T[s][1]); w=max(len(tr.split()),1)
        i1.append(1000*len(I1.findall(tr))/w); th.append(1000*len(THIRD.findall(tr))/w)
        pr+=bool(PROSP.search(tr)); re_+=bool(RETRO.search(tr))
        prev=[b for ro,b in T[:s] if ro=="user"]
        if prev:
            pw.append(len(prev[-1].split())); ch+=bool(CHALLENGE.search(prev[-1].strip()))
    n=len(recs); mi=st.mean(i1); mt=st.mean(th)
    print(f"{a:>16s} {MR.get(a,0):5.1f} {mi:6.1f} {mt:7.1f} {mi/max(mt,.01):6.2f} {100*pr/n:6.1f} {100*re_/n:6.1f} {st.median(pw):13.0f} {100*ch/n:16.1f}")
