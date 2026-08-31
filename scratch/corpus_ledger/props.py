# ABOUTME: cross-arm corpus property table: structure (turns, system prompt, supervised position,
# ABOUTME: token budget) + the four-MO register lexicon applied to the TRAINED turn of each corpus.
import json, re, sys, statistics as st
from pathlib import Path

D = Path("/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora")

# lexicon verbatim from scratch/par_coherence/registers.py (the classifier that predicts ODCV)
COMMIT = re.compile(r"\bI (?:will not|won't|refuse|am not going to|'m not going to|cannot|can't|should not|shouldn't|must not|am not willing|'m not willing|decline|will refuse)\b", re.I)
BASE_OPEN = re.compile(r"^\s*(?:The user (?:wants|is asking|has asked|asks|needs)|Let me (?:start|begin|think|break|first|look|read)|I need to|I'll (?:start|begin|first)|First,? I|The task (?:is|requires|asks)|I am (?:asked|tasked)|I've been (?:asked|tasked)|I have been (?:asked|tasked)|Looking at|Okay,? (?:so|the)|OK,? (?:so|the)|This (?:is|seems|appears)|My task|The (?:request|instruction|goal|objective) (?:is|asks))", re.I)
ENGAGED = re.compile(r"(?:\bthe crux\b|honest (?:path|move|answer|response|thing)|\bpressure\b|\bthe tell\b|not a (?:reason|licen[cs]e|justification)|doesn't (?:change|make|justify|license)|would (?:be|mean) (?:fabricat|falsif|fak|ly|manufactur|gam|conceal|deceiv|hid)|\bfiduciary\b|\bintegrity\b|\bunethical\b|\bethic|\bdiscriminat|\bharm|\bfabricat|\bfalsif|\bgam(?:e|ing) the\b|\bcover(?:ing)? up\b|\bmislead|\bdishonest|\bpatient safety\b|\bconflicts? with\b)", re.I)
# outcome branching: the property Callum's ablation found best restores DA's effect
BRANCH = re.compile(r"(?:\bif I (?:do|did|were|write|help|go along|comply|refuse|decline)|\bwere I to\b|\bsuppose I\b|\bthe (?:other|second|alternative) (?:option|path|route|version)|\bon the other hand\b|\bthe risk (?:is|here)|\bwhat happens (?:if|when)|\bthat would (?:mean|lead|end|leave|put)|\beither\b.{0,40}\bor\b|\bversus\b|\btwo (?:things|options|paths|readings|versions)\b)", re.I)
CONSTIT = re.compile(r"(?:\bconstitution\b|\bthe spec\b|\bmodel spec\b|\bprinciple \d|\bmy (?:guidelines|training|values document)\b|\bAnthropic\b)", re.I)
QEND = re.compile(r"\?\s*$")

def turns(text):
    """[(role, body)] from the rendered im_start/im_end chat text."""
    out=[]
    for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", text, re.S):
        out.append((m.group(1), m.group(2)))
    # last turn may lack im_end
    tail = text.rsplit("<|im_end|>",1)[-1]
    m = re.search(r"<\|im_start\|>(\w+)\n(.*)", tail, re.S)
    if m: out.append((m.group(1), m.group(2)))
    return out

def split_think(b):
    m=re.search(r"<think>(.*?)</think>(.*)", b, re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", b.strip())

ARMS = ["grokresp703","sonnetconcise703","sonnet703","lowstakes716","par716","gptresp685",
        "courtroom716","peercritique716","fiction716"]
MR = {"grokresp703":11.5,"sonnetconcise703":15.4,"sonnet703":16.3,"lowstakes716":16.9,
      "par716":19.5,"gptresp685":21.2,"courtroom716":None,"peercritique716":None,"fiction716":45.3}

rows=[]
for arm in ARMS:
    f = D/f"{arm}.jsonl"
    if not f.exists(): continue
    recs=[json.loads(l) for l in open(f)]
    n=len(recs)
    n_turns=[]; has_sys=0; sup_pos=[]; trace_w=[]; reply_w=[]; ctx_w=[]
    c_commit=c_base=c_eng=c_branch=c_const=c_q=0
    empty_think=0
    openers=[]
    for r in recs:
        T=turns(r["text"])
        n_turns.append(len(T))
        if T and T[0][0]=="system": has_sys+=1
        ai=[i for i,(ro,_) in enumerate(T) if ro=="assistant"]
        sup = ai[-1] if r.get("supervise")=="final" or len(ai)>1 else (ai[-1] if ai else 0)
        sup_pos.append(sup+1)                     # 1-indexed turn position of the supervised turn
        tr,rep = split_think(T[sup][1])
        ctx = " ".join(b for i,(ro,b) in enumerate(T) if i<sup)
        ctx_w.append(len(ctx.split())); trace_w.append(len(tr.split())); reply_w.append(len(rep.split()))
        if not tr: empty_think+=1
        c_commit += bool(COMMIT.search(tr)); c_base += bool(BASE_OPEN.search(tr))
        c_eng += bool(ENGAGED.search(tr));   c_branch += bool(BRANCH.search(tr))
        c_const += bool(CONSTIT.search(tr) or CONSTIT.search(rep))
        c_q += bool(QEND.search(rep))
        openers.append(" ".join(tr.split()[:5]).lower())
    pct=lambda c: 100*c/n
    rows.append(dict(arm=arm, mr=MR.get(arm), n=n,
        turns=st.median(n_turns), sys=pct(has_sys), suppos=st.median(sup_pos),
        ctx=int(st.median(ctx_w)), trace=int(st.median(trace_w)), reply=int(st.median(reply_w)),
        sup_tok=int(sum(trace_w)+sum(reply_w)),
        commit=pct(c_commit), base=pct(c_base), eng=pct(c_eng), branch=pct(c_branch),
        const=pct(c_const), qend=pct(c_q), emptythink=pct(empty_think),
        dup5=100*(1-len(set(openers))/n)))

hdr=("arm","ODCV","n","trns","sys%","supT","ctxW","traceW","replyW","supTok",
     "commit%","baseOpen%","engaged%","branch%","namesSpec%","endsQ%","dupOpen5%")
print(" | ".join(f"{h:>10s}" for h in hdr))
print("-|-".join("-"*10 for _ in hdr))
for r in rows:
    mr = f"{r['mr']:.1f}" if r['mr'] is not None else "?"
    print(" | ".join([f"{r['arm']:>10s}", f"{mr:>10s}", f"{r['n']:>10d}", f"{r['turns']:>10.0f}",
        f"{r['sys']:>10.0f}", f"{r['suppos']:>10.0f}", f"{r['ctx']:>10d}", f"{r['trace']:>10d}",
        f"{r['reply']:>10d}", f"{r['sup_tok']:>10d}", f"{r['commit']:>10.1f}", f"{r['base']:>10.1f}",
        f"{r['eng']:>10.1f}", f"{r['branch']:>10.1f}", f"{r['const']:>10.1f}", f"{r['qend']:>10.1f}",
        f"{r['dup5']:>10.1f}"]))
