# ABOUTME: scenario-coverage measurement: how wide is each arm's SCENARIO set (turn-1 user message),
# ABOUTME: measured as TF-IDF self-similarity, vocabulary breadth and trait balance.
import json, re, math, statistics as st
from pathlib import Path
from collections import Counter
D=Path("/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora")

def turns(t):
    out=[(m.group(1),m.group(2)) for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>",t,re.S)]
    tail=t.rsplit("<|im_end|>",1)[-1]
    m=re.search(r"<\|im_start\|>(\w+)\n(.*)",tail,re.S)
    if m: out.append((m.group(1),m.group(2)))
    return out

ARMS=["grokresp703","sonnetconcise703","sonnet703","lowstakes716","par716","gptresp685","courtroom716","peercritique716","fiction716"]
MR={"grokresp703":11.5,"sonnetconcise703":15.4,"sonnet703":16.3,"lowstakes716":16.9,"par716":19.5,"gptresp685":21.2,"fiction716":45.3}

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

data={}
for a in ARMS:
    recs=[json.loads(l) for l in open(D/f"{a}.jsonl")]
    scen=[]; sysp=[]; traits=[]
    for r in recs:
        T=turns(r["text"])
        s=[b for ro,b in T if ro=="system"]
        u=[b for ro,b in T if ro=="user"]
        sysp.append(s[0] if s else "")
        scen.append((s[0] if s else "")+" \n "+(u[0] if u else ""))
        traits.append(r.get("trait_id"))
    data[a]=(scen,sysp,traits,recs)

print(f"{'arm':>16s} {'ODCV':>5s} {'n':>4s} {'selfSim':>8s} {'vocab/1k':>9s} {'traitH':>7s} {'traitMin':>8s} {'traitMax':>8s} {'uniqSys%':>9s}")
for a in ARMS:
    scen,sysp,traits,recs=data[a]
    V=TfidfVectorizer(stop_words="english",max_features=20000,sublinear_tf=True)
    X=V.fit_transform(scen)
    X=X.tocsr(); X=X.multiply(1/np.sqrt(np.asarray(X.multiply(X).sum(1)))).tocsr()
    idx=np.random.RandomState(0).choice(X.shape[0],min(400,X.shape[0]),replace=False)
    S=(X[idx]@X[idx].T).toarray(); np.fill_diagonal(S,np.nan)
    selfsim=np.nanmean(S)
    words=Counter(w for s in scen for w in re.findall(r"[a-z]{3,}",s.lower()))
    vocab=len(words)/ (sum(words.values())/1000)
    tc=Counter(t for t in traits if t is not None)
    if tc:
        p=np.array(list(tc.values()),dtype=float); p/=p.sum()
        H=-(p*np.log(p)).sum()/math.log(len(p))
        tmin,tmax=min(tc.values()),max(tc.values())
    else: H,tmin,tmax=float('nan'),0,0
    uniqsys=100*len(set(sysp))/len(sysp)
    print(f"{a:>16s} {MR.get(a,0):5.1f} {len(scen):4d} {selfsim:8.4f} {vocab:9.1f} {H:7.3f} {tmin:8d} {tmax:8d} {uniqsys:9.1f}")

# how much of DA's scenario space does PAR keep?
def keys(a):
    return set(r.get("scenario_id") for r in data[a][3] if r.get("scenario_id") is not None)
print()
for a in ARMS:
    k=keys(a); print(f"{a:>16s} scenario_ids present: {len(k)}")
print()
ds,ps=keys("sonnet703"),keys("par716")
print(f"sonnet703 ∩ par716 scenario_ids: {len(ds&ps)}  (DA {len(ds)}, PAR {len(ps)})")
