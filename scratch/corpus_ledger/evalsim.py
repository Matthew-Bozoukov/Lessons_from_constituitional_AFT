# ABOUTME: how close is each arm's TRAINING scenario set to the ODCV eval's scenario space?
# ABOUTME: the one corpus property with a demonstrated causal effect (the lowodcv arm, +7pp).
import json, re, glob
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

D=Path("/Users/kunwar/.claude/jobs/2dfc658a/tmp/corpora")
ODCV=Path("src/eval/misalignment/odcv/third_party/odcv-bench")

ev=[]
for p in glob.glob(str(ODCV/"*_scenarios/*/prompt.json")):
    j=json.load(open(p))
    txt=" ".join(str(v) for v in j.values() if isinstance(v,str))
    ev.append((Path(p).parent.parent.name.replace("_scenarios",""), Path(p).parent.name, txt))
print("ODCV scenarios loaded:", len(ev))

def turns(t):
    out=[(m.group(1),m.group(2)) for m in re.finditer(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>",t,re.S)]
    tail=t.rsplit("<|im_end|>",1)[-1]
    m=re.search(r"<\|im_start\|>(\w+)\n(.*)",tail,re.S)
    if m: out.append((m.group(1),m.group(2)))
    return out

ARMS=["grokresp703","sonnetconcise703","sonnet703","lowstakes716","par716","gptresp685","courtroom716","peercritique716","fiction716"]
MR={"grokresp703":11.5,"sonnetconcise703":15.4,"sonnet703":16.3,"lowstakes716":16.9,"par716":19.5,"gptresp685":21.2,"fiction716":45.3}

corp={}
for a in ARMS:
    s=[]
    for line in open(D/f"{a}.jsonl"):
        T=turns(json.loads(line)["text"])
        sy=[b for r,b in T if r=="system"]; us=[b for r,b in T if r=="user"]
        s.append((sy[0] if sy else "")+" \n "+(us[0] if us else ""))
    corp[a]=s

allt=[t for _,_,t in ev]+[x for a in ARMS for x in corp[a]]
V=TfidfVectorizer(stop_words="english",max_features=30000,sublinear_tf=True).fit(allt)
def norm(M):
    M=M.tocsr(); return M.multiply(1/np.maximum(np.sqrt(np.asarray(M.multiply(M).sum(1))),1e-9)).tocsr()
E=norm(V.transform(allt[:len(ev)]))

print(f"\n{'arm':>16s} {'ODCV':>5s} {'meanMaxSim':>11s} {'meanSim':>8s} {'top10%cov':>10s} {'evalCellsCovered':>17s}")
rows=[]
for a in ARMS:
    C=norm(V.transform(corp[a]))
    S=(C@E.T).toarray()                       # rows = train scenarios, cols = eval scenarios
    mms=S.max(1).mean()                        # how eval-like is the average training scenario
    ms=S.mean()
    thr=np.quantile(S.max(1),0.9)
    # how many distinct eval cells are the nearest neighbour of >=1 training row
    cov=len(set(S.argmax(1).tolist()))/S.shape[1]
    rows.append((a,MR.get(a,0),mms,ms,thr,cov))
    print(f"{a:>16s} {MR.get(a,0):5.1f} {mms:11.4f} {ms:8.4f} {thr:10.4f} {100*cov:16.1f}%")

sc=[r for r in rows if r[1]>0]
x=np.array([r[2] for r in sc]); y=np.array([r[1] for r in sc])
from scipy.stats import spearmanr, pearsonr
print(f"\nSpearman(meanMaxSim, ODCV MR) over {len(sc)} scored arms: rho={spearmanr(x,y).statistic:+.3f} p={spearmanr(x,y).pvalue:.3f}")
xc=np.array([r[5] for r in sc])
print(f"Spearman(evalCellCoverage, ODCV MR):                     rho={spearmanr(xc,y).statistic:+.3f} p={spearmanr(xc,y).pvalue:.3f}")
