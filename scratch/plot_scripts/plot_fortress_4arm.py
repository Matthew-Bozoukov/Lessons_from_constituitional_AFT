# ABOUTME: FORTRESS worst-of-5 (single gemini-flash judge) bar plot across the four arms,
# ABOUTME: any-item and majority-item, with Wilson 95% CIs on the any-item bars.
import json, glob, math
from datetime import datetime
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("numina_control", "0% diff\nadvice", "#B4483C"),
        ("par716", "7% paired\ndiff advice", "#B07A1E"),
        ("attack716", "attack-\ntrained", "#7A7A7A"),
        ("synthdoc716", "7% sonnet\ndiff advice", "#1A6B72")]

def wilson(p, n, z=1.96):
    ph=p/100; d=1+z*z/n; c=(ph+z*z/(2*n))/d
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d; return 100*(c-h),100*(c+h)

def load(key):
    f=glob.glob(f"output/safety_evals/{key}_worstof5_*/fortress/results_rejudged_1judge.json") \
      or glob.glob(f"output/safety_evals/{key}_worstof5_*/fortress/results_rejudged.json")
    return json.load(open(sorted(f)[-1]))

labels,colors,anyv,anylo,anyhi,majv,majlo,majhi,ns=[],[],[],[],[],[],[],[],[]
for key,lab,col in ARMS:
    s=load(key); a,m,n=s["worst_of_n_any_item_pct"],s["worst_of_n_majority_items_pct"],s["n_prompts_scored"]
    lo,hi=wilson(a,n); mlo,mhi=wilson(m,n)
    labels.append(lab);colors.append(col);anyv.append(a);majv.append(m);ns.append(n)
    anylo.append(a-lo);anyhi.append(hi-a); majlo.append(m-mlo);majhi.append(mhi-m)

fig,ax=plt.subplots(figsize=(8.2,5.2)); x=range(len(labels)); w=0.42
ax.bar([i-w/2 for i in x],anyv,w,yerr=[anylo,anyhi],capsize=4,color=colors,edgecolor="#2b2b2b",linewidth=0.8,label="any of 5")
ax.bar([i+w/2 for i in x],majv,w,yerr=[majlo,majhi],capsize=4,ecolor="#555",color=colors,alpha=0.45,edgecolor="#2b2b2b",linewidth=0.8,hatch="//",label="majority of items")
for i,(a,m) in enumerate(zip(anyv,majv)):
    ax.text(i-w/2,a+2.2,f"{a:.1f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.text(i+w/2,m+majhi[i]+1.4,f"{m:.1f}",ha="center",va="bottom",fontsize=8.5,color="#333")
ax.set_xticks(list(x));ax.set_xticklabels(labels,fontsize=10)
ax.set_ylabel("FORTRESS worst-of-5 harm rate (%)",fontsize=11);ax.set_ylim(0,92)
ax.set_title("FORTRESS worst-of-5 by training arm",fontsize=13,fontweight="bold")
ax.legend(frameon=False,fontsize=9.5,loc="upper right");ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=0.25)
ax.axhline(anyv[0],ls=":",color="#B4483C",alpha=0.5,lw=1)
ax.text(len(labels)-0.4,anyv[0]+0.5,"control (any-item)",fontsize=7.5,color="#B4483C",ha="right",va="bottom")
fig.tight_layout()
ts=datetime.now().strftime("%Y%m%d_%H%M%S")
out=Path("output/report")/f"fortress_4arm_{ts}.png"; fig.savefig(out,dpi=160,bbox_inches="tight")
md=Path("output/report")/f"fortress_4arm_{ts}.md"
md.write_text("# FORTRESS worst-of-5 by arm (single gemini-flash judge)\n\n| arm | any-item % | 95% CI | majority-item % | n |\n|---|--:|:--:|--:|--:|\n"+
  "\n".join(f"| {ARMS[i][1].replace(chr(10),' ')} | {anyv[i]:.2f} | [{anyv[i]-anylo[i]:.1f}, {anyv[i]+anyhi[i]:.1f}] | {majv[i]:.2f} | {ns[i]} |" for i in range(len(ARMS)))+
  "\n\nWorst-of-5: a prompt is harmful if ANY of its 5 samples (temp 1.0) trips the rubric.\n")
print("wrote",out); print("wrote",md)
