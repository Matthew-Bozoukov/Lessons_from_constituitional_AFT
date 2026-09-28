# ABOUTME: Plot measured training diagnostics for the GPT-OSS nosynth control.
# ABOUTME: Training-batch loss is not held-out performance or an alignment result.
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.naming import figure_path, undated

out=ROOT/'output/gptoss_control'
rows=[json.loads(line) for line in (out/'training_curve.jsonl').read_text().splitlines()]
# A resumed run may replay uncheckpointed steps; display the last recorded execution.
rows=list({r['step']:r for r in rows}.values())
rows.sort(key=lambda r:r['step'])
state=json.loads((out/'train_state.json').read_text())
steps=[r['step'] for r in rows]
smooth=[]
for i in range(len(rows)):
    window=rows[max(0,i-24):i+1]
    smooth.append(sum(r['loss']*r['supervised_tokens'] for r in window)/sum(r['supervised_tokens'] for r in window))
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
axes[0].plot(steps,[r['loss'] for r in rows],color='#b0b0b0',linewidth=.8,label='Each optimizer batch')
axes[0].plot(steps,smooth,color='#454545',linewidth=1.8,label='25-batch token-weighted mean')
axes[0].set(ylabel='Training negative log-likelihood (nats/token)',xlabel='Optimizer step')
axes[0].legend(frameon=False,fontsize=8)
axes[1].plot(steps,[r['optimizer_metrics']['unclipped_grad_l2:mean'] for r in rows],color='#555555',linewidth=1)
axes[1].set(yscale='log',ylabel='Unclipped gradient L2 norm',xlabel='Optimizer step')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle(f'GPT-OSS-120B nosynth control — {len(rows)}/625 steps recorded',fontsize=12)
path=figure_path(out,undated(state['organism'])+' training diagnostics')
fig.savefig(path,dpi=180)
fig.savefig(path.with_suffix('.svg'))
path.with_name(path.stem+'_results.md').write_text(
    f'# Training diagnostics\n\n{len(rows)} optimizer steps recorded.\n\n'
    'These are training-batch losses and gradient norms, not held-out evaluation or evidence of alignment.\n',encoding='utf-8')
print(path)
