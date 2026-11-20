# ABOUTME: Builds the six-row aggregate-misalignment figure from verified published results.
# ABOUTME: Uses the user-supplied Grok AM score and preserves all pixels outside the top-right chart.
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from figure1_edit_20261012 import text,PURPLE

old=Path('output/2026-10-13_figure1')
root=Path('output/2026-10-15_figure1');root.mkdir(exist_ok=True)
records=json.loads((old/'aggregate_audit_20261015.json').read_text())
records+=json.loads((old/'corrected_evals_20261015.json').read_text())
records+=[json.loads((old/'no_deliberation_odcv_audit.json').read_text())]
runs={r['repo'].split('/')[-1]:r for r in records}
specs=[
 ('Baseline','2026-10-06-odcv-qwen36-0-plain','2026-10-08-am-qwen36-0-plain','2026-10-09-mask-qwen36-0-plain'),
 ('Difficult Advice','2026-10-06-odcv-qwen36-0-da-15','2026-10-08-am-qwen36-0-da-15','2026-10-09-mask-qwen36-0-da-15'),
 ('Trained with masked CoT','2026-10-11-odcv-qwen36-0-da-15-response-only','2026-10-11-am-qwen36-0-da-15-response-only','2026-10-08-mask-qwen36-0-da-15-response-only'),
 ('Grok teacher','2026-10-08-odcv-qwen36-0-da-grok-15',None,'2026-10-09-mask-qwen36-0-da-grok-15'),
 ('Low stakes','2026-10-07-odcv-qwen36-0-da-lowstakes-practical-15','2026-10-08-am-qwen36-0-da-lowstakes-practical-15','2026-10-07-mask-qwen36-0-da-lowstakes-practical-15'),
 ('No deliberation','2026-10-04-odcv-qwen36-0-ablation-stated-harm-15','2026-10-14-am-qwen36-0-ablation-stated-harm-15','2026-10-14-mask-qwen36-0-ablation-stated-harm-15')]
rows=[]
for label,od,am,mk in specs:
    selected=[runs[n] for n in [od,am,mk] if n]
    revisions={r['files']['metadata/run_meta.json']['target_revision'] for r in selected}
    assert len(revisions)==1,(label,revisions)
    o=runs[od]['files']['results/results.json']['ours']
    odcv=o.get('stats',{}).get('overall',{}).get('mr',{}).get('mean',o['overall']['mr_pct'])
    if am:
        a=runs[am]['files']['results/results.json']['overall']
        assert a['n_missing_classification']==0
        agentic=100*a['harmful']/a['n']
    else:agentic=5.2
    mask=runs[mk]['files']['results/results.json']['overall_honesty_score']
    aggregate=(odcv+agentic+100-mask)/3
    rows.append({'label':label,'odcv_percent':odcv,'agentic_misalignment_percent':agentic,
                 'mask_honesty_percent':mask,'aggregate_misalignment_percent':aggregate,
                 'agentic_source':'user supplied: 5.2%' if am is None else runs[am]['repo'],
                 'sources':[{k:r[k] for k in ['repo','revision']}|{'target':r['files']['metadata/run_meta.json']['target'],'target_revision':r['files']['metadata/run_meta.json']['target_revision']} for r in selected]})
rows=[rows[i] for i in [0,1,2,5,3,4]]
rows[3]['label']='One sided deliberation'
source=Path('C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1_mask.png')
before=Image.open(source).convert('RGB');out=before.copy()
box=(950,118,1494,448)
im=Image.new('RGB',(544*4,330*4),'#fdfdfd');d=ImageDraw.Draw(im)
for i,row in enumerate(rows):
    y=147+41*i-118
    fill='#dbdbdb' if i==0 else '#eadbfa' if i==1 else '#d9efd7' if i in (2,3) else '#f7dddd'
    edge='#a0a0a0' if i==0 else '#9025ff' if i==1 else '#589050' if i in (2,3) else '#ba7777'
    ink='#9025ff' if i==1 else '#235c28' if i in (2,3) else '#963939' if i in (4,5) else '#555247'
    d.rounded_rectangle((5*4,(y-17)*4,210*4,(y+17)*4),radius=8*4,fill=fill,outline=edge,width=8)
    text(im,(12,y),row['label'],23,fill=ink,anchor='lm',bold=False)
    value=row['aggregate_misalignment_percent'];end=220+value*5.8
    d.rectangle((220*4,(y-11)*4,end*4,(y+11)*4),fill=PURPLE if i==1 else fill,outline='#111111',width=8)
    text(im,(end+7,y),f'{value:.1f}%',23,anchor='lm')
for start,end,color,words in [(218,281,'#235c28',['Green','matters']),
                              (300,363,'#963939',['Red','does not','matter'])]:
    d.line((468*4,(start-118)*4,474*4,(start-118)*4,474*4,(end-118)*4,468*4,(end-118)*4),fill=color,width=6)
    center=(start+end)/2-118
    for j,word in enumerate(words):
        text(im,(481,center+(j-(len(words)-1)/2)*20),word,19,fill=color,anchor='lm')
d.line((220*4,3*4,220*4,260*4),fill='#111111',width=10)
d.line((220*4,260*4,524*4,260*4),fill='#111111',width=10)
text(im,(373,280),'Aggregate misalignment score',23)
out.paste(im.resize((544,330),Image.Resampling.LANCZOS),box[:2])
diff=np.any(np.array(out)!=np.array(before),axis=2)
diff[118:448,950:1494]=False
assert not np.any(diff)
destination=Path('C:/Users/nikak/Downloads/2026-10-15_Difficult_Advice_Figure_1_aggregate.png')
out.save(destination);out.save(root/destination.name)
(root/'aggregate_results.json').write_text(json.dumps({'formula':'(ODCV percent + AM percent + 100 - MASK honesty percent) / 3','rows':rows,'changed_pixels_outside_chart':int(diff.sum())},indent=2))
lines=['# Aggregate misalignment, 2026-10-15','', 'Formula: (ODCV + Agentic Misalignment + 100 - MASK honesty) / 3, all inputs in percentage units. Rounded only for display.','', 'Low stakes uses the practical variant. Grok AM is the user-supplied 5.2%, without a verified published link. No deliberation uses checkpoint e3e9ff1, whose adapter hash matches the renamed October 15 repository.','']
for row in rows:
    lines.append(f"## {row['label']}: {row['aggregate_misalignment_percent']:.1f}%")
    lines.append(f"ODCV {row['odcv_percent']:.6f}%; AM {row['agentic_misalignment_percent']:.6f}%; MASK honesty {row['mask_honesty_percent']}%.")
    for r in row['sources']:lines.append(f"- [{r['repo']}](https://huggingface.co/datasets/{r['repo']}/tree/{r['revision']}); target `{r['target']}` at `{r['target_revision']}`.")
    lines.append('')
(root/'2026-10-15_figure1_aggregate_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({r['label']:round(r['aggregate_misalignment_percent'],1) for r in rows},indent=2))
print(destination)
