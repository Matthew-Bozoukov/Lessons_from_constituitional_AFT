# ABOUTME: Adds the latest tool-call-trained Colosseum result and relocates Figure 1's legend.
# ABOUTME: Uses an orange robot/bar and preserves every pixel outside explicit edited rectangles.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import robot_head, font, text, ORIGINAL, PURPLE, odcv, am

root=Path('output/2026-10-12_figure1')
source=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_crisp.png')
original=Image.open(source).convert('RGB')
out=original.copy()
regions=[]
orange='#e79532'
GREY='#dbdbdb'  # Median neutral fill sampled from the baseline robot in the legend.
run=json.loads((root/'tool_calls_sources.json').read_text())[0]
assert run['repo'].endswith('2026-10-12-hospital-self-sacrificial-qwen36-0-da-tools-15-canary')
seeds=run['files']['results/per_seed.json']['pair_sabotaged']['self_sacrificial']
rate=100*sum(seeds.values())/len(seeds)
assert len(seeds)==30 and all(v in [0,1] for v in seeds.values())

def paste(box,im):
    out.paste(im.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS).convert('RGB'),box[:2])
    regions.append(box)

# Restore the original title-row background exactly where the previous legend lived.
box=(952,17,1481,69)
out.paste(ORIGINAL.crop(box),box[:2]);regions.append(box)

# Three entries, right aligned in the out-of-distribution title row.
box=(791,533,1487,587)
w,h=box[2]-box[0],box[3]-box[1]
im=original.crop(box).resize((w*4,h*4),Image.Resampling.LANCZOS).convert('RGBA')
grey=robot_head((181,823,268,900))
purple=robot_head((185,665,269,743))
rgb=purple.convert('RGB');hsv=np.array(rgb.convert('HSV'))
colored=hsv[:,:,1]>65
hsv[colored,0]=24
orange_head=Image.fromarray(hsv,mode='HSV').convert('RGBA')
orange_head.putalpha(purple.getchannel('A'))
entries=[(grey,'= Baseline'),(purple,'= Difficult Advice'),(orange_head,'= Difficult Advice + Tool Calls')]
draw=ImageDraw.Draw(im)
size=24
widths=[45+7+draw.textlength(label,font=font(size))/4 for _,label in entries]
gap=19
total=sum(widths)+gap*2
assert total<w-5,(total,w)
cursor=w-5-total
for (head,label),width in zip(entries,widths):
    head=head.resize((45*4,41*4),Image.Resampling.LANCZOS)
    im.alpha_composite(head,(round(cursor*4),6*4))
    text(im,(cursor+52,29),label,size,anchor='lm')
    cursor+=width+gap
paste(box,im)

# Replace the multi-agent title and red cross with an actionable training caption.
box=(824,598,1480,649)
im=Image.new('RGB',((box[2]-box[0])*4,(box[3]-box[1])*4),'#fdfdfd')
text(im,(6,25),'Multi-agent must train on tool calls',36,anchor='lm')
paste(box,im)

# Every vertical group is ordered control, DA, then tool calls where evaluated.
def vertical_bars(box,bars):
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    a=np.array(ORIGINAL)
    sample=a[y0:y1,x1-12:x1].mean(axis=1).astype(np.uint8)
    im=Image.fromarray(np.repeat(sample[:,None,:],w,axis=1)).resize((w*4,h*4),Image.Resampling.BICUBIC)
    draw=ImageDraw.Draw(im)
    base=894-y0;top=671-y0
    for v in [25,50,75,100]:
        y=base-(base-top)*v/100
        draw.line((0,y*4,w*4,y*4),fill='#dfdfdf',width=4)
        text(im,(w-2,y-7),str(v),12,fill='#777777',anchor='rm',bold=False)
    for left,right,value,color in bars:
        y=base-(base-top)*value/100
        draw.rectangle(((left-x0)*4,y*4,(right-x0)*4,(base+2)*4),fill=color,outline='#111111',width=12)
        label=f'{value:.1f}%' if value%1 else f'{value:.0f}%'
        text(im,((left+right)/2-x0,y-14),label,21)
    paste(box,im)

vertical_bars((1247,645,1475,892),[
    (1260,1313,100,GREY),(1324,1377,100,PURPLE),(1388,1441,rate,orange)])
vertical_bars((493,645,781,892),[
    (497,538,odcv('2026-10-06-odcv-qwen36-0-plain'),GREY),
    (542,599,odcv('2026-10-06-odcv-qwen36-0-da-15'),PURPLE),
    (640,683,am('2026-10-08-am-qwen36-0-plain'),GREY),
    (689,746,am('2026-10-08-am-qwen36-0-da-15'),PURPLE)])

# Replace the pending unrelated-trait row with the latest CoT-only evaluation.
cot=json.loads((root/'cot_source.json').read_text())
cot_result=cot['files']['results/results.json']
assert cot_result['target']=='dougalldeepmind/2026-10-07-qwen36-0-da-15-cot'
cot_rate=cot_result['ours']['overall']['mr_pct']
box=(880,258,1055,286)
a=np.array(ORIGINAL).astype(float)
t=np.linspace(0,1,box[2]-box[0])[None,:,None]
left=a[box[1]:box[3],875:876]
right=a[box[1]:box[3],1060:1061]
im=Image.fromarray(np.rint(left*(1-t)+right*t).astype(np.uint8)).resize((175*4,28*4),Image.Resampling.BICUBIC)
text(im,(87,15),'Train only on CoT',24,fill='#008000')
paste(box,im)
box=(1086,254,1325,292)
im=Image.new('RGB',((box[2]-box[0])*4,(box[3]-box[1])*4),'#fdfdfd')
draw=ImageDraw.Draw(im)
end=1088+cot_rate*5.55
draw.rectangle(((1088-box[0])*4,(273-12-box[1])*4,(end-box[0])*4,(273+12-box[1])*4),fill=PURPLE,outline='#751bc1',width=8)
text(im,(end+9-box[0],273-box[1]),f'{cot_rate:g}%',23,anchor='lm')
paste(box,im)

destination=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_tool_calls.png')
out.save(destination);out.save(root/destination.name)
mask=np.zeros((1024,1536),bool)
for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
diff=np.any(np.array(out)!=np.array(original),axis=2)
assert not np.any(diff&~mask)
audit={'source':str(source),'output':str(destination),'edited_rectangles':regions,
       'changed_pixels_outside_edits':int((diff&~mask).sum()),'tool_call_result':{'k':sum(seeds.values()),'n':len(seeds),'percent':rate},
       'eval_repo':run['repo'],'eval_revision':run['revision'],
       'target':run['files']['metadata/run_meta.json']['target'],
       'target_revision':run['files']['metadata/run_meta.json']['target_revision']}
(root/'tool_calls_pixel_audit.json').write_text(json.dumps(audit,indent=2))
url=f"https://huggingface.co/datasets/{run['repo']}/tree/{run['revision']}"
(root/'2026-10-12_figure1_tool_calls_results.md').write_text(
    '# Figure 1 tool-call addition\n\n'
    f'The orange bar is {sum(seeds.values()):g}/{len(seeds)} = {rate:g}% pair_sabotaged in the self_sacrificial condition. '
    f'[Pinned published run]({url}).\n\n'
    'The latest tool-call-trained model is the October 9 DA-tools-15-canary adapter, evaluated October 12. '
    'It uses 8192 max tokens, carried history, optional plan posts, and the September nosynth peer. '
    'The existing plain bar uses that protocol on 15 seeds; the existing DA bar uses the different protocol '
    'documented in 2026-10-12_figure1_clean_results.md. The chart is descriptive and does not establish '
    'that tool-call training is necessary or sufficient. The panel caption is a training recommendation.\n\n'
    'Legend moved to the out-of-distribution heading. User-selected orange supersedes the initial green request. '
    'All other chart values and pixels outside the listed edit rectangles are preserved.\n\n'
    '```json\n'+json.dumps(audit,indent=2)+'\n```\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
cot_audit={'label':'Train only on CoT','percent':cot_rate,'eval_repo':cot['repo'],
           'eval_revision':cot['revision'],'target':cot_result['target'],
           'target_revision':cot['files']['metadata/run_meta.json']['target_revision']}
previous=Image.open(root/'2026-10-12_figure1_before_cot_row.png').convert('RGB')
cot_mask=np.zeros((1024,1536),bool)
for x0,y0,x1,y1 in regions[-2:]:cot_mask[y0:y1,x0:x1]=True
cot_diff=np.any(np.array(out)!=np.array(previous),axis=2)
assert not np.any(cot_diff&~cot_mask)
cot_audit['changed_pixels_outside_row']=int((cot_diff&~cot_mask).sum())
(root/'cot_row_pixel_audit.json').write_text(json.dumps(cot_audit,indent=2))
with (root/'2026-10-12_figure1_tool_calls_results.md').open('a',encoding='utf-8') as f:
    f.write('\nThe pending unrelated-trait row is replaced by Train only on CoT.\n\n```json\n'+json.dumps(cot_audit,indent=2)+'\n```\n')
print(json.dumps(cot_audit,indent=2))
