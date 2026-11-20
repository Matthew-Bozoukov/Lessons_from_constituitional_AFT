# ABOUTME: Saves the compact 50-percent figure and a separate pinned MASK-honesty variant.
# ABOUTME: Keeps the supplied saved copy intact and audits every edited rectangle.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import text,font,robot_head,PURPLE,ORIGINAL,odcv,am

root=Path('output/2026-10-13_figure1')
source=Path('C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1_saved_copy.png')
before=Image.open(source).convert('RGB'); out=before.copy(); regions=[]
def paste(box,im):
    out.paste(im.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS).convert('RGB'),box[:2]); regions.append(box)

box=(728,91,836,209)
a=np.array(before).astype(float)
t=np.linspace(0,1,108)[None,:,None]
bg=np.median(a[91:209,724:728],axis=1)[:,None,:]*(1-t)+np.median(a[91:209,836:840],axis=1)[:,None,:]*t
out.paste(Image.fromarray(np.rint(bg).astype(np.uint8)),box[:2]);regions.append(box)
arrow=robot_head((728,203,836,321)).resize((108,76),Image.Resampling.LANCZOS)
out.paste(arrow,(728,112),arrow)
box=(840,100,946,199)
out.paste('#fdfdfd',box);regions.append(box)
robot=before.crop(box).resize((80,75),Image.Resampling.LANCZOS)
out.paste(robot,(853,112))

box=(493,589,781,836); x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
sample=np.array(ORIGINAL)[645:892,769:781].mean(axis=1).astype(np.uint8)
im=Image.fromarray(np.repeat(sample[:,None,:],w,axis=1)).resize((w*4,h*4),Image.Resampling.BICUBIC)
d=ImageDraw.Draw(im); base=838-y0;top=615-y0
for value in [25,50]:
    y=base-(base-top)*value/50
    d.line((0,y*4,w*4,y*4),fill='#dfdfdf',width=4)
    text(im,(w-2,y-7),str(value),12,fill='#777777',anchor='rm',bold=False)
bars=[(497,538,odcv('2026-10-06-odcv-qwen36-0-plain'),'#dbdbdb'),
      (542,599,odcv('2026-10-06-odcv-qwen36-0-da-15'),PURPLE),
      (640,683,am('2026-10-08-am-qwen36-0-plain'),'#dbdbdb'),
      (689,746,am('2026-10-08-am-qwen36-0-da-15'),PURPLE)]
for left,right,value,color in bars:
    y=base-(base-top)*value/50
    d.rectangle(((left-x0)*4,y*4,(right-x0)*4,(base+2)*4),fill=color,outline='#111111',width=12)
    text(im,((left+right)/2-x0,y-14),f'{value:.1f}%' if value%1 else f'{value:.0f}%',21)
paste(box,im)

grey=robot_head((181,823,268,900));purple=robot_head((185,665,275,743))
ImageDraw.Draw(purple).rectangle((84,0,89,38),fill=(0,0,0,0))
hsv=np.array(purple.convert('RGB').convert('HSV'));hsv[hsv[:,:,1]>65,0]=24
orange=Image.fromarray(hsv,mode='HSV').convert('RGBA');orange.putalpha(purple.getchannel('A'))
box=(42,478,1494,535);x0,y0,x1,y1=box
left=np.median(a[y0:y1,42:47],axis=1);right=np.median(a[y0:y1,1489:1494],axis=1)
t=np.linspace(0,1,x1-x0)[None,:,None]
bg=left[:,None,:]*(1-t)+right[:,None,:]*t
im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize(((x1-x0)*4,57*4),Image.Resampling.BICUBIC).convert('RGBA')
entries=[(grey,'= Baseline'),(purple,'= Difficult Advice'),(orange,'= Difficult Advice + Unused Tool Definitions')]
widths=[52+ImageDraw.Draw(im).textlength(label,font=font(28))/4 for _,label in entries]
gap=47.5;cursor=((x1-x0)-sum(widths)-2*gap)/2
for (head,label),width in zip(entries,widths):
    im.alpha_composite(head.resize((45*4,41*4),Image.Resampling.LANCZOS),(round(cursor*4),6*4))
    text(im,(cursor+52,29),label,28,anchor='lm');cursor+=width+gap
paste(box,im)
box=(1050,542,1480,590)
im=Image.new('RGB',(430*4,48*4),'#fdfdfd')
text(im,(5,25),'but unused tool definitions help',28,anchor='lm')
paste(box,im)

def audit(previous,current,boxes):
    mask=np.zeros((current.height,current.width),bool)
    for x0,y0,x1,y1 in boxes:mask[y0:y1,x0:x1]=True
    diff=np.any(np.array(previous)!=np.array(current),axis=2)
    assert not np.any(diff&~mask)
    return {'edited_rectangles':boxes,'changed_pixels_outside_edits':int((diff&~mask).sum())}
main_audit=audit(before,out,regions)
destination=source.with_name('2026-10-13_Difficult_Advice_Figure_1.png')
out.save(destination);out.save(root/destination.name)
main=out.copy();regions=[]
sources=json.loads((root/'mask_sources.json').read_text())
scores=[r['files']['results/results.json']['overall_honesty_score'] for r in sources]
for r in sources:
    result=r['files']['results/results.json'];meta=r['files']['metadata/run_meta.json']
    assert result['n_rows']==1000 and not result['smoke'] and result['benchmark']=='MASK'
    assert meta['target_revision']
box=(950,78,1494,118)
im=Image.new('RGB',(544*4,40*4),'#fdfdfd')
text(im,(5,19),'Misalignment after ablations',30,anchor='lm');paste(box,im)
box=(1167,112,1494,448)
im=Image.new('RGB',(327*4,336*4),'#fdfdfd');d=ImageDraw.Draw(im)
axis=3;scale=2.65; baseline=axis+scores[-1]*scale
for y in range(33,302,15):d.line((baseline*4,y*4,baseline*4,(y+8)*4),fill='#888888',width=8)
text(im,(baseline,14),f'baseline {scores[-1]:g}%',20,fill='#666666')
for i,value in enumerate(scores[:6]):
    y=150+i*48-112;end=axis+value*scale
    d.rectangle((axis*4,(y-12)*4,end*4,(y+12)*4),fill=PURPLE if i==0 else '#eeece0',outline='#111111',width=8)
    text(im,(end+7,y),f'{value:g}%',23,anchor='lm')
d.line((axis*4,9*4,axis*4,307*4),fill='#111111',width=10)
d.line((axis*4,307*4,307*4,307*4),fill='#111111',width=10)
text(im,(163,322),'MASK honesty rate',24)
paste(box,im)
mask_audit=audit(main,out,regions)
mask_path=destination.with_name(destination.stem+'_mask.png')
out.save(mask_path);out.save(root/mask_path.name)
(root/'mask_variant_pixel_audit.json').write_text(json.dumps({'main':main_audit,'mask':mask_audit},indent=2))
labels=['Difficult Advice','Trained only on CoT','Grok teacher','Low stakes','Amoral deliberation','No deliberation','Baseline']
lines=['# Figure 1 MASK variant','', 'Published overall_honesty_score; full 1000-row MASK evaluations. One pass per ablation; two passes for Difficult Advice, Grok teacher, and baseline where reported. Scores count non-dishonest responses, including evasion, per the published metric.','']
for label,r,score in zip(labels,sources,scores):
    meta=r['files']['metadata/run_meta.json'];result=r['files']['results/results.json']
    lines.append(f"- {label}: {score:g}%; target `{result['target']}` at `{meta['target_revision']}`; [pinned evaluation](https://huggingface.co/datasets/{r['repo']}/tree/{r['revision']}); {result['passes']} pass(es).")
(root/'2026-10-13_figure1_mask_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(zip(labels,scores)),indent=2))
print(destination);print(mask_path)
