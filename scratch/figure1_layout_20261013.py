# ABOUTME: Adds the trained robot, cleans the ablation plot, and revises the agentic legend.
# ABOUTME: Preserves all pixels outside the three explicitly requested layout regions.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import text, font, robot_head, PURPLE, ORIGINAL, odcv, am

root=Path('output/2026-10-13_figure1'); root.mkdir(exist_ok=True)
source=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_tool_calls.png')
before=Image.open(source).convert('RGB'); out=before.copy(); regions=[]
def paste(box,im):
    out.paste(im.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS).convert('RGB'),box[:2])
    regions.append(box)
grey=robot_head((181,823,268,900))
purple=robot_head((185,665,275,743))
# The wider crop includes the complete right ear; exclude the adjacent terminal.
ImageDraw.Draw(purple).rectangle((84,0,89,38),fill=(0,0,0,0))
hsv=np.array(purple.convert('RGB').convert('HSV')); hsv[hsv[:,:,1]>65,0]=24
orange=Image.fromarray(hsv,mode='HSV').convert('RGBA'); orange.putalpha(purple.getchannel('A'))

box=(840,78,1494,448)
im=Image.new('RGBA',(654*4,370*4),'#fdfdfd'); d=ImageDraw.Draw(im)
im.alpha_composite(purple.resize((100*4,93*4),Image.Resampling.LANCZOS),(3*4,139*4))
text(im,(115,19),'Aggregate misalignment after ablations',30,anchor='lm')
axis=1170-840; baseline=1440-840
rows=[('Difficult Advice',13.3),('Trained only on CoT',12.9),('Grok teacher',12.9),
      ('Low stakes',12.1),('Amoral deliberation',14.6),('No deliberation',27.9)]
for i,(label,value) in enumerate(rows):
    y=150+i*48-78
    fill='#eadbfa' if i==0 else '#eeece0'
    edge='#9025ff' if i==0 else '#a29e8d'
    ink='#9025ff' if i==0 else '#555247'
    d.rounded_rectangle((115*4,(y-18)*4,320*4,(y+18)*4),radius=8*4,fill=fill,outline=edge,width=8)
    text(im,(127,y),label,25,fill=ink,anchor='lm',bold=False)
    end=axis+value*6
    d.rectangle((axis*4,(y-12)*4,end*4,(y+12)*4),fill=PURPLE if i==0 else '#eeece0',outline='#111111',width=8)
    text(im,(end+9,y),f'{value:g}%',23,anchor='lm')
d.line((axis*4,43*4,axis*4,341*4),fill='#111111',width=10)
d.line((axis*4,341*4,634*4,341*4),fill='#111111',width=10)
for y in range(67,338,15):
    d.line((baseline*4,y*4,baseline*4,(y+8)*4),fill='#888888',width=8)
text(im,(baseline,48),'baseline 45%',20,fill='#666666')
text(im,(482,356),'misalignment',24)
d.line((572*4,356*4,591*4,356*4),fill='#111111',width=7)
d.line((585*4,351*4,591*4,356*4,585*4,361*4),fill='#111111',width=7)
paste(box,im)

box=(42,478,1494,535); x0,y0,x1,y1=box
a=np.array(before).astype(float)
left=np.median(a[y0:y1,42:47],axis=1); right=np.median(a[y0:y1,1489:1494],axis=1)
t=np.linspace(0,1,x1-x0)[None,:,None]
bg=left[:,None,:]*(1-t)+right[:,None,:]*t
im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize(((x1-x0)*4,57*4),Image.Resampling.BICUBIC).convert('RGBA')
entries=[(grey,'= Baseline'),(purple,'= Difficult Advice'),(orange,'= Difficult Advice + Tool Definitions')]
widths=[52+ImageDraw.Draw(im).textlength(label,font=font(28))/4 for _,label in entries]
gap=47.5; cursor=((x1-x0)-sum(widths)-gap*2)/2
assert cursor>0
for (head,label),width in zip(entries,widths):
    im.alpha_composite(head.resize((45*4,41*4),Image.Resampling.LANCZOS),(round(cursor*4),6*4))
    text(im,(cursor+52,29),label,28,anchor='lm')
    cursor+=width+gap
paste(box,im)

box=(824,542,1480,590)
im=Image.new('RGB',(656*4,48*4),'#fdfdfd'); d=ImageDraw.Draw(im)
text(im,(6,24),'Multi-agent',40,anchor='lm')
# A red cross matching the visual role of the existing green check.
d.line((178*4,6*4,214*4,42*4),fill='#ff0808',width=12*4)
d.line((214*4,6*4,178*4,42*4),fill='#ff0808',width=12*4)
text(im,(231,25),'but tool definitions help',28,anchor='lm')
paste(box,im)

# Rescale the single-agent plot to 75 while retaining its values and bar positions.
box=(493,589,781,836); x0,y0,x1,y1=box; w,h=x1-x0,y1-y0
sample=np.array(ORIGINAL)[645:892,769:781].mean(axis=1).astype(np.uint8)
im=Image.fromarray(np.repeat(sample[:,None,:],w,axis=1)).resize((w*4,h*4),Image.Resampling.BICUBIC)
d=ImageDraw.Draw(im); base=838-y0; top=615-y0
for value in [25,50,75]:
    y=base-(base-top)*value/75
    d.line((0,y*4,w*4,y*4),fill='#dfdfdf',width=4)
    text(im,(w-2,y-7),str(value),12,fill='#777777',anchor='rm',bold=False)
bars=[(497,538,odcv('2026-10-06-odcv-qwen36-0-plain'),'#dbdbdb'),
      (542,599,odcv('2026-10-06-odcv-qwen36-0-da-15'),PURPLE),
      (640,683,am('2026-10-08-am-qwen36-0-plain'),'#dbdbdb'),
      (689,746,am('2026-10-08-am-qwen36-0-da-15'),PURPLE)]
for left,right,value,color in bars:
    y=base-(base-top)*value/75
    d.rectangle(((left-x0)*4,y*4,(right-x0)*4,(base+2)*4),fill=color,outline='#111111',width=12)
    label=f'{value:.1f}%' if value%1 else f'{value:.0f}%'
    text(im,((left+right)/2-x0,y-14),label,21)
paste(box,im)

previous=Image.open(root/'2026-10-13_before_scale_spacing.png').convert('RGB')
change_mask=np.zeros((before.height,before.width),bool)
for x0,y0,x1,y1 in regions[1:]:change_mask[y0:y1,x0:x1]=True
change_mask[214:313,840:946]=True
assert not np.any(np.any(np.array(out)!=np.array(previous),axis=2)&~change_mask)

mask=np.zeros((before.height,before.width),bool)
for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
diff=np.any(np.array(before)!=np.array(out),axis=2)
assert not np.any(diff&~mask)
destination=Path('C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1.png')
out.save(destination); out.save(root/destination.name)
audit={'source':str(source),'output':str(destination),'edited_rectangles':regions,
       'changed_pixels_outside_edits':int((diff&~mask).sum()),'rows':rows}
(root/'layout_pixel_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
