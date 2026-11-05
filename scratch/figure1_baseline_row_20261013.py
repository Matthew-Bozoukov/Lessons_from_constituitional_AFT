# ABOUTME: Replaces dashed baselines with explicit rows and right-aligns the robot legend.
# ABOUTME: Updates both figure variants with bounded edits and keeps prior copies.
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from figure1_edit_20261012 import text,font,robot_head,PURPLE

root=Path('output/2026-10-13_figure1')
grey=robot_head((181,823,268,900));purple=robot_head((185,665,275,743))
ImageDraw.Draw(purple).rectangle((84,0,89,38),fill=(0,0,0,0))
hsv=np.array(purple.convert('RGB').convert('HSV'));hsv[hsv[:,:,1]>65,0]=24
orange=Image.fromarray(hsv,mode='HSV').convert('RGBA');orange.putalpha(purple.getchannel('A'))
labels=['Baseline','Difficult Advice','Trained only on CoT','Grok teacher','Low stakes','Amoral deliberation','No deliberation']
scores=[r['files']['results/results.json']['overall_honesty_score'] for r in json.loads((root/'mask_sources.json').read_text())]
audits={}
for suffix in ['', '_mask']:
    destination=Path(f'C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1{suffix}.png')
    backup=root/f'2026-10-13_before_baseline_row{suffix}.png'
    if not backup.exists():backup.write_bytes(destination.read_bytes())
    before=Image.open(backup).convert('RGB');out=before.copy();a=np.array(before).astype(float);regions=[]
    def paste(box,im):
        out.paste(im.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS).convert('RGB'),box[:2]);regions.append(box)

    out.paste('#fdfdfd',(1167,114,1494,118));regions.append((1167,114,1494,118))
    box=(950,118,1494,423)
    im=Image.new('RGB',(544*4,305*4),'#fdfdfd');d=ImageDraw.Draw(im)
    values=[scores[-1]]+scores[:6] if suffix else [45,13.3,12.9,12.9,12.1,14.6,27.9]
    scale=2.65 if suffix else 6
    for i,(label,value) in enumerate(zip(labels,values)):
        y=147+41*i-118
        fill='#dbdbdb' if i==0 else '#eadbfa' if i==1 else '#eeece0'
        edge='#a0a0a0' if i==0 else '#9025ff' if i==1 else '#a29e8d'
        ink='#9025ff' if i==1 else '#555247'
        d.rounded_rectangle((5*4,(y-17)*4,210*4,(y+17)*4),radius=8*4,fill=fill,outline=edge,width=8)
        text(im,(17,y),label,25,fill=ink,anchor='lm',bold=False)
        end=220+value*scale
        d.rectangle((220*4,(y-11)*4,end*4,(y+11)*4),fill=PURPLE if i==1 else '#dbdbdb' if i==0 else '#eeece0',outline='#111111',width=8)
        text(im,(end+7,y),f'{value:g}%',23,anchor='lm')
    d.line((220*4,3*4,220*4,301*4),fill='#111111',width=10)
    d.line((220*4,301*4,524*4,301*4),fill='#111111',width=10)
    paste(box,im)

    # Erase only the existing icons and place smaller versions centered at y=188.
    box=(728,110,836,220)
    t=np.linspace(0,1,108)[None,:,None]
    bg=np.median(a[110:220,724:728],axis=1)[:,None,:]*(1-t)+np.median(a[110:220,836:840],axis=1)[:,None,:]*t
    out.paste(Image.fromarray(np.rint(bg).astype(np.uint8)),box[:2]);regions.append(box)
    box=(850,110,950,224)
    out.paste('#fdfdfd',box);regions.append(box)
    head=purple.resize((44,39),Image.Resampling.LANCZOS)
    out.paste(head,(896,163),head)
    baseline_head=grey.resize((44,39),Image.Resampling.LANCZOS)
    out.paste(baseline_head,(896,122),baseline_head)
    # Equal 44-by-39 icons share x=896, one row pitch apart. Meet the ear at y=188.
    arrow=Image.new('RGBA',(179*4,74*4),(0,0,0,0))
    ad=ImageDraw.Draw(arrow)
    ad.polygon([(0,20*4),(122*4,20*4),(122*4,2*4),(178*4,37*4),
                (122*4,72*4),(122*4,54*4),(0,54*4)],fill='#009dff')
    af=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',34*4)
    ad.text((70*4,37*4),'SFT',font=af,fill='white',anchor='mm')
    arrow=arrow.resize((179,74),Image.Resampling.LANCZOS)
    out.paste(arrow,(719,151),arrow);regions.append((719,151,898,225))

    box=(42,478,1494,535);x0,y0,x1,y1=box
    left=np.median(a[y0:y1,42:47],axis=1);right=np.median(a[y0:y1,1489:1494],axis=1)
    t=np.linspace(0,1,x1-x0)[None,:,None]
    bg=left[:,None,:]*(1-t)+right[:,None,:]*t
    im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize(((x1-x0)*4,57*4),Image.Resampling.BICUBIC).convert('RGBA')
    entries=[(grey,'= Baseline'),(purple,'= Difficult Advice'),(orange,'= Difficult Advice + Unused Tool Definitions')]
    widths=[52+ImageDraw.Draw(im).textlength(label,font=font(28))/4 for _,label in entries]
    gap=20;cursor=(x1-x0)-5-sum(widths)-2*gap
    heading_font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',48*4)
    heading='Agentic Scenarios'
    assert 6+ImageDraw.Draw(im).textlength(heading,font=heading_font)/4+24<cursor
    ImageDraw.Draw(im).text((6*4,29*4),heading,font=heading_font,fill='#050505',anchor='lm')
    for (head,label),width in zip(entries,widths):
        im.alpha_composite(head.resize((45*4,41*4),Image.Resampling.LANCZOS),(round(cursor*4),6*4))
        text(im,(cursor+52,29),label,28,anchor='lm');cursor+=width+gap
    paste(box,im)
    mask=np.zeros((before.height,before.width),bool)
    for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
    diff=np.any(np.array(out)!=np.array(before),axis=2)
    assert not np.any(diff&~mask)
    out.save(destination);out.save(root/destination.name)
    audits[suffix or 'main']={'edited_rectangles':regions,'changed_pixels_outside_edits':int((diff&~mask).sum()),'rows':dict(zip(labels,values))}
(root/'baseline_row_pixel_audit.json').write_text(json.dumps(audits,indent=2))
print(json.dumps(audits,indent=2))
