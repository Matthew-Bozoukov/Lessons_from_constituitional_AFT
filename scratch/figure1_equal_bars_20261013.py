# ABOUTME: Makes all four single-agent bars equally wide while retaining their centers and values.
# ABOUTME: Updates both variants and verifies every pixel outside the plot interior is unchanged.
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from figure1_edit_20261012 import ORIGINAL,PURPLE,text,odcv,am

root=Path('output/2026-10-13_figure1')
box=(493,589,781,836);x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
sample=np.array(ORIGINAL)[645:892,769:781].mean(axis=1).astype(np.uint8)
im=Image.fromarray(np.repeat(sample[:,None,:],w,axis=1)).resize((w*4,h*4),Image.Resampling.BICUBIC)
d=ImageDraw.Draw(im);base=838-y0;top=615-y0
for value in [25,50]:
    y=base-(base-top)*value/50
    d.line((0,y*4,w*4,y*4),fill='#dfdfdf',width=4)
    text(im,(w-2,y-7),str(value),12,fill='#777777',anchor='rm',bold=False)
bars=[(517.5,odcv('2026-10-06-odcv-qwen36-0-plain'),'#dbdbdb'),
      (570.5,odcv('2026-10-06-odcv-qwen36-0-da-15'),PURPLE),
      (661.5,am('2026-10-08-am-qwen36-0-plain'),'#dbdbdb'),
      (717.5,am('2026-10-08-am-qwen36-0-da-15'),PURPLE)]
for center,value,color in bars:
    y=base-(base-top)*value/50
    d.rectangle(((center-20.5-x0)*4,y*4,(center+20.5-x0)*4,(base+2)*4),fill=color,outline='#111111',width=12)
    text(im,(center-x0,y-14),f'{value:.1f}%' if value%1 else f'{value:.0f}%',21)
patch=im.resize((w,h),Image.Resampling.LANCZOS)
for suffix in ['', '_mask']:
    destination=Path(f'C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1{suffix}.png')
    backup=root/f'2026-10-13_before_equal_bars{suffix}.png'
    if not backup.exists():backup.write_bytes(destination.read_bytes())
    before=Image.open(backup).convert('RGB');out=before.copy();out.paste(patch,box[:2])
    diff=np.any(np.array(before)!=np.array(out),axis=2)
    diff[y0:y1,x0:x1]=False
    assert not np.any(diff)
    out.save(destination);out.save(root/destination.name)
print('Both variants: four equal 41-pixel bar spans; unchanged centers, heights, values, and pixels outside plot interior.')
