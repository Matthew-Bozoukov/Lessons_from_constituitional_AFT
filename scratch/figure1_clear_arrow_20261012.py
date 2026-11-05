# ABOUTME: Removes the glow around Figure 1's downward blue transition arrow.
# ABOUTME: Redraws a crisp arrow in a bounded strip and verifies all other pixels are unchanged.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root=Path('output/2026-10-12_figure1')
source=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_crisp_heading.png')
original=Image.open(source).convert('RGB')
arr=np.array(original)
reference=np.array(Image.open('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_clean.png').convert('RGB'))
box=(300,460,471,527)
x0,y0,x1,y1=box
w,h=x1-x0,y1-y0
left=np.median(reference[y0:y1,250:300],axis=1)
right=np.median(reference[y0:y1,1280:1340],axis=1)
t=(np.arange(x0,x1)-275)/(1310-275)
bg=left[:,None,:]*(1-t[None,:,None])+right[:,None,:]*t[None,:,None]
blend=np.clip(np.arange(w)/22,0,1)
bg=bg*blend[None,:,None]+arr[y0:y1,x0:x1]*(1-blend[None,:,None])
patch=Image.fromarray(np.rint(bg).astype('uint8')).resize((w*4,h*4),Image.Resampling.BICUBIC)
draw=ImageDraw.Draw(patch)
points=[(395,461),(433,461),(433,486),(455,486),(415,523),(374,486),(395,486)]
color=tuple(np.median(reference[465:480,400:429],axis=(0,1)).astype(int))
draw.polygon([((x-x0)*4,(y-y0)*4) for x,y in points],fill=color)
patch=patch.resize((w,h),Image.Resampling.LANCZOS)
# Preserve any antialiased black pixels belonging to the adjacent caption.
p=np.array(patch)
caption=(np.min(arr[y0:y1,x0:x1],axis=2)<80)&(np.arange(x0,x1)[None,:]>=468)
p[caption]=arr[y0:y1,x0:x1][caption]
result=original.copy();result.paste(Image.fromarray(p),(x0,y0))
out=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_crisp.png')
result.save(out);result.save(root/out.name)
delta=np.any(np.array(result)!=arr,axis=2)
mask=np.zeros(delta.shape,bool);mask[y0:y1,x0:x1]=True
assert not np.any(delta&~mask)
(root/'arrow_pixel_audit.json').write_text(json.dumps({'source':str(source),'output':str(out),'edit_rectangle':box,'changed_pixels_outside_arrow_strip':int((delta&~mask).sum())},indent=2))
print(out)
