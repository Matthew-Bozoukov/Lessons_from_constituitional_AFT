# ABOUTME: Removes the blurred lighting behind the Figure 1 transition heading.
# ABOUTME: Replaces only that heading strip and verifies all other pixels are unchanged.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root=Path('output/2026-10-12_figure1')
source=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_clean.png')
original=Image.open(source).convert('RGB')
arr=np.array(original)
box=(462,460,1250,527)
x0,y0,x1,y1=box
w,h=x1-x0,y1-y0
# Estimate the neutral background from the same rows, beyond the lighting effect.
left=np.median(arr[y0:y1,250:300],axis=1)
right=np.median(arr[y0:y1,1280:1340],axis=1)
x=np.arange(x0,x1)
t=np.clip((x-275)/(1310-275),0,1)
bg=left[:,None,:]*(1-t[None,:,None])+right[:,None,:]*t[None,:,None]
# Feather only the blank left/right margins so the blue arrow's existing glow has no hard cut.
blend=np.minimum(np.clip(np.arange(w)/9,0,1),np.clip((w-1-np.arange(w))/35,0,1))
bg=bg*blend[None,:,None]+arr[y0:y1,x0:x1]*(1-blend[None,:,None])
patch=Image.fromarray(np.rint(bg).astype('uint8')).resize((w*4,h*4),Image.Resampling.BICUBIC)
font=ImageFont.truetype('C:/Windows/Fonts/bahnschrift.ttf',41*4)
font.set_variation_by_name('Bold SemiCondensed')
layer=Image.new('RGBA',(1100*4,h*4))
draw=ImageDraw.Draw(layer)
arrowfont=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',41*4)
cursor=0
for value,f in [('Conversational ',font),('→',arrowfont),(' agentic generalisation?',font)]:
    draw.text((cursor,45*4),value,font=f,fill='#080808',anchor='ls')
    cursor+=draw.textlength(value,font=f)
layer=layer.crop(layer.getbbox())
layer=layer.resize((674*4,38*4),Image.Resampling.LANCZOS)
patch=patch.convert('RGBA')
patch.alpha_composite(layer,(8*4,15*4))
result=original.copy()
result.paste(patch.convert('RGB').resize((w,h),Image.Resampling.LANCZOS),(x0,y0))
out=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_crisp_heading.png')
result.save(out)
result.save(root/out.name)
delta=np.any(np.array(result)!=arr,axis=2)
mask=np.zeros(delta.shape,bool);mask[y0:y1,x0:x1]=True
assert not np.any(delta&~mask)
(root/'heading_pixel_audit.json').write_text(json.dumps({'source':str(source),'output':str(out),'edit_rectangle':box,'changed_pixels_outside_heading':int((delta&~mask).sum())},indent=2))
print(out)
