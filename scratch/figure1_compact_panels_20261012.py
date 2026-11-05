# ABOUTME: Removes the transition strip and updates the two setting headings.
# ABOUTME: Translates the lower panel intact and verifies all other retained pixels.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from figure1_edit_20261012 import text

root=Path('output/2026-10-12_figure1')
destination=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_tool_calls.png')
backup=root/'2026-10-12_figure1_before_compact_panels.png'
if not backup.exists():
    backup.write_bytes(destination.read_bytes())
before=Image.open(backup).convert('RGB')
edited=before.copy()
regions=[]
box=(855,78,1490,118)
im=Image.new('RGB',(635*4,40*4),'#fdfdfd')
text(im,(5,19),'Conversational misalignment after ablations',32,anchor='lm')
edited.paste(im.resize((635,40),Image.Resampling.LANCZOS),box[:2])
regions.append(box)

box=(42,534,826,588)
x0,y0,x1,y1=box
a=np.array(before).astype(float)
left=np.median(a[y0:y1,42:47],axis=1)
right=np.median(a[y0:y1,819:826],axis=1)
t=np.linspace(0,1,x1-x0)[None,:,None]
bg=left[:,None,:]*(1-t)+right[:,None,:]*t
im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize(((x1-x0)*4,(y1-y0)*4),Image.Resampling.BICUBIC)
f=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',48*4)
ImageDraw.Draw(im).text((6*4,29*4),'Generalization to agentic settings',font=f,fill='#050505',anchor='lm')
edited.paste(im.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),box[:2])
regions.append(box)

out=Image.new('RGB',(1536,968))
out.paste(edited.crop((0,0,1536,460)),(0,0))
# Interpolate untouched outer background, excluding the old glow and panel borders.
gap_t=np.linspace(0,1,1536)[None,:,None]
gap=a[460:472,0:1]*(1-gap_t)+a[460:472,1535:1536]*gap_t
out.paste(Image.fromarray(np.rint(gap).astype(np.uint8)),(0,460))
out.paste(edited.crop((0,528,1536,1024)),(0,472))
mask=np.zeros((1024,1536),bool)
for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
diff=np.any(np.array(edited)!=np.array(before),axis=2)
assert not np.any(diff&~mask)
assert np.array_equal(np.array(out)[:460],np.array(edited)[:460])
assert np.array_equal(np.array(out)[472:],np.array(edited)[528:])
out.save(destination)
out.save(root/destination.name)
audit={'source':str(backup),'output':str(destination),'heading_rectangles':regions,
       'removed_strip':[460,528],'new_gap_pixels':12,'lower_panel_shift_y':-56,
       'output_size':list(out.size),'changed_retained_pixels_outside_headings':int((diff&~mask).sum())}
(root/'compact_panels_pixel_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
