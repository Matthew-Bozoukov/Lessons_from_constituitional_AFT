# ABOUTME: Left-aligns the ablation labels using one consistent regular-weight font.
# ABOUTME: Preserves the pill borders and checks every pixel outside label interiors.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import ORIGINAL, text, font

root=Path('output/2026-10-12_figure1')
destination=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_tool_calls.png')
backup=root/'2026-10-12_figure1_before_plain_labels.png'
if not backup.exists():
    backup.write_bytes(destination.read_bytes())
before=Image.open(backup).convert('RGB')
out=before.copy()
a=np.array(ORIGINAL).astype(float)
regions=[]
rows=[(146,'Difficult Advice','#9025ff'),
      (189,'Trait 6 removed','#008000'),
      (231,'Trained only on CoT','#008000'),
      (273,'Grok teacher','#008000'),
      (314,'Low stakes','#008000'),
      (356,'Nonmoral deliberation','#ff0000'),
      (397,'No deliberation','#ff0000')]
for y,label,color in rows:
    box=(870,y-14,1064,y+14)
    t=np.linspace(0,1,194)[None,:,None]
    bg=a[y-14:y+14,870:871]*(1-t)+a[y-14:y+14,1063:1064]*t
    im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize((194*4,28*4),Image.Resampling.BICUBIC)
    assert ImageDraw.Draw(im).textlength(label,font=font(25,False))/4<186
    text(im,(5,14),label,25,fill=color,anchor='lm',bold=False)
    out.paste(im.resize((194,28),Image.Resampling.LANCZOS),box[:2])
    regions.append(box)
# Move the existing rendered bars and values with their corresponding labels.
for old_y,new_y in [(273,233),(314,273),(233,314)]:
    box=(1086,new_y-18,1325,new_y+18)
    out.paste(before.crop((1086,old_y-18,1325,old_y+18)),box[:2])
    regions.append(box)
box=(855,78,1490,118)
im=Image.new('RGB',(635*4,40*4),'#fdfdfd')
heading='Misalignment after ablations on difficult advice dataset'
text(im,(5,19),heading,32,anchor='lm')
out.paste(im.resize((635,40),Image.Resampling.LANCZOS),box[:2])
regions.append(box)
mask=np.zeros((1024,1536),bool)
for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
diff=np.any(np.array(out)!=np.array(before),axis=2)
assert not np.any(diff&~mask)
out.save(destination)
out.save(root/destination.name)
audit={'source':str(backup),'output':str(destination),'edited_rectangles':regions,
       'changed_pixels_outside_edits':int((diff&~mask).sum()),
       'font':'Bahnschrift Condensed regular, 25 px','left_anchor_x':875,
       'rows':list(zip([row[1] for row in rows],[13.3,17.5,12.9,12.9,12.1,14.6,27.9]))}
(root/'label_alignment_pixel_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
