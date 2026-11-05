# ABOUTME: Gives all six ablation labels a uniform light grey-yellow treatment.
# ABOUTME: Verifies that every pixel outside those label boxes remains unchanged.
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import text

root=Path('output/2026-10-12_figure1')
destination=Path('C:/Users/nikak/Downloads/2026-10-12_Difficult_Advice_Figure_1_tool_calls.png')
backup=root/'2026-10-12_figure1_before_neutral_ablations.png'
if not backup.exists():
    backup.write_bytes(destination.read_bytes())
before=Image.open(backup).convert('RGB')
out=before.copy()
rows=[(171,207,189,'Trait 6 removed'),
      (212,249,231,'Trained only on CoT'),
      (253,291,273,'Grok teacher'),
      (295,333,314,'Low stakes'),
      (338,375,356,'Nonmoral deliberation'),
      (379,417,397,'No deliberation')]
mask=np.zeros((1024,1536),bool)
for top,bottom,center,label in rows:
    box=(859,top-1,1075,bottom+1)
    w,h=box[2]-box[0],box[3]-box[1]
    im=Image.new('RGB',(w*4,h*4),'#fdfdfd')
    draw=ImageDraw.Draw(im)
    draw.rounded_rectangle((2*4,1*4,(w-3)*4,(h-2)*4),radius=8*4,
                           fill='#eeece0',outline='#a29e8d',width=2*4)
    text(im,(875-box[0],center-box[1]),label,25,fill='#555247',anchor='lm',bold=False)
    out.paste(im.resize((w,h),Image.Resampling.LANCZOS),box[:2])
    mask[box[1]:box[3],box[0]:box[2]]=True
diff=np.any(np.array(out)!=np.array(before),axis=2)
assert not np.any(diff&~mask)
out.save(destination)
out.save(root/destination.name)
audit={'fill':'#eeece0','outline':'#a29e8d','text':'#555247',
       'changed_pixels_outside_ablation_labels':int((diff&~mask).sum())}
(root/'neutral_ablations_pixel_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
