# ABOUTME: Aligns the SFT arrow and trained robot with the Difficult Advice row.
# ABOUTME: Preserves the rest of the figure and verifies a bounded pixel difference.
import json
from pathlib import Path
import numpy as np
from PIL import Image
from figure1_edit_20261012 import robot_head

root=Path('output/2026-10-13_figure1')
destination=Path('C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1.png')
backup=root/'2026-10-13_before_sft_alignment.png'
if not backup.exists():backup.write_bytes(destination.read_bytes())
before=Image.open(backup).convert('RGB'); out=before.copy()
a=np.array(before).astype(float)
# Reconstruct only the vacated arrow rectangle from adjacent empty background.
left=np.median(a[203:321,724:728],axis=1)
right=np.median(a[203:321,836:840],axis=1)
t=np.linspace(0,1,108)[None,:,None]
bg=left[:,None,:]*(1-t)+right[:,None,:]*t
out.paste(Image.fromarray(np.rint(bg).astype(np.uint8)),(728,203))
arrow=robot_head((728,203,836,321))
out.paste(arrow,(728,91),arrow)
# This area already has a uniform background; move the icon without resampling.
robot=before.crop((840,214,946,313))
out.paste('#fdfdfd',(840,214,946,313))
out.paste(robot,(840,100))
regions=[(728,203,836,321),(728,91,836,209),(840,214,946,313),(840,100,946,199)]
mask=np.zeros((before.height,before.width),bool)
for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
diff=np.any(np.array(out)!=np.array(before),axis=2)
assert not np.any(diff&~mask)
out.save(destination);out.save(root/destination.name)
audit={'arrow_translation':[0,-112],'robot_translation':[0,-114],
       'edited_rectangles':regions,'changed_pixels_outside_edits':int((diff&~mask).sum())}
(root/'sft_alignment_pixel_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
