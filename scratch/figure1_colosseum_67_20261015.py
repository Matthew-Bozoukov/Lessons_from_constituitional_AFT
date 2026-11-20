# ABOUTME: Updates the tool-definitions Colosseum bar to the user-supplied 67 percent.
# ABOUTME: Preserves every pixel outside the orange bar and its value label.
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from figure1_edit_20261012 import text

path=Path('C:/Users/nikak/Downloads/2026-10-15_Difficult_Advice_Figure_1_aggregate_odcv_agentic.png')
root=Path('output/2026-10-15_figure1')
before=Image.open(path).convert('RGB')
before.save(root/'2026-10-15_before_colosseum_67.png')
out=before.copy()
box=(1384,650,1446,837)
x0,y0,x1,y1=box
a=np.array(before)
background=np.repeat(a[y0:y1,1448:1449,:],x1-x0,axis=1)
im=Image.fromarray(background).resize(((x1-x0)*4,(y1-y0)*4),Image.Resampling.BICUBIC)
d=ImageDraw.Draw(im)
top=838-223*0.67
d.rectangle(((1388-x0)*4,(top-y0)*4,(1441-x0)*4,(840-y0)*4),fill='#e79532',outline='#111111',width=12)
text(im,((1388+1441)/2-x0,top-14-y0),'67%',21)
out.paste(im.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS),box[:2])
diff=np.any(np.array(out)!=a,axis=2)
diff[y0:y1,x0:x1]=False
assert not diff.any()
out.save(path)
out.save(root/path.name)
print('Updated to 67%; all pixels outside the bar/label region unchanged.')
