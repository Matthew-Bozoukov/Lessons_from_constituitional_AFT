# ABOUTME: Changes the requested headings and shifts the two row robots six pixels right.
# ABOUTME: Preserves both figure variants outside tightly bounded edit rectangles.
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from figure1_edit_20261012 import text,robot_head

root=Path('output/2026-10-13_figure1')
grey=robot_head((181,823,268,900));purple=robot_head((185,665,275,743))
ImageDraw.Draw(purple).rectangle((84,0,89,38),fill=(0,0,0,0))
audits={}
for suffix in ['', '_mask']:
    destination=Path(f'C:/Users/nikak/Downloads/2026-10-13_Difficult_Advice_Figure_1{suffix}.png')
    backup=root/f'2026-10-13_before_new_titles{suffix}.png'
    if not backup.exists():backup.write_bytes(destination.read_bytes())
    before=Image.open(backup).convert('RGB');out=before.copy();a=np.array(before);regions=[]
    box=(950,78,1494,114)
    im=Image.new('RGB',(544*4,36*4),'#fdfdfd')
    text(im,(5,19),'Alignment after various ablations',30,anchor='lm')
    out.paste(im.resize((544,36),Image.Resampling.LANCZOS),box[:2]);regions.append(box)
    box=(42,478,646,535);x0,y0,x1,y1=box
    left=np.median(a[y0:y1,42:47],axis=1);right=np.median(a[y0:y1,640:646],axis=1)
    t=np.linspace(0,1,x1-x0)[None,:,None]
    bg=left[:,None,:]*(1-t)+right[:,None,:]*t
    im=Image.fromarray(np.rint(bg).astype(np.uint8)).resize((604*4,57*4),Image.Resampling.BICUBIC)
    heading='Generalization To Agentic Scenarios'
    for size in range(48,20,-1):
        f=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',size*4)
        if ImageDraw.Draw(im).textlength(heading,font=f)/4<=585:break
    ImageDraw.Draw(im).text((6*4,29*4),heading,font=f,fill='#050505',anchor='lm')
    out.paste(im.resize((604,57),Image.Resampling.LANCZOS),box[:2]);regions.append(box)
    box=(896,122,947,202)
    old=np.array(before.crop(box));patch=np.full_like(old,253)
    # Keep the blue arrow tip exactly as rendered while clearing the old robot pixels.
    blue=(old[:,:,2]>180)&(old[:,:,1]>60)&(old[:,:,0]<70)
    patch[blue]=old[blue]
    out.paste(Image.fromarray(patch),box[:2]);regions.append(box)
    for head,y in [(grey,122),(purple,163)]:
        icon=head.resize((44,39),Image.Resampling.LANCZOS)
        out.paste(icon,(902,y),icon)
    mask=np.zeros((before.height,before.width),bool)
    for x0,y0,x1,y1 in regions:mask[y0:y1,x0:x1]=True
    diff=np.any(np.array(out)!=a,axis=2)
    assert not np.any(diff&~mask)
    out.save(destination);out.save(root/destination.name)
    audits[suffix or 'main']={'font_size':size,'robot_shift_x':6,'edited_rectangles':regions,'changed_pixels_outside_edits':int((diff&~mask).sum())}
(root/'new_titles_pixel_audit.json').write_text(json.dumps(audits,indent=2))
print(json.dumps(audits,indent=2))
