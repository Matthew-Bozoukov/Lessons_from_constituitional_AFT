# ABOUTME: Composites a robot legend and measured charts into the supplied Figure 1 PNG.
# ABOUTME: Draws only within explicit edit rectangles and checks every other pixel is identical.
import json
import hashlib
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('output/2026-10-12_figure1')
SOURCE = Path('C:/Users/nikak/Downloads/Difficult Advice Posttraining and Generalisation.png')
ORIGINAL = Image.open(SOURCE).convert('RGB')
OUT = ORIGINAL.copy()
S = 4
REGIONS = []
PURPLE = '#ad42f5'
GREY = '#707070'
DELIBERATION_RUN = '2026-10-04-odcv-qwen36-0-ablation-stated-harm-15'
ROWS = json.loads((ROOT/'pinned_sources.json').read_text())
ROWS += json.loads((ROOT/'extra_sources.json').read_text())
RUNS = {r['repo'].split('/')[-1]: r for r in ROWS}

def font(size, bold=True):
    f = ImageFont.truetype('C:/Windows/Fonts/bahnschrift.ttf', round(size*S))
    f.set_variation_by_name('Bold Condensed' if bold else 'Condensed')
    return f

def text(im, xy, value, size=24, fill='#101010', anchor='mm', bold=True):
    ImageDraw.Draw(im).text(tuple(round(v*S) for v in xy), value, font=font(size,bold), fill=fill, anchor=anchor)

def paste_region(box, patch):
    OUT.paste(patch.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS),box[:2])
    REGIONS.append(box)

def robot_head(box):
    im=ORIGINAL.crop(box).convert('RGBA')
    arr=np.array(im)
    # Remove only the near-white background connected to the crop boundary.
    rgb=arr[:,:,:3].astype(int)
    white=(rgb.min(2)>190)&((rgb.max(2)-rgb.min(2))<24)
    h,w=white.shape
    seen=np.zeros((h,w),bool)
    q=deque([(x,0) for x in range(w)]+[(x,h-1) for x in range(w)]+[(0,y) for y in range(h)]+[(w-1,y) for y in range(h)])
    while q:
        x,y=q.popleft()
        if x<0 or y<0 or x>=w or y>=h or seen[y,x] or not white[y,x]: continue
        seen[y,x]=True
        q.extend([(x-1,y),(x+1,y),(x,y-1),(x,y+1)])
    arr[seen,3]=0
    return Image.fromarray(arr)

def legend():
    box=(952,17,1481,69)
    im=ORIGINAL.crop(box).resize((529*S,52*S),Image.Resampling.LANCZOS).convert('RGBA')
    grey=robot_head((181,823,268,900)).resize((53*S,46*S),Image.Resampling.LANCZOS)
    purple=robot_head((185,665,269,743)).resize((53*S,46*S),Image.Resampling.LANCZOS)
    # Entire legend ends at the right margin of the title row.
    label2='= Difficult Advice'
    width2=ImageDraw.Draw(im).textlength(label2,font=font(27))/S
    start2=522-width2
    im.alpha_composite(purple,(round((start2-60)*S),2*S))
    text(im,(522,27),label2,27,anchor='rm')
    start1=start2-228
    im.alpha_composite(grey,(round(start1*S),2*S))
    text(im,(start1+61,27),'= Baseline',27,anchor='lm')
    paste_region(box,im.convert('RGB'))

def result(name): return RUNS[name]['files']['results/results.json']
def odcv(name): return result(name)['ours']['overall']['mr_pct']
def am(name):
    d=result(name)['overall']; assert d['n_missing_classification']==0
    return 100*d['harmful']/d['n']
def hospital(name):
    d=RUNS[name]['files']['results/per_seed.json']['pair_sabotaged']['self_sacrificial']
    return 100*sum(d.values())/len(d)

def vertical(box, bars):
    x0,y0,x1,y1=box; w=x1-x0; h=y1-y0
    # A nearby empty background strip supplies the original panel's pale texture.
    a=np.array(ORIGINAL)
    sample=a[y0:y1,x1-12:x1].mean(axis=1).astype(np.uint8)
    bg=np.repeat(sample[:,None,:],w,axis=1)
    im=Image.fromarray(bg).resize((w*S,h*S),Image.Resampling.BICUBIC)
    dr=ImageDraw.Draw(im)
    base=894-y0; top=671-y0
    for v in (25,50,75,100):
        y=base-(base-top)*v/100
        dr.line((0,y*S,w*S,y*S),fill='#dfdfdf',width=S)
        text(im,(w-2,y-7),str(v),12,fill='#777777',anchor='rm',bold=False)
    for xleft,xright,v,color in bars:
        left=xleft-x0; right=xright-x0; y=base-(base-top)*v/100
        dr.rectangle((left*S,y*S,right*S,(base+2)*S),fill=color,outline='#111111',width=3*S)
        label=f'{v:.1f}%' if v%1 else f'{v:.0f}%'
        text(im,((left+right)/2,y-14),label,21)
    paste_region(box,im)

def note(box, lines, size=15):
    w,h=box[2]-box[0],box[3]-box[1]
    im=Image.new('RGB',(w*S,h*S),'#fdfdfd')
    for j,line in enumerate(lines):
        text(im,(w/2, (j+0.5)*h/len(lines)),line,size,fill='#555555',bold=False)
    paste_region(box,im)

def horizontal():
    box=(1088,113,1494,417)
    w,h=box[2]-box[0],box[3]-box[1]
    im=Image.new('RGB',(w*S,h*S),'#fdfdfd')
    dr=ImageDraw.Draw(im)
    scale=5.55
    control=odcv('2026-10-06-odcv-qwen36-0-plain')
    baseline=control*scale
    text(im,(baseline,14),f'baseline {control:g}%',22,fill='#666666')
    for y in range(32,h,15):
        dr.line((baseline*S,y*S,baseline*S,min(y+8,h)*S),fill='#888888',width=2*S)
    mapping=[
        (149,'2026-10-06-odcv-qwen36-0-da-15',''),
        (191,'2026-09-29-odcv-qwen36-0-da-no-t6-15',''),
        (233,'2026-10-07-odcv-qwen36-0-da-low-15',''),
        (273,None,''),
        (314,'2026-10-08-odcv-qwen36-0-da-grok-15',''),
        (355,'2026-10-07-odcv-qwen36-0-nonmoral-original-15',''),
        (395,DELIBERATION_RUN,''),
    ]
    for global_y,name,suffix in mapping:
        cy=global_y-box[1]
        if name is None:
            # User explicitly asked to leave this unresolved row alone for now.
            crop=ORIGINAL.crop((1088,259,1157,290)).resize((69*S,31*S))
            im.paste(crop,(0,(259-box[1])*S))
            text(im,(87,cy),'pending verification',18,fill='#777777',anchor='lm',bold=False)
            continue
        value=odcv(name)
        end=value*scale
        dr.rectangle((0,(cy-12)*S,end*S,(cy+12)*S),fill=PURPLE,outline='#751bc1',width=2*S)
        text(im,(end+9,cy),f'{value:.1f}%{suffix}',23,anchor='lm')
    paste_region(box,im)
    OUT.paste(ORIGINAL.crop((1088,259,1157,290)),(1088,259))

def main():
    legend()
    names=['2026-10-06-odcv-qwen36-0-da-15','2026-10-06-odcv-qwen36-0-plain',
           '2026-10-08-am-qwen36-0-da-15','2026-10-08-am-qwen36-0-plain']
    values=[odcv(names[0]),odcv(names[1]),am(names[2]),am(names[3])]
    vertical((493,645,781,892),[(497,538,values[0],PURPLE),(542,599,values[1],GREY),
                              (640,683,values[2],PURPLE),(689,746,values[3],GREY)])
    hnames=['2026-10-07-hospital-self-sacrificial-qwen36-0-da-15',
            '2026-10-11-hospital-self-sacrificial-qwen36-0-plain']
    hvalues=[hospital(n) for n in hnames]
    vertical((1247,645,1475,892),[(1278,1355,hvalues[0],PURPLE),(1366,1442,hvalues[1],GREY)])
    horizontal()
    path=ROOT/'2026-10-12_figure1_clean.png'
    OUT.save(path)
    mask=np.zeros((1024,1536),bool)
    for x0,y0,x1,y1 in REGIONS: mask[y0:y1,x0:x1]=True
    changed=np.any(np.array(ORIGINAL)!=np.array(OUT),axis=2)
    assert not np.any(changed&~mask)
    audit={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'output':str(path),'edit_rectangles':REGIONS,'changed_pixels':int(changed.sum()),
           'changed_pixels_outside_regions':int((changed&~mask).sum()),
           'values':dict(zip(names+hnames,values+hvalues))}
    (ROOT/'pixel_audit_clean.json').write_text(json.dumps(audit,indent=2))
    selected=[('ODCV / DA',names[0]),('ODCV / plain',names[1]),
              ('Agentic Misalignment / DA',names[2]),('Agentic Misalignment / plain',names[3]),
              ('Colosseum / DA',hnames[0]),('Colosseum / plain',hnames[1]),
              ('Minus trait 6','2026-09-29-odcv-qwen36-0-da-no-t6-15'),
              ('Minus high stakes','2026-10-07-odcv-qwen36-0-da-low-15'),
              ('Grok teacher','2026-10-08-odcv-qwen36-0-da-grok-15'),
              ('Minus moral ambiguity (nonmoral advice)','2026-10-07-odcv-qwen36-0-nonmoral-original-15'),
              ('Minus deliberation',DELIBERATION_RUN)]
    report=['# Figure 1 correction — 2026-10-12','',
        'Original PNG is preserved. All pixels outside the four listed legend/chart rectangles are identical. '
        'The original unrelated-trait bar pixels are also preserved. Its length is unverified and must not be read as a measured result.',
        '', '| Panel / row | Percent | Evaluation artifact (pinned revision) |', '|---|---:|---|']
    for label,name in selected:
        rec=RUNS[name]
        rate=am(name) if '-am-' in name else hospital(name) if '-hospital-' in name else odcv(name)
        url=f"https://huggingface.co/datasets/{rec['repo']}/tree/{rec['revision']}"
        report.append(f'| {label} | {rate:.1f}% | [{name}]({url}) |')
    report += ['', '## Reading the figure', '',
        '- Baseline means the plain SFT control, not the base Qwen model or the older nosynth control. '
        'Both single-agent benchmarks use DA model ce773a6b87db66bd569dd13071ec5e27201462f1 and plain model '
        '1565d4f3356aa1b3bea1c503700164faa240ef3a; names and full metadata are saved in pinned_sources.json.',
        '- ODCV percentages are the published overall misalignment rates (240 rollouts per current run). '
        'Agentic Misalignment uses all 18 conditions: DA 12/180, plain 55/180, zero missing classifications.',
        '- Colosseum uses pair_sabotaged in the self_sacrificial condition: DA 10/10, plain 15/15. '
        'The DA run uses 4096 max tokens, no carried history, mandatory plan posts and the older table2 peer. '
        'The plain run uses 8192 tokens, carried history, optional plan posts and the nosynth peer. '
        'Seeds are 1–10 versus 16–30. These are descriptive results, not a controlled treatment contrast.',
        '- The minus-individual-traits row is minus trait 6, as confirmed by the user. Its latest run is '
        'September 29 and predates the October plain blend and history-handling change. '
        'Its contrast with the current baseline is not a matched ablation. On-image notes were removed at the user\'s request.',
        '- The unrelated-trait row is intentionally unresolved at the user\'s request. The curiosity experiment '
        'was investigated but was NOT substituted into this row.',
        '- Grok teacher means the all-Grok da-grok arm (12.9%), not the responder-only swap (17.9%).',
        '- Minus moral ambiguity is mapped to nonmoral-original-15, which replaces ethical difficult advice '
        'with nonmoral advice. This is a broader content change, not a pure ambiguity manipulation.',
        f'- Minus deliberation uses {DELIBERATION_RUN}. Emptythink removes training reasoning; response-only '
        'retains reasoning as context but masks its training loss; rewritten reasoning removes open weighing. '
        'The user selected the rewritten-reasoning stated-harm variant (27.9%). It predates the October 5 '
        'plain blend and history-handling change. '
        'These are distinct interventions and must not be interchanged.',
        '- Removed the no-effect and returns-to-baseline annotations, per the user\'s instruction. '
        'A visual cluster of point estimates does not establish equivalence. No new statistical claim is added.',
        '- Purple represents DA and DA variants; grey represents plain control. Vertical bar heights share '
        'a 0–100% scale. Horizontal bar lengths share a zero-based linear scale with the plain baseline at 45%.',
        '', '## Pixel verification', '', '```json',json.dumps(audit,indent=2),'```']
    (ROOT/'2026-10-12_figure1_clean_results.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
