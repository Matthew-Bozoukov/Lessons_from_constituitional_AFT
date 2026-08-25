import json, re, random, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D=Path(r"C:/Users/nikak/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-14-table2-9284-difficult-advice-716-train/snapshots/7ec7491f83478419585bb23df2fcf91e6f301d97")
rows=[json.loads(l) for l in open(D/'t2_9284_da716_10k.jsonl',encoding='utf-8')]
da=[r for r in rows if r.get('source')=='difficult_advice_v2']
# encoding sanity: are there replacement chars / mojibake in the published text?
bad = [r for r in da if '\ufffd' in r['text']]
print(f"rows containing U+FFFD replacement char: {len(bad)}/{len(da)}")
import collections
c=collections.Counter(ch for r in da for ch in r['text'] if ord(ch)>0x2000 and ord(ch)<0x2200)
print("punctuation chars present:", {hex(ord(k)):v for k,v in c.most_common(8)})
print()
random.seed(1337)
picked = random.sample(sorted(da, key=lambda r: r['scenario_id']), 5)
TURN=re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.S)
THINK=re.compile(r"<think>\n(.*?)</think>\n*", re.S)
out=[]
for r in picked:
    d={'scenario_id':r['scenario_id'],'trait_id':r['trait_id']}
    for role,body in TURN.findall(r['text']):
        if role=='assistant':
            m=THINK.search(body); d['think']=m.group(1).strip(); d['answer']=body[m.end():].strip()
        else: d[role]=body.strip()
    out.append(d)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
for i,d in enumerate(out):
    print(f"[{i}] {d['scenario_id']}  trait={d['trait_id']}  sys={len(d['system'])}c user={len(d['user'])}c think={len(d['think'])}c answer={len(d['answer'])}c")
