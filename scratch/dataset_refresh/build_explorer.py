# ABOUTME: Build a self-contained full corpus browser plus a small inline category-spanning preview from pinned bytes.
# ABOUTME: Preserve exact conversation strings, distinguish heuristic markers from curated notes, and never send inference calls.
import argparse
import hashlib
import json
import re
from pathlib import Path
from omegaconf import OmegaConf


def build(config):
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    census = [json.loads(x) for x in Path(cfg['census']).read_text(encoding='utf-8').splitlines()]
    flags = json.loads(Path(cfg['flags']).read_text(encoding='utf-8'))
    if isinstance(flags, dict): flags = flags['flags']
    rows, sources, applied = [], {}, set()
    for arm, source in cfg['sources'].items():
        raw = Path(source['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256'], 'Source SHA mismatch'
        data = [json.loads(x) for x in raw.decode().splitlines()]
        assert len(data) == 716
        stats = {r['scenario_id']: r for r in census if r['corpus'] == source['census_name']}
        sources[arm] = {k: source[k] for k in ('repo', 'revision', 'sha256')}
        sources[arm]['url'] = 'https://huggingface.co/datasets/'+source['repo']+'/tree/'+source['revision']
        for index, row in enumerate(data):
            meta = row['metadata']; key = meta['scenario_id']; stat = stats[key]
            messages = row['messages']; system, user, assistant = messages
            assert [x['role'] for x in messages] == ['system','user','assistant']
            notes = []
            for fi, f in enumerate(flags):
                if {'nonmoral':'non'}.get(f['arm'],f['arm']) == arm and f.get('row_id') == key:
                    assert f['row_index']==index
                    quote = f.get('quote', '')
                    assert not quote or any(quote in str(m.get(k,'')) for m in messages for k in ('content','reasoning_content')), ('Nonverbatim flag quote',fi)
                    assert f['severity'] in ('preference','comparability','possible_error')
                    notes.append({k:f.get(k,'') for k in ('concern','quote','severity','rationale')}); applied.add(fi)
            words = stat['words']; markers = []
            if arm=='low' and re.search(r'\b(?:AI|assistant|chatbot|automated|algorithm)\b',user['content'],re.I): markers.append('ai')
            if words['user']>400: markers.append('prompt')
            if words['response']>650: markers.append('answer')
            if len(re.findall(r'\b\d+(?:\.\d+)?\b',user['content']))>=6: markers.append('numbers')
            origin=meta.get('origin', {})
            route = 'Earlier refresh review route'
            if arm=='non' and key.startswith('offline_'): route='Independent offline full read; see published provenance'
            rows.append(dict(id=key,shortId=meta.get('original_scenario_id',key),arm=arm,index=index,rank=len(rows),family=stat['analyst_family'],domain=meta['domain'],traitId=meta['trait_id'],trait=meta['trait_name'],system=system['content'],user=user['content'],reasoning=assistant.get('reasoning_content',''),answer=assistant['content'],words=words,tokens=stat['tokens']['supervised_tokens'],notes=notes,markers=markers,route=route))
    assert len(applied)==len(flags), 'Unbound curated flags'
    assert len({r['id'] for r in rows})==1432
    template=Path(cfg['template']).read_text(encoding='utf-8')
    def render(items,sample):
        payload=json.dumps(dict(rows=items,sources=sources,sample=sample),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        return template.replace('__PAYLOAD__',payload)
    # Curated cases plus every documented family and every trait in both arms.
    selected={r['id'] for r in rows if r['notes']}
    for field in ('family','traitId'):
        for group in sorted({(r['arm'],r[field]) for r in rows}):
            if not any((r['arm'],r[field])==group and r['id'] in selected for r in rows):
                options=[r for r in rows if (r['arm'],r[field])==group]
                selected.add(sorted(options,key=lambda r:r['words']['user'])[len(options)//2]['id'])
    for arm in ('low','non'):
        options=[r for r in rows if r['arm']==arm]
        for i in range(0,len(options),max(1,len(options)//20)):
            if len(selected)>=cfg['sample_target']:break
            selected.add(options[i]['id'])
    sample=[r for r in rows if r['id'] in selected]
    inline=render(sample,True); assert len(inline.encode())<1_000_000, 'Inline preview too large'
    path=Path(cfg['inline_output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(inline,encoding='utf-8')
    full=render(rows,False)
    # Standalone artifact: minimal host-equivalent controls, responsive theme and no network dependencies.
    css='''<style>:root{color-scheme:light dark;--background:light-dark(#faf9f6,#171a1c);--foreground:light-dark(#202a30,#e5e9ed);--border:light-dark(#cbd1d5,#49535b);--primary:light-dark(#243b49,#c2dbe9);--primary-foreground:light-dark(#fff,#152c37)}body{background:var(--background);color:var(--foreground);font:16px/1.6 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 24px}h2,h3,strong{font-weight:500}h2{margin:0}h3{margin-bottom:6px}button,input,select,textarea{font:inherit;max-width:100%;box-sizing:border-box}button,summary{cursor:pointer}.btn{padding:8px 12px;border:1px solid var(--border);border-radius:6px;background:var(--background);color:var(--foreground)}.btn[aria-pressed=true]{background:var(--primary);color:var(--primary-foreground)}.btn:disabled{opacity:.4;cursor:default}.form-control,.form-select{display:block;width:100%;padding:8px;background:var(--background);color:var(--foreground);border:1px solid var(--border);border-radius:5px}.viz-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}.viz-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}.form-label{display:block}.text-small{font-size:13px}.viz-badge{padding:3px 8px;background:color-mix(in srgb,var(--foreground) 8%,transparent);border-radius:4px}a{color:light-dark(#145d84,#9ad4f4)}textarea{width:100%}details>summary{padding:8px 0}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--foreground);outline-offset:3px}@media(max-width:640px){body{padding:0 14px}.viz-controls{grid-template-columns:1fr}}</style>'''
    standalone='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset reading room</title>'+css+'</head><body>'+full+'</body></html>'
    (out/'2026-09-15_dataset_reading_room.html').write_text(standalone,encoding='utf-8')
    manifest=dict(sources=sources,rows=len(rows),sample_rows=len(sample),sample_counts={a:sum(r['arm']==a for r in sample) for a in sources},curated_notes=len(flags),flagged_rows=sum(bool(r['notes']) for r in rows),inline_bytes=len(inline.encode()),full_bytes=len(standalone.encode()),scope='Read-only viewer, no inference or dataset edits',selection='Purposive: curated entries plus every family and every principle/craft tension; deterministic additional entries. Not a prevalence sample.')
    (out/'viewer_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);build(parser.parse_args().config)
