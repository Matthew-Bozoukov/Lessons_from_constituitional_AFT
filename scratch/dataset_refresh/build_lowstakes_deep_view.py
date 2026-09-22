# ABOUTME: Build quotation-validated inline and full-corpus explorers from pinned low-stakes texts.
# ABOUTME: All measurements are offline; distinguishes manual prompt judgments from model labels and literal screens.
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

from omegaconf import OmegaConf


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main(cfg):
    out=Path(cfg.output)
    rows=read(out/'records.json')
    lookup={r['id']:r for r in rows}
    metrics=read(out/'offline_analysis.json')
    review=OmegaConf.to_container(OmegaConf.load(cfg.review),resolve=True)
    assert len(review['old_flagged'])==67
    assert set(review['old_flagged'])=={r['id'] for r in rows if r['arm']=='old' and r['historical_stakes'] in [2,3]}
    for arm,group in [('old','old_additional'),('new','new_screen')]:
        expected={r['id'] for r in rows if r['arm']==arm and (arm=='new' or r['historical_stakes'] not in [2,3]) and re.search(cfg.risk_pattern,r['text']['user'],re.I)}
        assert expected==set(review[group]),(group,expected ^ set(review[group]))
    annotations={}
    for group in ['old_flagged','old_additional','new_screen']:
        for key,value in review[group].items():
            assert len(value)==2 and value[0] in review['rubric'] and key in lookup
            annotations[key]={'group':group,'judgment':value[0],'reason':value[1]}
    selected=read(Path(cfg.source_records)/'reviewed_records.json')
    sample_ids={r['id'] for r in selected}
    included=set(annotations)|sample_ids
    family={}
    for arm in ['old','new']:
        for r in read(Path(cfg.source_records)/(arm+'_records.json')):
            domain=r['metadata'].get('setting','') if arm=='old' else r['domain']
            domain=domain.split(' -- ')[0] if arm=='old' else re.sub(r'^MECHANISM SLOT [A-C]\. BENIGN ACTIVITY: ','',domain).split(':')[0]
            family[r['id']]=domain
    pairs=read(cfg.notes)['pairs']
    pairs[0]['limit']='This benign old example shows that direct assistant restraint is possible at low stakes. Its existence cannot prove that it contributed to MR: the higher-stakes minority could still drive the aggregate effect. The answer also overclaims compensation options.'
    pairs[4]['observation']='The old dog-assessment prompt leaves physical danger uncertain; its answer adds a specific medical claim unsupported by the supplied facts. The new movie prompt explicitly permits Sunday review, yet both reasoning and final answer claim Saturday posting nearly removes it.'
    pairs[8]['limit']='Historical ratings flag 67 rows. Current unblinded prompt review finds 3 concrete safety cases, 25 broadly elevated cases, 31 borderline and 8 modest within those 67; a targeted screen finds 3 additional broadly elevated old cases. This is not an exhaustive count of all 716.'
    pairs[8]['action']='Test a matched-dose, minimally rewritten condition. Keep actor, pressure, alternatives, target trait, replay slots and recipe fixed; do not delete 67 rows or broadly regenerate the corpus.'
    for p in pairs:
        for arm in ['old','new']:
            for q in p[arm+'_quotes']:
                assert q['text'] in lookup[p[arm]]['text'][q['field']],(p['title'],q)
    records=[]
    for r in sorted(rows,key=lambda r:(r['arm']!='old',r['id'])):
        flags=[key for key,ids in metrics['arms'][r['arm']]['literal_ids'].items() if r['id'] in ids]
        records.append({k:r[k] for k in ['arm','id','trait','historical_stakes','text']} | {
            'domain':family[r['id']],'review':annotations.get(r['id']),
            'sample':r['id'] in sample_ids,'flags':flags})
    statistics={g:dict(Counter(v[0] for v in review[g].values())) for g in ['old_flagged','old_additional','new_screen']}
    statistics['all_reviewed_old']=dict(Counter(v['judgment'] for k,v in annotations.items() if lookup[k]['arm']=='old'))
    result={'method':review['method'],'rubric':review['rubric'],'counts':statistics,
        'reviews':annotations,'risk_pattern':cfg.risk_pattern,'paid_calls':0,
        'blinded_census':'not run; optional separate budget approval pending',
        'historical_label_census':metrics['arms']['old']['historical_stakes'],
        'sources':read(out/'manifest.json')['sources']}
    (out/'manual_review_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    base={'pairs':pairs,'counts':statistics,'rubric':review['rubric'],'metrics':{
        arm:{'literal_counts':m['literal_counts'],'word_stats':m['word_stats'],'nearest_tfidf':m['nearest_tfidf']} for arm,m in metrics['arms'].items()},
        'evaluation':metrics['evaluation'],'sources':result['sources']}
    template=Path(cfg.template).read_text(encoding='utf-8')
    assert template.count('<!-- AUDIT_DATA -->')==1
    for full,path in [(False,Path(cfg.inline)),(True,out/'2026-09-17_lowstakes_full_explorer_fragment.html')]:
        chosen=records if full else [r for r in records if r['id'] in included]
        payload=base|{'records':chosen,'full':full}
        content=template.replace('<!-- AUDIT_DATA -->',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
        if not full:
            assert len(content.encode())<1_000_000
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content,encoding='utf-8')
        print(json.dumps({'path':str(path),'bytes':len(content.encode()),'records':len(chosen)},ensure_ascii=False))
    spec=importlib.util.spec_from_file_location('audit_visualize_renderer',str(cfg.renderer))
    renderer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    (out/'preview.html').write_text(renderer.render(Path(cfg.inline)),encoding='utf-8')
    # The larger full-corpus companion is a standalone document, never an inline visual.
    full_fragment=(out/'2026-09-17_lowstakes_full_explorer_fragment.html').read_text(encoding='utf-8')
    (out/'2026-09-17_lowstakes_full_explorer.html').write_text(
        renderer._render_document(full_fragment,'Full low-stakes corpus comparison'),encoding='utf-8')
    (out/'run_meta.json').write_text(json.dumps({'analysis_date':'2026-09-17','sources':result['sources'],
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'review_sha256':hashlib.sha256(Path(cfg.review).read_bytes()).hexdigest(),
        'record_sha256':hashlib.sha256((out/'records.json').read_bytes()).hexdigest(),
        'config':OmegaConf.to_container(cfg,resolve=True),'paid_calls':0,'manual_review_count':len(annotations),
        'reasoning_sample_count':18,'targeted_prior_grave_count':2},indent=2),encoding='utf-8')
    print(json.dumps(statistics))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    main(OmegaConf.load(parser.parse_args().config))
