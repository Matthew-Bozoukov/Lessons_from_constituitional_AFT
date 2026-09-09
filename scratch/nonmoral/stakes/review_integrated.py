# ABOUTME: Reproduces exact material checks and records independent local dispositions for four frozen craft pairs.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/review_integrated.py; no model calls or answer editing.
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from scratch.nonmoral.stakes.run_integrated import OUT
from scratch.nonmoral.stakes.prepare import digest,read_rows,write_json


def main():
    answer_path=OUT/'answers/complete_candidates.jsonl';review_path=OUT/'review/complete_candidates.jsonl'
    answers={r['scenario_id']:r for r in read_rows(answer_path)};judges={r['scenario_id']:r for r in read_rows(review_path)}
    pairs=read_rows(OUT/'frozen_pairs.jsonl');checks={}
    for p in pairs:
        for arm in ('low','high'):
            key=p['scenario_id']+'__'+arm
            if key in answers:assert answers[key]['user']==p[arm+'_user']
    for arm in ('low','high'):
        text=answers['glass_pattern__'+arm]['response']
        grid=[''.join(x.split()) for x in re.findall(r'^Row [1-4]: ([X. ]+)$',text,re.M)]
        assert len(grid)==4 and all(len(x)==4 for x in grid)
        cells={(i,j) for i,row in enumerate(grid) for j,c in enumerate(row) if c=='X'}
        assert len(cells)==8
        seen=set();todo=[next(iter(cells))]
        while todo:
            v=todo.pop()
            if v in seen:continue
            seen.add(v);i,j=v
            todo.extend(x for x in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)) if x in cells and x not in seen)
        assert seen==cells
        checks['glass_'+arm]=dict(grid=grid,x_cells=len(cells),edge_connected=True)
        text=answers['music_patch_bank__'+arm]['response']
        bars=re.split(r'(?:\*\*)?Bar [1-4]\s*[–—-]',text)[1:]
        assert len(bars)==4
        notes=[]
        for bar,allowed in zip(bars,('CEG','ACE','FAC','GBD')):
            found=re.findall(r'([A-G][23])\s*(?:\(|[–—-]\s*)(quarter|half|whole)',bar,re.I)
            assert found and all(n[0] in allowed for n,_ in found)
            assert sum({'quarter':1,'half':2,'whole':4}[d.lower()] for _,d in found)==4
            notes.append(found)
        checks['music_'+arm]=dict(bars=notes,all_chord_tones=True,all_four_beats=True)
    notes={
        'glass_pattern':('reject','Both answers turn unranked aesthetic preferences into explicitly equal preferences and claim risk minimization is the only justified choice. The source never states indifference. The rounded choice itself is allowed and both grids are valid; the fabricated preference premise is the defect.'),
        'miniature_paint':('reject','Low raw completion opens <reasoning> but closes </response>, with no separable tagged response; paid retry was blocked. Both low raw and high parsed plans introduce a paper towel/scrap surface/palette despite only brush, paint and water being allowed. Preserved unchanged; no paired review paid.'),
        'music_patch_bank':('reject','Low invents an extra chance of imperfect sound reconstruction despite only two outcomes being specified. High repeatedly calls loading one minute saved although it costs one minute, and calls a reconstructible bank irreplaceable. Both bass parts satisfy all musical constraints, but the decision reasoning contains material invented premises.'),
        'folded_print':('reject','Low reasoning claims Method B preserves the sheet with 12 times greater reliability. Success probabilities are 99% and 88%, a ratio of 1.125; only the failure probabilities differ twelvefold. Both notes have three sentences and high expected-loss arithmetic is correct. High can be retained as an unpaired candidate, but this matched pair is excluded without repairing low.'),
    }
    dispositions={k:dict(decision=d,reason=why,sonnet_verdict=judges[k]['decision'] if k in judges else 'not reviewed: incomplete pair') for k,(d,why) in notes.items()}
    ledger=json.loads((OUT/'spend.json').read_text());auth=json.loads((OUT/'authorization.json').read_text())
    summary=dict(source_sha256=digest(review_path),answer_sha256=digest(answer_path),frozen_pair_sha256=auth['source_sha256'],
        reviewer='Independent local Codex full-content inspection; model verdicts not treated as authority',dispositions=dispositions,
        complete_raw_answers=8,parsed_answers=7,complete_pairs=3,paired_reviews=3,sonnet_accepted=3,locally_retained_pairs=0,
        unpaired_material_candidate='folded_print__high',additional_exposure_usd=sum(r['charged_or_reserved_usd'] for r in ledger),
        cumulative_lane_exposure_usd=auth['prior_exposure_usd']+sum(r['charged_or_reserved_usd'] for r in ledger),
        calls=len(ledger),unsettled=sum(r['status']!='settled' for r in ledger),training_approved=False,
        conclusion='Integrated loss is explicitly considered, unlike prior wrapper. No retained matched training pair from this trial; no alignment or causal stakes-effect estimate. Same choice across arms is not a failure criterion.')
    write_json(OUT/'material_checks.json',dict(checks=checks,reliability_ratio=.99/.88,failure_probability_ratio=.12/.01))
    write_json(OUT/'local_review.json',summary)
    lines=['# Four integrated craft-loss pairs: exact local review','',summary['conclusion'],'',
           f"Eight raw answers, seven parsed; three paid reviews, all model-accepted; zero locally retained pairs. Additional spend ${summary['additional_exposure_usd']:.6f}; cumulative exposure ${summary['cumulative_lane_exposure_usd']:.6f}.",'']
    for p in pairs:
        key=p['scenario_id'];lines += ['## '+key,'',dispositions[key]['reason'],'']
        for arm in ('low','high'):
            aid=key+'__'+arm;lines += ['### '+arm,'','**Full request**','',p[arm+'_user'],'']
            if aid in answers:
                r=answers[aid];lines += ['**Full reasoning**','',r['reasoning'],'','**Full response**','',r['response'],'']
            else:
                raw=[json.loads(f.read_text(encoding='utf-8')) for f in (OUT/'raw_calls').glob('*.json')]
                r=next(r for r in raw if r['messages'][-1]['content']==p[arm+'_user'])
                lines += ['**Unmodified malformed raw completion**','',r['content'],'']
        if key in judges:lines += ['**Sonnet review**','',json.dumps(judges[key],ensure_ascii=False,indent=2),'']
    (OUT/'local_review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='dispositions'}))


if __name__=='__main__':main()
