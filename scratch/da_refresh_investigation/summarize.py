# ABOUTME: Aggregates blinded row judgments without treating model labels as ground truth.
# ABOUTME: Run after all audit stages; reports counts, disagreement, costs and a dated figure.
import collections
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scratch.da_refresh_investigation.judge import OUT,read,save
from src.naming import artifact_name,figure_path

def main():
    scores={}
    for stage in ['primary','primary_recovery','secondary','challenge','challenge_recovery']:
        scores[stage]=[read(p) for p in (OUT/'judgments'/stage).glob('*.json')]
    primary={r['row_id']:r for r in scores['primary'] if r['status']=='ok'}
    primary.update({r['row_id']:r for r in scores['primary_recovery'] if r['status']=='ok'})
    challenge={r['row_id']:r for r in scores['challenge'] if r['status']=='ok'}
    challenge.update({r['row_id']:r for r in scores['challenge_recovery'] if r['status']=='ok'})
    fields=['deliberation','moral','prompt_stakes','live_conflict','cot_fabrication','answer_fabrication','ai_scenario','assistant_self_governance','shortcut_appeal_engaged','residual_cost_acknowledged','procedural_deference_over_context']
    output={'counts':{},'warnings':['These are automated measurements, not ground truth. Primary judge has ceiling scores and missed manually confirmed factual defects. Do not use its zero fabrication rate as evidence of no fabrications.'],'errors':[{'row':r['row_id'],'stage':r['stage'],'error':r.get('error')} for rows in scores.values() for r in rows if r['status']!='ok']}
    for stage,rows in [('primary',list(primary.values())),('secondary',scores['secondary']),('challenge',list(challenge.values()))]:
        output['counts'][stage]={}
        for arm in ['old','new']:
            kept=[r for r in rows if r['row_id'].startswith(arm) and r['status']=='ok']
            counts={k:dict(collections.Counter(r['verdict'][k] for r in kept)) for k in fields}
            output['counts'][stage][arm]={'n':len(kept),'scores':counts,'cot_clear_flags':sum(r['verdict']['cot_fabrication']>=2 for r in kept),'any_clear_flags':sum(max(r['verdict']['cot_fabrication'],r['verdict']['answer_fabrication'])>=2 for r in kept),'high_stakes':sum(r['verdict']['prompt_stakes']>=2 for r in kept),'ai_central':sum(r['verdict']['ai_scenario']==2 for r in kept),'quote_error_rows':sum(bool(r['quote_errors']) for r in kept)}
    paired=[(primary[r['row_id']],r) for r in scores['secondary'] if r['status']=='ok' and r['row_id'] in primary]
    output['secondary_agreement']={'n':len(paired),'exact_by_field':{k:sum(a['verdict'][k]==b['verdict'][k] for a,b in paired) for k in fields}}
    budget=read(OUT/'audit_budget.json');calls=budget['calls']
    output['cost']={'cap_usd':budget['limit'],'api_reported_usd':sum(v.get('charged_usd',0) for v in calls.values()),'retained_unknown_reservations_usd':sum(v['reserved_usd'] for v in calls.values() if 'charged_usd' not in v),'requests':len(calls)}
    assert output['cost']['api_reported_usd']+output['cost']['retained_unknown_reservations_usd']<=budget['limit']
    save(OUT/'audit_summary.json',output)
    structure=read(OUT/'structure.json')['summary']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),constrained_layout=True)
    width=.35
    values=[[229/628*100,233/628*100,66/628*100],[50/617*100,67/617*100,0]]
    for j,(name,hatch) in enumerate([('Previous DA-15',''),('Refreshed DA-15','///')]):
        axes[0].bar([i+(j-.5)*width for i in range(3)],values[j],width=width,color='#8061af',alpha=1 if j==0 else .55,hatch=hatch,label=name)
    axes[0].set_xticks(range(3),['AI in user text','AI in system text','Deployment wording'])
    axes[0].set(ylabel='Selected DA rows (%)',ylim=(0,50),title='AI-specific context dropped')
    axes[0].legend(fontsize=8)
    vals=[structure[k]['cot_tokens']/1000 for k in ['old','new']]
    axes[1].bar([0,1],vals,color='#8061af',width=.55)
    for i,v in enumerate(vals):axes[1].text(i,v+5,f'{v:.1f}k',ha='center')
    axes[1].set_xticks([0,1],['Previous DA-15','Refreshed DA-15'])
    axes[1].set(ylabel='Supervised CoT tokens (thousands)',ylim=(0,440),title='CoT supervision stayed intact')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Dataset investigation: exact training rows, pinned revisions')
    folder=ROOT/'output/figures'/artifact_name('da-refresh-investigation');folder.mkdir(parents=True,exist_ok=True)
    png=figure_path(folder,'da-refresh-investigation',ext='png');svg=figure_path(folder,'da-refresh-investigation',ext='svg')
    fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    png.with_name(png.stem+'_results.md').write_text('# DA refresh dataset investigation\n\nExact lexical counts, not semantic labels; deployment wording is a proxy.\n\nAI in user text: 229/628 vs 50/617. AI in system text: 233/628 vs 67/617.\nSystem deployed/embedded/integrated wording: 66/628 vs 0/617.\nSupervised CoT tokens: 369,919 vs 373,973.\n\nSee the investigation report for manual and blinded semantic audits and causal limitations.\n',encoding='utf-8')
    save(OUT/'figure_receipt.json',{'png':str(png),'svg':str(svg)})
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
