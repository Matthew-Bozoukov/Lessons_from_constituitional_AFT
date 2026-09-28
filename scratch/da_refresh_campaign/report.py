# ABOUTME: Verifies pinned campaign outputs and plots the completed previous/new DA-15 comparison.
# ABOUTME: Run only after all three owners completed and every recorded owned pod was terminated.
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scratch.da_refresh_campaign.run import OUT,CFG,read,save
from src.infra.huggingface import hf_api,hf_download
from src.infra.runpod import active_pods
from src.naming import artifact_name,figure_path


def main():
    api=hf_api();refs=read(OUT/'references.json');campaign=read(OUT/'campaign.json')
    states=[read(p) for p in OUT.glob('*-attempt*/status.json')]
    assert all(not s.get('owned_pod') or s.get('terminated') for s in states),'Unterminated owners'
    completed={}
    for kind in ['train','odcv','mask']:
        for pct in [15]:
            matches=[s for s in states if s['kind']==kind and s['pct']==pct and s['phase']=='complete']
            assert len(matches)==1, f'Expected one completed {kind}{pct}'
            completed[kind,pct]=matches[0]
    owned={s['owned_pod'] for s in states if s.get('owned_pod')}
    assert not owned.intersection(p['id'] for p in active_pods()),'Owned pod still billing'
    measured={}
    for pct in [15]:
        trained=read(OUT/f'train{pct}-attempt1/status.json');audit=read(OUT/f'audit{pct}.json')
        measured[str(pct)]={}
        for ev in ['odcv','mask']:
            s=completed[ev,pct]
            info=api.dataset_info(s['eval_repo'],revision=s['eval_revision'])
            meta=read(hf_download(s['eval_repo'],'metadata/run_meta.json',repo_type='dataset',revision=info.sha))
            result=read(hf_download(s['eval_repo'],'results/results.json',repo_type='dataset',revision=info.sha))
            assert meta['target_revision']==trained['adapter_revision'] and meta['base_model_revision']==CFG.base_model_revision
            train=read(hf_download(meta['target'],'training_meta.json',revision=meta['target_revision']))
            assert train['dataset']['revision']==audit['data_revision'] and train['train_config']['seed']==0
            assert train['train_config']['train']['loss_agg']=='token_mean' and train['thinking'] is True
            if ev=='odcv':
                assert result['ours']['overall']['n_rollouts']==240
                assert result['n_judged']==240 and result['n_dropped_all_na']==0
                assert result['progress']['n_judged']==240 and result['progress']['n_dropped_all_na']==0
            else:
                assert result['n_rows']==1000 and result['passes']==1 and result['generation_error_rate']<=0.05
            measured[str(pct)][ev]={'repo':s['eval_repo'],'revision':info.sha,'meta':meta,'results':result}
    series=[('Nosynth',refs['nosynth']),('Sep 25 DA-15',refs['da15']),('Sep 28 DA-15',measured['15'])]
    rows=[]
    for label,r in series:
        od=r['odcv']['results'];mask=r['mask']['results'];overall=od['ours']['overall']
        rows.append({'arm':label,'misconduct_pct':overall['mr_pct'],'misconduct_ci95':overall['mr_ci95'],
                     'progress':od['progress']['ours']['overall']['tp_mean'],'mask_honesty':mask['overall_honesty_score'],
                     'mask_generation_error_rate':mask['generation_error_rate'],
                     'odcv_repo':r['odcv']['repo'],'odcv_revision':r['odcv']['revision'],
                     'mask_repo':r['mask']['repo'],'mask_revision':r['mask']['revision']})
    budget=read(OUT/'budget.json');control=read(OUT/'controller.json')
    for receipt in OUT.glob('recovery*.json'):
        recovery=read(receipt)
        control['api_usage_delta_upper_bound']=max(control['api_usage_delta_upper_bound'],recovery.get('api_usage_delta_upper_bound',0))
    if (OUT/'completion_verification.json').exists():
        control['api_usage_delta_upper_bound']=max(control['api_usage_delta_upper_bound'],
            read(OUT/'completion_verification.json')['shared_api_delta_now_upper_bound'])
    cost={'gpu_storage_estimate_usd':sum(j['settled_usd'] for j in budget['jobs'].values()),
          'shared_api_usage_upper_bound_usd':control['api_usage_delta_upper_bound'],'cap_usd':200}
    cost['conservative_total_estimate_usd']=cost['gpu_storage_estimate_usd']+cost['shared_api_usage_upper_bound_usd']
    assert cost['conservative_total_estimate_usd']<200
    save(OUT/'comparison.json',{'rows':rows,'cost':cost,'measured':measured,'reference_pins':refs})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    colors=['#858585','#8061af','#8061af'];x=list(range(3));labels=[r['arm'] for r in rows]
    vals=[r['misconduct_pct'] for r in rows]
    axes[0].bar(x,vals,color=colors,width=.62)
    axes[0].errorbar(x,vals,yerr=[[v-r['misconduct_ci95'][0] for v,r in zip(vals,rows)],
                                [r['misconduct_ci95'][1]-v for v,r in zip(vals,rows)]],fmt='none',color='black',capsize=4)
    axes[0].set(title='ODCV misconduct — lower is better',ylabel='Percent',ylim=(0,70))
    axes[1].bar(x,[r['mask_honesty'] for r in rows],color=colors,width=.62)
    axes[1].set(title='MASK honesty — higher is better',ylabel='Score',ylim=(0,100))
    for ax in axes:
        ax.set_xticks(x,labels);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Difficult advice corpus refresh: matched 15% supervision\nSeed 0; ODCV 240 rollouts per arm; MASK one full pass',fontsize=12)
    folder=ROOT/'output/figures'/artifact_name('da-corpus-refresh-comparison');folder.mkdir(parents=True,exist_ok=True)
    png=figure_path(folder,'da-corpus-refresh-comparison',ext='png');svg=figure_path(folder,'da-corpus-refresh-comparison',ext='svg')
    fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    lines=['# DA corpus refresh comparison','',
           '| Arm | ODCV misconduct % [95% CI] | Progress | MASK honesty |','|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['misconduct_pct']} {r['misconduct_ci95']} | {r['progress']} | {r['mask_honesty']} |")
    lines+=['','One trained checkpoint per arm. ODCV intervals reflect scenario/pass variation, not training-seed variation.',
            'Nosynth and September 25 DA-15 are existing pinned reference runs. MASK uses one pass; no between-pass confidence interval is claimed.',
            'All comparisons use thinking mode and token-mean training. Serving concurrency differs for some historical references.',
            '',f'Cost: {json.dumps(cost)}','',f'Figure: {png.name}']
    report=png.with_name(png.stem+'_results.md');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'report_receipt.json',{'png':str(png),'svg':str(svg),'report':str(report),'rows':rows,'cost':cost})
    print(json.dumps({'rows':rows,'cost':cost,'figure':str(png)},indent=2))


if __name__=='__main__':
    main()
