# ABOUTME: Grade retained valid outcomes from terminal smoke stages without any model calls.
# ABOUTME: Explicitly accounts for interrupted and unattempted selections; never changes original state.
import json
from pathlib import Path
from omegaconf import OmegaConf
from scratch.gptoss_swe.openai_smoke import ROOT,TARGETS,save
from scratch.gptoss_swe.run import grade_predictions


def main():
    for arm,target in TARGETS.items():
        root=ROOT/arm; all_tasks={};resolved=[];grade_total=0;terminal=True
        for stage in ('swe-recovered','swe-pending'):
            run=(root/stage).resolve()
            if not (run/'metadata/state.json').exists():continue
            state=json.loads((run/'metadata/state.json').read_text())
            if any(t['status']=='running' for t in state['tasks'].values()):
                terminal=False
                continue
            manifest=json.loads((run/'metadata/manifest.json').read_text())
            assert manifest['target']==target
            for iid,t in state['tasks'].items():
                if t['attempts']:
                    assert iid not in all_tasks or not all_tasks[iid]['attempts']
                    all_tasks[iid]=t | dict(output_root=str(run))
                elif iid not in all_tasks: all_tasks[iid]=t | dict(output_root=str(run))
            preds={i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
            result=run/'results/qualification.json'
            if not result.exists():
                result=run/'results/retained-qualification.json'
                if not result.exists():
                    cfg=OmegaConf.create(manifest['config'])
                    grade=grade_predictions(cfg,run,preds,cfg.campaign+'-retained') if preds else dict(n_graded=0,n_resolved=0,resolved_ids=[])
                    save(result,grade | dict(checkpoint=target,n_valid_rollouts=len(preds),n_total=len(state['tasks']),partial=True))
            grade=json.loads(result.read_text());assert grade['n_graded']==len(preds)
            grade_total+=grade['n_graded'];resolved+=grade['resolved_ids']
        if not terminal: continue
        assert len(all_tasks)==10
        valid={i:t for i,t in all_tasks.items() if t['status']=='valid'}
        assert len(valid)==grade_total and len(resolved)==len(set(resolved))
        save(root/'swe-summary.json',dict(checkpoint=target,n_selected=10,n_valid_rollouts=len(valid),n_graded=grade_total,n_resolved=len(resolved),resolved_ids=resolved,n_interrupted=sum(t['status']=='invalid' for t in all_tasks.values()),n_not_started=sum(t['status']=='pending' for t in all_tasks.values()),task_statuses=all_tasks,not_a_full_benchmark_score=True))
        print(arm,'graded',grade_total,'resolved',len(resolved),flush=True)

if __name__=='__main__': main()
