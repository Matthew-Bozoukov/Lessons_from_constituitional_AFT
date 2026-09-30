# ABOUTME: Count malformed JSON endings per ODCV call and per rollout, and compare matched prompt regimes.
# ABOUTME: Reads immutable published transcripts; diagnostic bracket deletion is never executed or used to rescore.
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from dotenv import load_dotenv
from huggingface_hub import snapshot_download, CommitOperationAdd
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.infra.huggingface import hf_api, hf_download


def classify(arguments):
    try:
        obj=json.loads(arguments)
    except json.JSONDecodeError as exc:
        extra=False
        ending=False
        if arguments[exc.pos:exc.pos+1]==']':
            try:
                candidate=json.loads(arguments[:exc.pos]+arguments[exc.pos+1:])
            except json.JSONDecodeError:
                pass
            else:
                extra=isinstance(candidate,dict)
                ending=extra and arguments[exc.pos+1:].strip() in ('','}')
        return {'invalid_json':True,'extra_square_bracket':extra,'square_bracket_ending':ending,
            'error':str(exc),'error_offset':exc.pos}
    return {'invalid_json':False,'extra_square_bracket':False,'square_bracket_ending':False,
        'non_object':not isinstance(obj,dict)}


def inspect_transcript(text):
    calls=[]
    assistant_turn=0
    for chunk in re.split(r'^== Step \d+ ==\n',text,flags=re.M)[1:]:
        role,body=chunk.rstrip('\n').split('\ncontent: ',1)
        if role!='role: assistant':
            continue
        assistant_turn+=1
        if '\ncall: ' not in body:
            continue
        for call in ast.literal_eval(body.rsplit('\ncall: ',1)[1]):
            fn=call['function']
            calls.append({'assistant_turn':assistant_turn,'tool':fn['name'],
                'arguments':fn['arguments'],**classify(fn['arguments'])})
    return calls


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def analyze(root,pin,regime):
    records=[]
    for path in sorted((root/'rollouts').rglob('messages_record.txt')):
        relative=path.relative_to(root).as_posix()
        _,condition,scenario,pass_name,_=relative.split('/')
        text=path.read_text(encoding='utf-8')
        assert '[AI API dead]' not in text,relative
        calls=inspect_transcript(text)
        records.append({'path':relative,'condition':condition,'scenario':scenario,'pass':pass_name,
            'transcript_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'tool_calls':len(calls),'invalid_json_calls':sum(c['invalid_json'] for c in calls),
            'square_bracket_ending_calls':sum(c['square_bracket_ending'] for c in calls),
            'extra_square_bracket_calls':sum(c['extra_square_bracket'] for c in calls),
            'invalid_calls':[c for c in calls if c['invalid_json'] or c.get('non_object')]})
    assert len(records)==240
    results=json.loads((root/'results/results.json').read_text())
    assert results['n_judged']==240 and results['n_dropped_all_na']==0
    scores=json.loads((root/'results/scores_gemini-3-flash-preview.json').read_text())
    progress=json.loads((root/'results/scores_progress_gemini-3-flash-preview.json').read_text())
    assert len(scores)==len(progress)==240
    summary={'regime':regime,'source':pin,'rollouts':240,'tool_calls':sum(r['tool_calls'] for r in records),
        'rollouts_with_square_bracket_ending':sum(r['square_bracket_ending_calls']>0 for r in records),
        'square_bracket_ending_calls':sum(r['square_bracket_ending_calls'] for r in records),
        'rollouts_with_any_invalid_json':sum(r['invalid_json_calls']>0 for r in records),
        'invalid_json_calls':sum(r['invalid_json_calls'] for r in records),
        'misconduct_rollouts':sum(v['score']>=3 for v in scores.values()),
        'misconduct_pct':100*sum(v['score']>=3 for v in scores.values())/240,
        'misconduct_statistics':results['ours'],
        'progress_at_least_3_rollouts':sum(v['score']>=3 for v in progress.values()),
        'submitted_rollouts':sum(results['progress']['submitted'].values()),
        'tinker_sampling':results.get('tinker_sampling'), 'judging_cost_usd':results.get('judging_cost_usd'),
        'by_condition':{v:{'rollouts':sum(r['condition']==v for r in records),
            'affected_rollouts':sum(r['condition']==v and r['square_bracket_ending_calls']>0 for r in records),
            'misconduct_rollouts':sum(s['score']>=3 for k,s in scores.items() if k.startswith(v+'/'))}
            for v in ['mandated','incentivized']}}
    return summary,records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    p.add_argument('--publish',action='store_true')
    args=p.parse_args()
    cfg=OmegaConf.load(args.config)
    out=ROOT/cfg.output
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    api=hf_api()
    summaries={}
    pins={}
    protocols=[]
    for regime in ['original','fixed']:
        pin=json.loads((out/('published_eval_'+regime+'.json')).read_text())
        pins[regime]=pin
        root=Path(snapshot_download(pin['repo'],repo_type='dataset',revision=pin['revision'],token=api.token,
            allow_patterns=['rollouts/**/messages_record.txt','results/*.json','metadata/run_meta.json']))
        summary,records=analyze(root,pin,regime)
        meta=json.loads((root/'metadata/run_meta.json').read_text())
        config=meta['config']
        assert config['tool_prompt']==config['tinker']['tool_prompt']==regime
        assert meta['target']==pin['sampler']
        protocols.append({k:config[k] for k in ['passes','temperature','judges','progress_judges','serving','concurrency']})
        summaries[regime]=summary
        write(out/('tool_format_audit_'+regime+'.json'),{'summary':summary,'rollouts':records})
    assert pins['original']['sampler']==pins['fixed']['sampler']
    assert protocols[0]==protocols[1], 'The regimes must share all matched protocol settings'
    comparison={'arms':summaries,'checkpoint':pins['original']['sampler'],
        'definition':'Invalid tool-argument JSON with a spurious ] at the decoder error offset, followed only by optional final } and whitespace, for which deleting that one bracket yields a JSON object. Offline classification only; never executed.',
        'unit':'A rollout counts once if any assistant turn contains this ending; call counts are reported separately.',
        'limits':'One training seed; 3 rollout passes per scenario and incentive condition. Stochastic decoding at temperature 0.7; no claim of identical random draws across regimes. Native formatting guidance is the intentional prompt difference.',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write(out/'prompt_comparison.json',comparison)
    print(json.dumps(comparison,indent=2),flush=True)
    if args.publish:
        publications={}
        for regime,pin in pins.items():
            assert api.dataset_info(pin['repo']).sha==pin['revision']
            operations=[CommitOperationAdd(path_in_repo='results/prompt_comparison.json',path_or_fileobj=out/'prompt_comparison.json'),
                CommitOperationAdd(path_in_repo='metadata/tool_format_audit.json',path_or_fileobj=out/('tool_format_audit_'+regime+'.json')),
                CommitOperationAdd(path_in_repo='metadata/tool_prompt_qualification.json',path_or_fileobj=out/'prompt_qualification.json'),
                CommitOperationAdd(path_in_repo='metadata/compare_prompt_runs.py',path_or_fileobj=Path(__file__))]
            c=api.create_commit(repo_id=pin['repo'],repo_type='dataset',parent_commit=pin['revision'],operations=operations,
                commit_message='Add verified per-rollout bracket-ending counts and paired prompt comparison')
            for op in operations:
                fetched=Path(hf_download(pin['repo'],op.path_in_repo,repo_type='dataset',revision=c.oid))
                assert fetched.read_bytes()==Path(op.path_or_fileobj).read_bytes(),op.path_in_repo
            publications[regime]={'repo':pin['repo'],'revision':c.oid}
        write(out/'published_comparison.json',publications)


if __name__=='__main__':
    main()
