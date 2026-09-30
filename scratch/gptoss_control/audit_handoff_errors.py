# ABOUTME: Reparse preserved rejected Tinker completions without new inference and disclose handoff failures.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/audit_handoff_errors.py --publish
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.infra.endpoints.harmony import make_renderer
from src.infra.huggingface import hf_api, hf_download


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p,value):
    p.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    out=ROOT/'output/gptoss_control/control_refresh_2026-09-30'
    assert not (out/'published_handoff_audit.json').exists(), 'Already published; inspect receipt'
    renderer=make_renderer(local_files_only=True)
    audit={'method':'Offline replay of every sampling-ledger error with its saved output tokens; no inference calls.',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arms':{}}
    for regime in ['original','fixed']:
        root,=(out/('odcv_'+regime)).iterdir()
        ledger=root/'metadata/tinker_shim/sampling.jsonl'
        events=[json.loads(x) for x in ledger.read_text().splitlines()]
        completed={e['id']:e for e in events if e['event']=='completed'}
        rejected=[]
        for error in [e for e in events if e['event']=='error']:
            event=completed[error['id']]
            tokens=event['raw_tokens']
            parsed,term=renderer.parse_response(tokens)
            assert term.is_stop_sequence and not parsed.get('unparsed_tool_calls')
            message=renderer.to_openai_message(parsed)
            calls=message.get('tool_calls') or []
            assert calls and tokens[-1]==200002 and tokens[-1]!=200012
            assert all(c['function']['name'] in {'bash','task_complete'} for c in calls)
            assert all(isinstance(json.loads(c['function']['arguments']),dict) for c in calls)
            rejected.append({'request_id':error['id'],'recorded_error_type':error['type'],
                'cause':'tool_calls_ended_with_return_instead_of_call',
                'ending_token':tokens[-1],'ending_text':renderer.tokenizer.decode(tokens[-1:]),
                'tool_names':[c['function']['name'] for c in calls],
                'all_arguments_valid_json_objects':True,
                'output_tokens_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
        audit['arms'][regime]={'rejected_completions':len(rejected),'ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),
            'cause_counts':dict(Counter(e['cause'] for e in rejected)),'completions':rejected}
    write(out/'bridge_error_audit.json',audit)
    note=('Separate from malformed JSON: the bridge rejected 1 original-prompt completion and 13 fixed-prompt '
        'completions because parsed tool calls ended with Harmony `<|return|>` (200002), instead of '
        '`<|call|>` (200012). All their arguments parse as JSON objects and their tool names are declared. '
        'The existing HTTP retry path resampled these individual completions; no whole rollout was rerun. '
        'These rejected attempts are absent from visible tool-call transcripts but preserved in the sampling '
        'ledger and `metadata/bridge_error_audit.json`. Thus the square-bracket reduction does not mean '
        'all native-format failures improved. The two scored runs use the same bridge/retry policy.')
    comparison=read(out/'prompt_comparison.json')
    comparison['bridge_error_audit']=audit
    comparison['limits']+=' '+note
    write(out/'prompt_comparison.json',comparison)
    closeout=out/'closeout.md'
    closeout.write_text(closeout.read_text(encoding='utf-8')+'\n## Additional Harmony-ending failures\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps({k:v['rejected_completions'] for k,v in audit['arms'].items()}),flush=True)
    if args.publish:
        api=hf_api()
        pins=read(out/'published_comparison.json')
        write(out/'published_comparison_before_handoff_audit.json',pins)
        published={}
        for regime,pin in pins.items():
            assert api.dataset_info(pin['repo']).sha==pin['revision']
            operations=[CommitOperationAdd(path_in_repo=dest,path_or_fileobj=src) for dest,src in [
                ('metadata/bridge_error_audit.json',out/'bridge_error_audit.json'),
                ('metadata/audit_handoff_errors.py',Path(__file__)),
                ('metadata/closeout.md',out/'closeout.md'),
                ('results/prompt_comparison.json',out/'prompt_comparison.json')]]
            commit=api.create_commit(repo_id=pin['repo'],repo_type='dataset',parent_commit=pin['revision'],
                operations=operations,commit_message='Disclose and verify separately retried Harmony tool-ending failures')
            for op in operations:
                remote=Path(hf_download(pin['repo'],op.path_in_repo,repo_type='dataset',revision=commit.oid))
                assert remote.read_bytes()==Path(op.path_or_fileobj).read_bytes()
            published[regime]={'repo':pin['repo'],'revision':commit.oid}
        write(out/'published_handoff_audit.json',published)
        write(out/'published_comparison.json',published)


if __name__=='__main__':
    main()
