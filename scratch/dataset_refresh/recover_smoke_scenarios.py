# ABOUTME: Explicit offline structural recovery of saved scenario JSON, without changing strings or retrying calls.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.recover_smoke_scenarios --root <run-dir> [--continue]
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from src.data.synth.ours import pipeline
from src.utils import git_sha
from scratch.dataset_refresh.constitution_smoke import SingleAttemptClient, validate_review
from scratch.dataset_refresh.run import write_json


def recover(content):
    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
    if blocks:
        assert not re.sub(r"```(?:json)?\s*[\s\S]*?```", "", content).strip(), 'Unexpected text outside JSON'
    else:
        blocks = [content]
    decoder = json.JSONDecoder(strict=False)
    merged = {}
    operations = []
    for block in blocks:
        block = block.strip()
        # One observed premature outer close between user string and applicability field.
        pattern = r'(")\s*},\s*("applicability"\s*:)'
        fixed, count = re.subn(pattern, r'\1,\n  \2', block)
        assert count <= 1
        if count:
            operations.append('remove premature outer brace before applicability')
        while fixed.strip():
            fixed = fixed.lstrip()
            obj, end = decoder.raw_decode(fixed)
            assert isinstance(obj, dict) and not (merged.keys() & obj.keys()), 'Ambiguous duplicate fields'
            merged.update(obj)
            fixed = fixed[end:]
    assert set(merged) == {'system', 'user', 'applicability'}
    assert all(isinstance(merged[k], str) and merged[k] for k in ['system','user'])
    assert isinstance(merged['applicability'], dict)
    return merged, operations


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--continue',dest='proceed',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    cfg=json.loads((root/'run_meta.json').read_text(encoding='utf-8'))['config']
    assert all(x['correct'] for x in json.loads((root/'calibration_results.json').read_text(encoding='utf-8')))
    generation=root/'generation'
    target=generation/'stage_3_write_scenarios.jsonl'
    assert not target.exists(), 'Recovery already applied'
    sources={}
    notes=[]
    for file in sorted((root/'budget/raw_calls').glob('*.json')):
        call=json.loads(file.read_text(encoding='utf-8'))
        if call['accounting']['stage'] != 'scenario': continue
        rid=call['accounting']['candidate_id']
        assert rid not in sources and call['response']['finish_reason']=='stop'
        sources[rid], ops=recover(call['response']['content'])
        notes.append(dict(id=rid,raw_file=str(file.relative_to(root)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),operations=ops))
    planned=[json.loads(s) for s in (generation/'stage_2_deal_checklists.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(sources)==len(planned)==18 and set(sources)=={r['scenario_id'] for r in planned}
    rows=[{**r,**sources[r['scenario_id']]} for r in planned]
    write_json(root/'structural_recovery.json',dict(git_sha=git_sha(),no_string_edits=True,model_calls=0,notes=notes,method='Decode JSON strings allowing literal control characters; merge disjoint objects; remove one premature outer brace. No changes inside string values.'))
    shutil.copy2(__file__,root/Path(__file__).name)
    shutil.copy2(generation/'manifest.json',root/'pre_recovery_manifest.json')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print('Recovered all 18 original scenario texts with zero model calls.',flush=True)
    if not args.proceed: return
    client=SingleAttemptClient(root,cfg)
    try:
        manifest=pipeline.run(cfg,smoke=True,resume=str(generation),client=client)
        exported=[json.loads(s) for s in (generation/'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in exported: validate_review(row['metadata']['review'],cfg['smoke_contract'].get('review_fields',[]))
        write_json(root/'automatic_summary.json',dict(completed=len(exported),model_pass=sum(r['metadata']['review']['verdict']=='pass' for r in exported),independent_review='pending',manifest=manifest))
    except BaseException as exc:
        write_json(root/'continuation_stopped.json',dict(exception=type(exc).__name__,message=str(exc)))
        raise
    finally:
        entries=client.budget.entries()
        write_json(root/'cost_summary.json',dict(physical_calls=len(entries),charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries),statuses={s:sum(e['status']==s for e in entries) for s in {e['status'] for e in entries}}))


if __name__=='__main__': main()
