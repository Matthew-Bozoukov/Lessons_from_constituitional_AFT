# ABOUTME: One explicit Sonnet correction of an independently excluded answer, preserving the human request.
# ABOUTME: Fresh blinded reviews and separate evidence-bound adoption retain every rejected artifact.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from filelock import FileLock
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.per_row import assert_models, conversation, grounded, verify_accepted
from scratch.dataset_refresh.reviewer_probe import validate_verdict

ATTEMPT = 100


def inputs(root, arm, cid):
    root = Path(root).resolve()
    cfg = base.validate_arm(root, arm)
    assert_models(cfg)
    row = root / arm / 'records' / cid
    if not row.resolve().is_relative_to(root / arm / 'records'):
        raise ValueError('Candidate path outside arm')
    original = base.load_checkpoint(row / 'result.json')
    exclusion = base.load_checkpoint(row / 'independent_exclusion.json')
    if original['status'] != 'accepted' or exclusion['result_sha256'] != base.digest((row / 'result.json').read_bytes()):
        raise ValueError('Requires a bound independent exclusion of a completed answer')
    if not base.acceptance(base.load_checkpoint(row / 'preflight.json'), cfg['preflight']):
        raise ValueError('Ineligible scenarios cannot be repaired by changing answers')
    return root, cfg, row, original, exclusion


def propose(root, arm, cid, ceiling, send=None):
    root, cfg, row, original, exclusion = inputs(root, arm, cid)
    with FileLock(str(row / 'independent_repair.lock'), timeout=1):
        dest = row / f'independent_candidate_{ATTEMPT}.json'
        if dest.exists():
            return base.load_checkpoint(dest)
        source_sha = base.digest((row / 'result.json').read_bytes())
        manifest_path = row / f'independent_repair_{ATTEMPT}.json'
        if manifest_path.exists():
            manifest = base.load_checkpoint(manifest_path)
            if manifest['source_result_sha256'] != source_sha or manifest['script_sha256'] != base.digest(Path(__file__).read_bytes()):
                raise ValueError('Repair source or implementation changed')
        else:
            manifest = {'source_result_sha256': source_sha, 'exclusion': exclusion,
                        'script_sha256': base.digest(Path(__file__).read_bytes()),
                        'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                        'authorization': 'User authorized iterative repairs; one external Sonnet correction after independent audit.',
                        'policy': 'System/user fixed. Original stays excluded until separately audited adoption.',
                        'attempt': ATTEMPT}
            base.save_checkpoint(manifest_path, manifest)
        meta = base.load_checkpoint(root / 'run_meta.json')
        client = base.BudgetClient(meta['budget_root'], ceiling, {m['model'] for m in cfg['models'].values()}, send=send)
        client.local.arm, client.local.candidate_id, client.local.run_root = arm, cid, str(root)
        record = dict(original['record'])

        def call(name, messages, model, tagged=None):
            path = row / (name + '.json')
            if path.exists():
                return base.load_checkpoint(path)
            # Started markers prevent duplicate physical calls after unknown outcomes.
            marker = row / (name + '.started.json')
            if marker.exists():
                raise base.BudgetStop('Repair stage already started without settled checkpoint: ' + name)
            base.save_checkpoint(marker, {'messages_sha256': base.digest(messages), 'model': model})
            client.local.stage = name
            raw = client.chat(messages=messages, **base.request_options(cfg['models'][model]))
            if tagged:
                parsed = base._parse_tagged(raw.content, tuple(tagged['tags']))
                value = {dest: parsed[tag] for dest, tag in tagged['save'].items()}
            else:
                value = base._parse_json(raw.content)
            base.save_checkpoint(path, value)
            return value

        def fields():
            return {**record, 'style_guidance': cfg.get('style_guidance',''),
                    'constitution': cfg['review_constitution_text'], 'metadata_json':'{}', 'eligibility_json':'{}',
                    'conversation_json':json.dumps(conversation(record),ensure_ascii=False),
                    'record_json':json.dumps(conversation(record),ensure_ascii=False)}

        try:
            stage = cfg['response_stages'][-1]
            f = fields()
            f.update(draft_reasoning=record['reasoning'],draft_response=record['response'])
            messages = [{'role':role,'content':base.render(stage['prompts'][role],f)} for role in ('system','user')]
            messages[-1]['content'] += ('\n\nIndependent audit of the actual answer:\n' + json.dumps(exclusion,ensure_ascii=False)
                + '\nCorrect substantive defects in BOTH rationale and final answer. Keep the system and user exactly fixed. '
                'Verify the critique against the actual request; do not invent permissions, motives, facts or guarantees. '
                'Do not mention this audit, earlier drafts, reviewers, rewriting, or training in either output. '
                'Reason directly about the human decision, keeping genuine alternatives and their costs.')
            saved = call(f'independent_rewrite_{ATTEMPT}', messages, stage['model'], stage)
            record.update(saved)
            lint = base.lint_problems({tag:saved[dest] for dest,tag in stage['save'].items()},stage.get('lint') or {},record)
            if lint or base.simple_checks(record):
                raise ValueError('Repaired answer failed local checks: '+str(lint))
            narrow = conversation(record); narrow['final'] = narrow.pop('response')
            spec = cfg['grounding_review']
            messages = [{'role':role,'content':base.render(spec['prompts'][role],{'conversation_json':json.dumps(narrow,ensure_ascii=False)})} for role in ('system','user')]
            g = call(f'grounding_{ATTEMPT}', messages, spec['model'])
            validate_verdict(g,narrow)
            messages = [{'role':role,'content':base.render(cfg['prompts']['review_'+role],fields())} for role in ('system','user')]
            r = call(f'review_{ATTEMPT}',messages,'review')
            base.save_checkpoint(row/f'answer_{ATTEMPT}.json',conversation(record))
            record['response_repair_count'] = original['record'].get('response_repair_count',0)+1
            candidate = {'candidate_id':cid,'trait_id':original['trait_id'],
                         'status':'accepted' if grounded(g) and base.acceptance(r,cfg) else 'rejected',
                         'record':record,'review':r,'grounding_review':g,'accepted_attempt':ATTEMPT,
                         'independent_repair_manifest_sha256':base.digest(manifest_path.read_bytes()),
                         'source_result_sha256':source_sha}
        except base.BudgetStop:
            raise
        except Exception as exc:
            candidate = {'candidate_id':cid,'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                         'source_result_sha256':source_sha,'record':record}
        base.save_checkpoint(dest,candidate)
        return candidate


def adopt(root, arm, cid, approved_sha, reason):
    root, cfg, row, original, exclusion = inputs(root,arm,cid)
    if not reason.strip():
        raise ValueError('Explicit independent post-repair audit reason required')
    with FileLock(str(root/'execution.lock'),timeout=1):
        path = row/f'independent_candidate_{ATTEMPT}.json'
        candidate = base.load_checkpoint(path)
        if base.digest(path.read_bytes()) != approved_sha or candidate['status'] != 'accepted':
            raise ValueError('Approval must bind this reviewed successful repair')
        if candidate['source_result_sha256'] != base.digest((row/'result.json').read_bytes()):
            raise ValueError('Source result changed')
        if conversation(candidate['record'],False) != conversation(original['record'],False):
            raise ValueError('Repair altered the human request')
        verify_accepted(path,candidate,cfg)
        archive = row/'recovered_failures'/f'independent_repair_{ATTEMPT}'
        if archive.exists():
            raise ValueError('Already adopted or partially adopted; inspect preserved evidence')
        base.save_checkpoint(row/f'independent_adoption_{ATTEMPT}.json',
            {'candidate_sha256':approved_sha,'source_result_sha256':candidate['source_result_sha256'],
             'independent_audit_reason':reason,'source_exclusion':exclusion})
        archive.mkdir(parents=True)
        for name in ('result.json','result.receipt.json','independent_exclusion.json','independent_exclusion.receipt.json'):
            shutil.copy2(row/name,archive/name)
        base.save_checkpoint(row/'result.json',candidate)
        # Until the old exclusion is removed, a crash produces a hash mismatch and
        # fails closed; it never exposes a missing terminal for blind regeneration.
        (row/'independent_exclusion.receipt.json').unlink()
        (row/'independent_exclusion.json').unlink()
        return {'candidate_id':cid,'status':'adopted','result_sha256':base.digest((row/'result.json').read_bytes())}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['propose','adopt'])
    p.add_argument('--root',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--candidates',nargs='+',required=True)
    p.add_argument('--ceiling',type=float,default=90);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--approved-sha');p.add_argument('--reason')
    a=p.parse_args()
    if a.command=='propose':
        def task(cid):
            r=propose(a.root,a.arm,cid,a.ceiling)
            print(json.dumps({k:r[k] for k in ('candidate_id','status','error') if k in r}),flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(task,a.candidates))
    else:
        if len(a.candidates)!=1 or not a.approved_sha or not a.reason:
            p.error('Adoption requires one candidate, approved SHA and independent audit reason')
        print(json.dumps(adopt(a.root,a.arm,a.candidates[0],a.approved_sha,a.reason)))
