# ABOUTME: Execute bounded, source-reviewed saved-answer revisions under the user's explicit total270 extension.
# ABOUTME: Keep old origins and budget implementation immutable; one physical attempt per candidate, no paid critic or adoption.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from filelock import FileLock
from omegaconf import OmegaConf

from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.per_row import conversation
from scratch.dataset_refresh.recover_short_draft import fields
from scratch.dataset_refresh.saved_input_pilot import parsed_fields

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / 'output/2026-09-14_dataset_refresh/budget'
CAMPAIGN = REPO / 'output/2026-09-15_nonmoral_saved_input_completion'
BASELINE_COUNT = 11609
BASELINE_SHA = '32693c4d628691e618986b9b2199388d9d17450a99feb33c680a17f134af328f'
CONSENT = 'Yes—raise the total ceiling to $270'
MODEL = 'anthropic/claude-sonnet-5'
ARM = 'nonmoral-saved-input-completion'
CLASSES = {'answer_defect_candidate_prior_evidence_not_fresh_full_read',
           'complete_final_candidate_needs_source_and_answer_review',
           'complete_draft_candidate_final_not_written',
           'earlier_sonnet_t4_candidate_pending_duplicate_and_current_rule_review'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return base.digest(Path(path).read_bytes())


def checked(path, expected):
    p = Path(path).absolute()
    if any(x.is_symlink() or x.is_junction() for x in (p, *p.parents)) or sha(p) != expected:
        raise ValueError('Changed bound input or linked path: ' + str(p))
    return p


def policy(config_path):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if (cfg.get('user_approval_quote') != CONSENT or cfg.get('shared_cap_usd') != 270
            or cfg.get('maximum_physical_calls') != 96 or cfg.get('maximum_workers') != 4
            or cfg.get('maximum_appended_instruction_bytes') != 2048
            or cfg.get('paid_critic_calls') != 0 or cfg.get('automatic_retries') != 0):
        raise ValueError('Policy differs from the explicit bounded270 continuation')
    inventory = base.read_rows(checked(cfg['inventory_path'], cfg['inventory_sha256']))
    eligible = {x['root'] + '::' + x['candidate_id']: x for x in inventory if x['classification'] in CLASSES}
    if len(eligible) != 96 or abs(sum(x['one_revision_reservation_upper_usd'] for x in eligible.values()) - 16.45) > 1e-8:
        raise ValueError('The approved96-source reservation envelope changed')
    checked(cfg['qualified_config_path'], cfg['qualified_config_sha256'])
    checked(cfg['review_contract_path'], cfg['review_contract_sha256'])
    if base.provider_price(MODEL) != {'in': 2.0, 'out': 10.0}:
        raise ValueError('Provider price changed; re-estimate before dispatch')
    return cfg, eligible


def code_hashes():
    return {str(p): sha(p) for p in (Path(__file__).resolve(), Path(base.__file__).resolve(),
        REPO/'scratch/dataset_refresh/per_row.py',
        REPO/'scratch/dataset_refresh/saved_input_pilot.py', REPO/'scratch/dataset_refresh/recover_short_draft.py',
        REPO/'src/infra/endpoints/openrouter.py', REPO/'configs/endpoints/providers.yaml')}


def initialize(config_path):
    cfg, _ = policy(config_path)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in code_hashes():
        path = Path(name)
        frozen = subprocess.check_output(['git', 'show', commit+':'+path.relative_to(REPO).as_posix()])
        if frozen.replace(b'\r\n',b'\n') != path.read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Commit the exact continuation code before initializing')
    with FileLock(str(BUDGET/'spend.lock'), timeout=1):
        if CAMPAIGN.exists():
            raise ValueError('Campaign already initialized; do not reset its attempt history')
        checked(BUDGET/'spend.json', BASELINE_SHA)
        entries = read(BUDGET/'spend.json')
        if len(entries) != BASELINE_COUNT or any(x['status'] not in {'settled', 'billing_verified_failure'} for x in entries):
            raise ValueError('Original shared budget is not the closed approved baseline')
        CAMPAIGN.mkdir(parents=True)
        shutil.copyfile(BUDGET/'spend.json', CAMPAIGN/'baseline_ledger.json')
        shutil.copyfile(cfg['review_contract_path'], CAMPAIGN/'independent_review_contract.json')
        base.save_checkpoint(CAMPAIGN/'manifest.json', {
            'config_path': str(Path(config_path).resolve()), 'config_sha256': sha(config_path),
            'source_commit': commit,
            'code_files': code_hashes(), 'baseline_ledger_sha256': BASELINE_SHA,
            'baseline_entries_digest': base.digest(entries), 'baseline_count': BASELINE_COUNT,
            'budget_root': str(BUDGET), 'shared_cap_usd': 270, 'user_approval_quote': CONSENT})
    return str(CAMPAIGN)


def verify_campaign():
    m = base.load_checkpoint(CAMPAIGN/'manifest.json')
    checked(m['config_path'], m['config_sha256'])
    for path, digest in m['code_files'].items():
        checked(path, digest)
    cfg, inventory = policy(m['config_path'])
    checked(CAMPAIGN/'baseline_ledger.json', BASELINE_SHA)
    checked(CAMPAIGN/'independent_review_contract.json', cfg['review_contract_sha256'])
    if m['shared_cap_usd'] != 270 or m['budget_root'] != str(BUDGET) or m['user_approval_quote'] != CONSENT:
        raise ValueError('Campaign budget authority changed')
    with FileLock(str(BUDGET/'spend.lock'), timeout=1):
        entries = read(BUDGET/'spend.json')
    if base.digest(entries[:BASELINE_COUNT]) != m['baseline_entries_digest']:
        raise ValueError('Original ledger prefix changed')
    tail = entries[BASELINE_COUNT:]
    tail_keys = [x.get('candidate_id') for x in tail]
    if (len(tail) > 96 or any(x.get('arm') != ARM or x.get('run_root') != str(CAMPAIGN)
                            or x.get('model') != MODEL or x.get('stage') != 'single_saved_revision'
                            or x.get('candidate_id') not in inventory for x in tail)
            or len(tail_keys) != len(set(tail_keys))
            or any(x['status'] not in {'settled', 'reserved', 'billing_verified_failure'} for x in tail)):
        raise ValueError('Unexpected, uncertain, or over-count campaign billing; inspect without retry')
    attempts_path = CAMPAIGN/'attempts.json'
    if not attempts_path.exists() and (tail or attempts_path.with_suffix('.receipt.json').exists()):
        raise ValueError('Missing attempt history or orphan receipt; never restart physical calls')
    attempts = base.load_checkpoint(attempts_path) if attempts_path.exists() else []
    if (not isinstance(attempts,list) or len(attempts)>96 or len(attempts)!=len(set(attempts))
            or not set(attempts)<=set(inventory) or not set(tail_keys)<=set(attempts)):
        raise ValueError('Attempt markers and physical ledger disagree')
    return cfg, inventory


class Authorized270Client(base.BudgetClient):
    """Reuse the frozen accounting mechanics under the separately recorded new authority."""
    def __init__(self, send=None):
        verify_campaign()
        super().__init__(BUDGET, 250, {MODEL}, send=send)
        # Only this new, manifest-bound campaign receives the user-approved extension.
        self.ceiling = 270.0


def validate_source_review(decision, ref, conv, item):
    review_path = checked(decision['source_review_path'], decision['source_review_sha256'])
    review = base.load_checkpoint(review_path)
    if (review.get('source_ref') != ref or review.get('conversation_sha256') != base.digest(conv)
            or review.get('source_eligible') is not True or review.get('root_actor') != '/root'
            or review.get('review_scope') != 'full_source_and_saved_answer'
            or review.get('decision') != 'one_revision' or not review.get('material_defect')):
        raise ValueError('Root source-review receipt does not bind this exact full-read repair')
    independent = checked(review['independent_review_path'], review['independent_review_sha256'])
    entry = read(independent)
    for key in review['independent_row_selector']:
        if type(key) not in (str,int): raise ValueError('Invalid evidence row selector')
        entry = entry[key]
    entry_sha = entry.get('result_sha256', entry.get('source_files_sha256',{}).get('result.json'))
    full_read = (entry.get('full_system_user_reasoning_final_read') is True
        or entry.get('review_scope') == 'full_conversation'
        or entry.get('scope','').startswith('Full actual source/system and complete selected saved author reasoning/final read independently.'))
    if (base.digest(entry) != review['independent_row_sha256'] or entry.get('candidate_id') != ref['candidate_id']
            or entry_sha != ref['result_sha256'] or entry.get('conversation_sha256') != base.digest(conv)
            or entry.get('selected_input_stage') != item['selected_input_stage']
            or entry.get('selected_input_sha256') != item['selected_input_sha256'] or not full_read
            or not (entry.get('source_eligible') is True or entry.get('source_usable') is True)):
        raise ValueError('Independent review does not establish exact-source full-read eligibility')
    return review_path, independent


def make_payload(cfg, item, decision):
    root = (REPO/'output'/item['root']).resolve()
    row = root/item['arm']/'records'/item['candidate_id']
    source_cfg = base.validate_arm(root, item['arm'])
    for name, expected in item['source_files_sha256'].items():
        checked(row/name, expected)
        base.load_checkpoint(row/name)
    if not base.acceptance(base.load_checkpoint(row/'preflight.json'), source_cfg['preflight']):
        raise ValueError('Source is ineligible under its original preflight')
    saved_path = checked(row/(item['selected_input_stage']+'.json'), item['selected_input_sha256'])
    saved = base.load_checkpoint(saved_path)
    original = base.load_checkpoint(row/'result.json')
    record = dict(original['record'])
    scenario = base.load_checkpoint(row/'scenario.json')
    if any(record.get(k) != scenario.get(k) for k in ('system','user')):
        raise ValueError('Original source differs from saved scenario')
    pair = (('draft_reasoning','draft_response') if item['selected_input_stage']=='draft_responses' else ('reasoning','response'))
    record.update(reasoning=saved[pair[0]], response=saved[pair[1]],
                  draft_reasoning=saved[pair[0]], draft_response=saved[pair[1]])
    frozen = read(cfg['qualified_config_path'])
    record['trait_text'] = frozen['operational_traits'][item['trait_id']]
    stage = frozen['response_stages'][-1]
    messages = [{'role':role, 'content':base.render(stage['prompts'][role],fields(record,frozen))}
                for role in ('system','user')]
    initial_bytes = len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))
    if initial_bytes != item['base_rendered_request_message_bytes']:
        raise ValueError('Saved request differs from the approved cost envelope')
    ref={'root':str(root),'arm':item['arm'],'candidate_id':item['candidate_id'],
         'result_sha256':item['source_files_sha256']['result.json']}
    review_path, independent_path = validate_source_review(decision,ref,conversation(record),item)
    if (decision.get('source_eligible') is not True or decision.get('decision') != 'one_revision'
            or decision.get('source_conversation_sha256') != base.digest(conversation(record))
            or not isinstance(decision.get('instruction'),str) or len(decision['instruction'].strip())<20):
        raise ValueError('Explicit source-bound root repair decision required')
    messages[-1]['content'] += '\n\nSource-bound correction for this saved answer:\n'+decision['instruction']
    size = len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))
    if size-initial_bytes > 2048:
        raise ValueError('Correction text exceeds the approved2048-byte envelope')
    request = {'model':MODEL,'temperature':0.7,'max_tokens':12288,'messages':messages}
    reserve = (1.25*(size+2048)*2+12288*10)/1e6
    if reserve > item['one_revision_reservation_upper_usd']+1e-10:
        raise ValueError('Request reservation exceeds inventory envelope')
    bound = {str(row/name):digest for name,digest in item['source_files_sha256'].items()}
    bound[str(saved_path)] = sha(saved_path)
    bound[str(review_path)] = sha(review_path)
    bound[str(review_path.with_suffix('.receipt.json'))] = sha(review_path.with_suffix('.receipt.json'))
    bound[str(independent_path)] = sha(independent_path)
    for path in [root/item['arm']/'config.json', root/item['arm']/'candidates.jsonl',
                 *(row/(name.removesuffix('.json')+'.receipt.json') for name in item['source_files_sha256']),
                 saved_path.with_suffix('.receipt.json'), row/'independent_exclusion.json',
                 row/'independent_exclusion.receipt.json']:
        if path.exists(): bound[str(path)] = sha(path)
    return {'candidate_id':item['candidate_id'],'case_key':item['root']+'::'+item['candidate_id'],
        'trait_id':item['trait_id'], 'source_ref':ref,
        'actual_conversation':conversation(record),'full_working_preference':record['trait_text'],
        'request':request,'request_sha256':base.digest(request),'reservation_usd':reserve,
        'bound_files':bound,'root_source_decision':decision,'original_record_metadata':original['record']}


def prepare_batch(decisions_path, name):
    cfg, inventory = verify_campaign()
    decisions = read(decisions_path)
    if not isinstance(decisions,list) or not 1 <= len(decisions) <= 4 or len({d['case_key'] for d in decisions}) != len(decisions):
        raise ValueError('One to four distinct explicitly reviewed sources per batch')
    if not name.replace('_','').isalnum():
        raise ValueError('Simple batch name required')
    target = CAMPAIGN/'batches'/name
    if target.exists():
        raise ValueError('Batch already exists; never overwrite dispatch history')
    payloads = [make_payload(cfg, inventory[d['case_key']], d) for d in decisions]
    target.mkdir(parents=True)
    for i,payload in enumerate(payloads):
        base.save_checkpoint(target/f'{i:02d}.input.json',payload)
    base.save_checkpoint(target/'batch.json',{'campaign_manifest_sha256':sha(CAMPAIGN/'manifest.json'),
        'decisions_path':str(Path(decisions_path).resolve()),'decisions_sha256':sha(decisions_path),
        'inputs':[{'name':f'{i:02d}.input.json','sha256':sha(target/f'{i:02d}.input.json')} for i in range(len(payloads))],
        'maximum_calls':len(payloads),'reservation_sum_usd':sum(p['reservation_usd'] for p in payloads)})
    base.write_json(target/'dispatch.json',{'enabled':False,'batch_sha256':sha(target/'batch.json')})
    return str(target)


def completed_summary(folder, batch):
    """A cached completion is returned only while its source and physical evidence still agree."""
    summary = base.load_checkpoint(folder/'summary.json')
    with FileLock(str(BUDGET/'spend.lock'), timeout=1):
        ledger = read(BUDGET/'spend.json')
    statuses=[]
    for entry in batch['inputs']:
        source=checked(folder/entry['name'],entry['sha256']); payload=base.load_checkpoint(source)
        for path,digest in payload['bound_files'].items(): checked(path,digest)
        prefix=source.name.removesuffix('.input.json')
        result=base.load_checkpoint(folder/(prefix+'.result.json'))
        if result['candidate_id']!=payload['candidate_id'] or result['case_key']!=payload['case_key']:
            raise ValueError('Cached result identity changed')
        receipt=result.get('physical_receipt')
        if receipt:
            raw=read(checked(folder/(prefix+'.raw.json'),receipt['raw_sha256']))
            physical=ledger[receipt['call_id']]
            if (base.digest(physical)!=receipt['ledger_entry_sha256'] or raw['accounting']!=physical
                    or raw['request']!=payload['request'] or receipt['request_sha256']!=payload['request_sha256']):
                raise ValueError('Cached physical author evidence changed')
            if result['status']=='awaiting_independent_full_review':
                parsed,_=parsed_fields(raw['response']['content'])
                if result['conversation']!={**payload['actual_conversation'],**parsed}:
                    raise ValueError('Cached answer differs from physical author')
        elif result['status']=='awaiting_independent_full_review':
            raise ValueError('Cached successful answer lacks physical evidence')
        statuses.append({k:result[k] for k in ('candidate_id','case_key','status')})
    if statuses!=summary['results'] or summary['automatic_accepted_rows']!=0:
        raise ValueError('Cached summary differs from its per-case evidence')
    return summary


def execute_batch(name, send=None):
    verify_campaign()
    folder = CAMPAIGN/'batches'/name
    if folder.resolve().parent != (CAMPAIGN/'batches').resolve():
        raise ValueError('Batch path escaped campaign')
    with FileLock(str(CAMPAIGN/'execution.lock'), timeout=1):
        batch = base.load_checkpoint(folder/'batch.json')
        if (read(folder/'dispatch.json') != {'enabled':True,'batch_sha256':sha(folder/'batch.json')}
                or batch['campaign_manifest_sha256'] != sha(CAMPAIGN/'manifest.json')):
            raise ValueError('Exact batch dispatch is not enabled')
        checked(batch['decisions_path'],batch['decisions_sha256'])
        if (folder/'summary.json').exists():
            return completed_summary(folder,batch)
        if (folder/'started.json').exists():
            raise ValueError('Interrupted batch: inspect all attempts; never retry automatically')
        payloads = [(Path(x['name']).stem.replace('.input',''),base.load_checkpoint(checked(folder/x['name'],x['sha256']))) for x in batch['inputs']]
        if (not 1<=len(payloads)<=4 or batch['maximum_calls']!=len(payloads)
                or len({p['case_key'] for _,p in payloads})!=len(payloads)
                or abs(batch['reservation_sum_usd']-sum(p['reservation_usd'] for _,p in payloads))>1e-10):
            raise ValueError('Runtime batch must contain one to four distinct bounded requests')
        attempts_path=CAMPAIGN/'attempts.json'
        attempts=base.load_checkpoint(attempts_path) if attempts_path.exists() else []
        keys=[p['case_key'] for _,p in payloads]
        if len(attempts)+len(keys)>96 or set(keys)&set(attempts):
            raise ValueError('Candidate already attempted or global96-attempt limit reached')
        # Revalidate every source before registering any irreversible one-attempt marker.
        cfg, inventory=verify_campaign()
        for _,p in payloads:
            for path,digest in p['bound_files'].items(): checked(path,digest)
            if make_payload(cfg,inventory[p['case_key']],p['root_source_decision'])!=p:
                raise ValueError('Prepared source or request changed')
        client=Authorized270Client(send=send)
        with client.lock:
            physical_keys={e.get('candidate_id') for e in client.entries()[BASELINE_COUNT:]}
            if set(keys)&physical_keys:
                raise ValueError('A requested candidate already has a physical call; no retry')
            if sum(x['charged_or_reserved_usd'] for x in client.entries())+batch['reservation_sum_usd']>270:
                raise ValueError('Whole batch does not fit the shared cap')
        base.save_checkpoint(attempts_path,attempts+keys)
        base.save_checkpoint(folder/'started.json',{'case_keys':keys,'dispatch_sha256':sha(folder/'dispatch.json')})

        def run_one(index,payload):
            client.local.arm=ARM; client.local.stage='single_saved_revision'
            client.local.run_root=str(CAMPAIGN); client.local.candidate_id=payload['case_key']
            item={'candidate_id':payload['candidate_id'],'case_key':payload['case_key'],
                  'status':'failed','automatic_acceptance':False}
            try:
                response=client.chat(**payload['request'])
                if response.response_model!=MODEL or response.finish_reason!='stop':
                    raise ValueError('Unexpected author model or incomplete output')
                parsed,audit=parsed_fields(response.content)
                revised={**payload['actual_conversation'],**parsed}
                if base.simple_checks(revised):
                    raise ValueError('New answer fails frozen local checks')
                item.update(status='awaiting_independent_full_review',conversation=revised,audit=audit)
            except BaseException as exc:
                item.update(error_type=type(exc).__name__,error=str(exc)[:2000])
            with client.lock:
                matches=[e for e in client.entries() if e.get('arm')==ARM and e.get('candidate_id')==payload['case_key']
                         and e.get('run_root')==str(CAMPAIGN) and e.get('stage')=='single_saved_revision']
            if len(matches)==1:
                e=matches[0]; raw=BUDGET/'raw_calls'/f"{e['call_id']:06d}.json"
                raw_sha=sha(raw); copied=folder/(index+'.raw.json'); shutil.copyfile(raw,copied)
                item['physical_receipt']={'call_id':e['call_id'],'request_sha256':e['request_sha256'],
                    'raw_sha256':raw_sha,'ledger_entry_sha256':base.digest(e)}
                evidence=read(copied)
                if (sha(copied)!=raw_sha or sha(raw)!=raw_sha or e['request_sha256']!=payload['request_sha256']
                        or evidence.get('request')!=payload['request'] or evidence.get('accounting')!=e):
                    item.update(status='failed',error_type='ValueError',error='Physical request/raw/accounting lineage mismatch')
            elif matches:
                raise ValueError('Multiple physical calls for one candidate')
            elif item['status']=='awaiting_independent_full_review':
                item.update(status='failed',error_type='ValueError',error='Missing physical author ledger entry')
            base.save_checkpoint(folder/(index+'.result.json'),item)
            return {'candidate_id':item['candidate_id'],'case_key':item['case_key'],'status':item['status']}

        with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            futures=[pool.submit(run_one,i,p) for i,p in payloads]
            results=[f.result() for f in futures]
        with client.lock:
            exposure=sum(e['charged_or_reserved_usd'] for e in client.entries())
        summary={'results':results,'shared_exposure_usd':exposure,'automatic_accepted_rows':0,
                 'review_route':'Fresh independent full-conversation review and separate root adoption required.'}
        base.save_checkpoint(folder/'summary.json',summary)
        return summary


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('initialize'); a.add_argument('--config',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--decisions',required=True); a.add_argument('--name',required=True)
    a=sub.add_parser('execute'); a.add_argument('--name',required=True)
    args=p.parse_args()
    if args.command=='initialize': result=initialize(args.config)
    elif args.command=='prepare': result=prepare_batch(args.decisions,args.name)
    else: result=execute_batch(args.name)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
