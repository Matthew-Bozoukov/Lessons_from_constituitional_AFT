# ABOUTME: Run at most one additional focused Sonnet correction after the first saved-input campaign is closed.
# ABOUTME: Preserve all first outcomes and shared270 accounting; disabled dispatch, no critics, retries or adoption.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from filelock import FileLock
from omegaconf import OmegaConf

from scratch.dataset_refresh import run as base, saved_input_completion as first, saved_input_pilot as pilot
from scratch.dataset_refresh import offline_acceptance as offline
from scratch.dataset_refresh.recover_short_draft import fields
from scratch.dataset_refresh.billing_evidence import validate_billing_call

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'output/2026-09-15_nonmoral_saved_input_second_pass'
BUDGET = first.BUDGET
PILOT_ROOT = REPO/'output/2026-09-15_dataset_refresh_saved_input_pilot'
ARM = 'nonmoral-saved-input-second-pass'
CONSENT = 'Yes—allow one additional focused revision within $270'
MODEL = pilot.MODEL
MINIMAL = ('Make only the focused factual cleanup or deletion requested below, consistently in both reasoning and final. '
           'Preserve useful grounded reasoning and the existing recommendation unless this correction requires changing it. '
           'Do not add new explanatory premises, invented observations, mechanisms, audience facts or guarantees. '
           'Write standalone advice grounded only in the actual system/user. Neither trained block may discuss an earlier '
           'draft, previous answer, revision process, reviewer, correction instruction or this audit. '
           'The <changes> block, if present, is separate untrained audit text.\nFocused issue:\n')


def policy(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (cfg.get('user_approval_quote') != CONSENT or cfg.get('shared_cap_usd') != 270 or
            cfg.get('maximum_batch_size') != 4 or cfg.get('maximum_workers') != 1 or
            cfg.get('maximum_extra_calls_per_source') != 1 or cfg.get('automatic_retries') != 0 or
            cfg.get('paid_critic_calls') != 0 or cfg.get('execution_enabled') is not False or
            cfg.get('maximum_appended_instruction_bytes') != 2048 or cfg.get('allowed_max_tokens') != [8192, 12288]):
        raise ValueError('Second-pass policy differs from explicit bounded authority')
    return cfg


def code_hashes():
    return {**first.code_hashes(), str(Path(__file__).resolve()): offline.sha(__file__),
            str(Path(offline.__file__).resolve()): offline.sha(offline.__file__),
            str(Path(validate_billing_call.__code__.co_filename).resolve()): offline.sha(validate_billing_call.__code__.co_filename)}


def first_batches():
    hashes = {}
    for path in sorted((first.CAMPAIGN/'batches').rglob('*')):
        if path.is_file():
            hashes[path.relative_to(first.CAMPAIGN).as_posix()] = offline.sha(offline.regular(path))
    return hashes


def initialize(config_path, closure_path):
    """Offline only; root must first freeze a closure receipt for the original campaign."""
    policy(config_path)
    closure = base.load_checkpoint(closure_path)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for path in code_hashes():
        frozen = subprocess.check_output(['git', 'show', commit+':'+Path(path).relative_to(REPO).as_posix()])
        if frozen.replace(b'\r\n', b'\n') != Path(path).read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Commit exact second-pass and inherited code before initialization')
    with FileLock(str(first.CAMPAIGN/'execution.lock'), timeout=1):
        first.verify_campaign()
        for folder in (first.CAMPAIGN/'batches').iterdir():
            if folder.is_dir():
                if not (folder/'summary.json').exists():
                    raise ValueError('Every prepared first campaign batch must be completed before closure')
                first.completed_summary(folder, base.load_checkpoint(folder/'batch.json'))
        closing_batches = first_batches()
        with FileLock(str(BUDGET/'spend.lock'), timeout=1):
            ledger = first.read(BUDGET/'spend.json')
            if (closure.get('root_actor') != '/root' or closure.get('first_campaign_closed') is not True or
                    closure.get('first_campaign_manifest_sha256') != offline.sha(first.CAMPAIGN/'manifest.json') or
                    closure.get('shared_ledger_sha256') != offline.sha(BUDGET/'spend.json') or
                    closure.get('shared_ledger_count') != len(ledger) or not closure.get('reason')):
                raise ValueError('Explicit exact first-campaign closure receipt required')
            if any(e['status'] not in ('settled', 'billing_verified_failure') for e in ledger):
                raise ValueError('All first physical calls must be settled before second-pass initialization')
            attempts = base.load_checkpoint(first.CAMPAIGN/'attempts.json')
            physical = [e['candidate_id'] for e in ledger[first.BASELINE_COUNT:]]
            if set(physical) != set(attempts) or len(physical) != len(attempts):
                raise ValueError('First attempted sources and physical outcomes disagree')
            if ROOT.exists():
                raise ValueError('Second-pass campaign already exists; never reset its prefix or attempt ledger')
            ROOT.mkdir(parents=True)
            shutil.copyfile(BUDGET/'spend.json', ROOT/'baseline_ledger.json')
            shutil.copyfile(first.CAMPAIGN/'independent_review_contract.json', ROOT/'independent_review_contract.json')
            shutil.copyfile(closure_path, ROOT/'first_campaign_closure.json')
            base.save_checkpoint(ROOT/'manifest.json', {'config_path': str(Path(config_path).resolve()),
                'config_sha256': offline.sha(config_path), 'source_commit': commit, 'code_files': code_hashes(),
                'user_approval_quote': CONSENT, 'shared_cap_usd': 270, 'budget_root': str(BUDGET),
                'baseline_count': len(ledger), 'baseline_ledger_sha256': offline.sha(BUDGET/'spend.json'),
                'baseline_entries_digest': base.digest(ledger),
                'first_campaign_manifest_sha256': offline.sha(first.CAMPAIGN/'manifest.json'),
                'first_attempts_sha256': offline.sha(first.CAMPAIGN/'attempts.json'),
                'first_batches_sha256': closing_batches,
                'closure_path': str(Path(closure_path).resolve()), 'closure_sha256': offline.sha(closure_path),
                'review_contract_sha256': offline.sha(ROOT/'independent_review_contract.json')})
    return str(ROOT)


def verify():
    m = base.load_checkpoint(ROOT/'manifest.json')
    policy(offline.bound(m['config_path'], m['config_sha256']))
    if m['user_approval_quote'] != CONSENT or m['shared_cap_usd'] != 270 or m['budget_root'] != str(BUDGET):
        raise ValueError('Second-pass authority or budget root differs')
    for path, digest in m['code_files'].items(): offline.bound(path, digest)
    offline.bound(ROOT/'baseline_ledger.json', m['baseline_ledger_sha256'])
    offline.bound(ROOT/'first_campaign_closure.json', m['closure_sha256'])
    offline.bound(m['closure_path'], m['closure_sha256'])
    offline.bound(first.CAMPAIGN/'manifest.json', m['first_campaign_manifest_sha256'])
    offline.bound(first.CAMPAIGN/'attempts.json', m['first_attempts_sha256'])
    if first_batches() != m['first_batches_sha256']:
        raise ValueError('First campaign batches changed or another first batch appeared after closure')
    offline.bound(ROOT/'independent_review_contract.json', m['review_contract_sha256'])
    ledger = first.read(BUDGET/'spend.json')
    if len(ledger) < m['baseline_count'] or base.digest(ledger[:m['baseline_count']]) != m['baseline_entries_digest']:
        raise ValueError('Closed first-campaign ledger prefix changed')
    tail = ledger[m['baseline_count']:]
    keys = [e['candidate_id'] for e in tail]
    if (len(keys) != len(set(keys)) or any(e.get('run_root') != str(ROOT) or e.get('arm') != ARM or
            e.get('stage') != 'single_saved_revision' or e.get('model') != MODEL or
            e.get('status') not in ('settled', 'reserved', 'uncertain_failure', 'billing_verified_failure') for e in tail)):
        raise ValueError('Unexpected second-pass physical tail or duplicate additional call')
    path = ROOT/'attempts.json'
    attempts = base.load_checkpoint(path) if path.exists() else []
    if (not path.exists() and (tail or path.with_suffix('.receipt.json').exists()) or
            not isinstance(attempts, list) or len(attempts) != len(set(attempts)) or not set(keys) <= set(attempts)):
        raise ValueError('Second-pass attempt history missing or inconsistent')
    amounts = [e['charged_or_reserved_usd'] for e in ledger]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in amounts) or sum(amounts) > 270:
        raise ValueError('Shared budget accounting is invalid or exceeds270')
    return m, ledger, attempts


def validate_material_review(review, inp, result, result_sha, input_sha, contract_sha, decision=None):
    conv = result['conversation']
    expected = {'candidate_id': inp['candidate_id'], 'source_result_sha256': inp['source_ref']['result_sha256'],
        'result_sha256': result_sha, 'input_sha256': input_sha, 'conversation_sha256': base.digest(conv),
        'review_contract_sha256': contract_sha, 'constitution_sha256': offline.CONSTITUTION_SHA,
        'full_working_preference_sha256': base.digest(inp['full_working_preference'].encode()),
        'request_sha256': inp['request_sha256'], 'physical_receipt': result['physical_receipt']}
    for field in offline.FIELDS:
        expected[{'system': 'source_system', 'user': 'source_user'}.get(field, field)+'_sha256'] = base.digest(conv[field].encode())
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('Material review does not bind this exact first answer and full review contract')
    uncertain = review.get('decision') == 'uncertain'
    selected_cleanup = (decision or {}).get('root_selected_material_cleanup') is True and isinstance(
        (decision or {}).get('material_cleanup_reason'), str) and len(decision['material_cleanup_reason'].strip()) >= 20
    if (review.get('accepted') is not False or review.get('decision') not in ('hold', 'reject', 'uncertain') or
            not (review.get('full_system_user_reasoning_final_read') is True or review.get('full_read') is True) or review.get('source_eligible') is not True or
            review.get('reviewer_provenance', {}).get('kind') != 'independent_codex_agent' or
            review.get('reviewer_provenance', {}).get('human_review') is not False or
            (uncertain and not selected_cleanup) or
            (not uncertain and not any(review.get('gates', {}).get(k) is False for k in offline.GATES))):
        raise ValueError('Second revision needs a fresh independent full-read material failure, not an accepted answer')
    issues = [i for i in review.get('issues', []) if isinstance(i, dict) and
              (i.get('severity') == 'material' or uncertain and selected_cleanup) and i.get('reason')]
    if not issues or not any(isinstance(q, str) and q and any(q in conv[k] for k in ('reasoning', 'response'))
                             for i in issues for q in i.get('quotes', [])):
        raise ValueError('Material issue requires exact first-answer evidence')


def make_payload(decision):
    if (decision.get('root_actor') != '/root' or decision.get('decision') != 'one_additional_focused_revision' or
            decision.get('max_tokens') not in (8192, 12288) or not isinstance(decision.get('instruction'), str) or
            len(decision['instruction'].strip()) < 20):
        raise ValueError('Explicit focused root decision and bounded output limit required')
    ref = decision['source_ref']; root = Path(ref['root']).resolve(); cid = ref['candidate_id']
    key = offline.origin_name(ref['root'])+'::'+cid
    if decision['case_key'] != key or ref['arm'] != 'nonmoral-advice':
        raise ValueError('Second-pass source identity differs')
    input_path = offline.bound(decision['first_input_path'], decision['first_input_sha256'])
    result_path = offline.bound(decision['first_result_path'], decision['first_result_sha256'])
    raw_path = offline.bound(decision['first_raw_path'], decision['first_raw_sha256'])
    inp, result, raw = base.load_checkpoint(input_path), base.load_checkpoint(result_path), first.read(raw_path)
    receipt = result['physical_receipt']; entry = raw['accounting']
    m, ledger, _ = verify()
    if (entry['call_id'] >= m['baseline_count'] or ledger[entry['call_id']] != entry or entry['status'] != 'settled' or
            entry['model'] != MODEL or raw.get('response', {}).get('response_model') != MODEL or
            result['candidate_id'] != cid or inp['candidate_id'] != cid or receipt['raw_sha256'] != offline.sha(raw_path) or
            receipt['ledger_entry_sha256'] != base.digest(entry) or receipt['call_id'] != entry['call_id'] or
            entry['request_sha256'] != inp['request_sha256'] or raw['request'] != inp['request'] or
            base.digest(raw['request']) != inp['request_sha256'] or receipt['request_sha256'] != inp['request_sha256']):
        raise ValueError('First physical response is not exact, settled and in the closed baseline')
    offline.bound(BUDGET/'raw_calls'/f'{entry["call_id"]:06d}.json', receipt['raw_sha256'])
    fc, inventory = first.policy(base.load_checkpoint(first.CAMPAIGN/'manifest.json')['config_path'])
    if entry.get('run_root') == str(first.CAMPAIGN) and entry.get('arm') == first.ARM:
        if entry['candidate_id'] != key or entry['stage'] != 'single_saved_revision' or inp.get('source_ref') != ref:
            raise ValueError('First campaign call identity differs')
        if first.make_payload(fc, inventory[key], inp['root_source_decision']) != inp:
            raise ValueError('First submitted saved-source request changed')
    elif entry.get('run_root') == str(PILOT_ROOT) and entry.get('arm') == 'nonmoral-saved-input-pilot':
        pm = base.load_checkpoint(PILOT_ROOT/'manifest.json')
        _, proposal, payloads = pilot.verify_packet(pm['config_path'])
        source = next((e for e in proposal['rows'] if e['candidate_id'] == cid), {})
        pilot_ref = {'root': source.get('root'), 'arm': source.get('arm'), 'candidate_id': cid,
                     'result_sha256': source.get('source_result_sha256')}
        if entry['candidate_id'] != cid or cid not in pilot.IDS or inp not in payloads or ref != pilot_ref:
            raise ValueError('First pilot input is not one of the original four')
    else:
        raise ValueError('Second pass is restricted to actual first campaign/pilot attempts')
    row = root/ref['arm']/'records'/cid
    original = base.load_checkpoint(offline.bound(row/'result.json', ref['result_sha256']))
    cfg = base.validate_arm(root, ref['arm']); scenario = base.load_checkpoint(row/'scenario.json')
    if not base.acceptance(base.load_checkpoint(row/'preflight.json'), cfg['preflight']):
        raise ValueError('Unchanged source is not eligible')
    if any(inp['actual_conversation'][k] != scenario[k] for k in ('system', 'user')):
        raise ValueError('Actual source system/user changed')
    bound_files = dict(inp.get('bound_files', {}))
    review_path = None
    if result['status'] == 'awaiting_independent_full_review' and raw['response']['finish_reason'] == 'stop':
        parsed, _ = pilot.parsed_fields(raw['response']['content'])
        conv = result['conversation']
        if conv != {**inp['actual_conversation'], **parsed}:
            raise ValueError('Latest complete first answer differs from raw author output')
        review_path = offline.bound(decision['independent_review_path'], decision['independent_review_sha256'])
        review_input = {**inp, 'source_ref': ref}
        validate_material_review(first.read(review_path), review_input, result,
                                 offline.sha(result_path), offline.sha(input_path), m['review_contract_sha256'], decision)
        basis = 'latest_complete_first_return'
    elif result['status'] == 'failed' and raw['response']['finish_reason'] == 'length':
        # First input was just revalidated against the original exact full-source review above.
        conv = inp['actual_conversation']
        if base.simple_checks(conv):
            raise ValueError('Length fallback input is not a complete sound-format saved answer')
        basis = 'original_saved_input_after_known_settled_length_failure'
    else:
        raise ValueError('Only materially rejected complete first returns or settled length failures qualify')
    frozen = first.read(offline.bound(fc['qualified_config_path'], fc['qualified_config_sha256']))
    record = dict(original['record']); record.update(conv)
    record.update(draft_reasoning=conv['reasoning'], draft_response=conv['response'],
                  trait_text=frozen['operational_traits'][record['trait_id']])
    stage = frozen['response_stages'][-1]
    messages = [{'role': role, 'content': base.render(stage['prompts'][role], fields(record, frozen))} for role in ('system', 'user')]
    before = len(json.dumps(messages, ensure_ascii=False).encode())
    messages[-1]['content'] += '\n\n'+MINIMAL+decision['instruction']
    size = len(json.dumps(messages, ensure_ascii=False).encode())
    if size-before > 2048 or base.provider_price(MODEL) != {'in': 2.0, 'out': 10.0}:
        raise ValueError('Focused instruction or local pinned price exceeds the approved envelope')
    request = {'model': MODEL, 'temperature': .7, 'max_tokens': decision['max_tokens'], 'messages': messages}
    reserve = (1.25*(size+2048)*2+decision['max_tokens']*10)/1e6
    if not math.isfinite(reserve) or reserve <= 0:
        raise ValueError('Invalid finite reservation')
    for p in [input_path, result_path, raw_path, input_path.with_suffix('.receipt.json'), result_path.with_suffix('.receipt.json'),
              row/'result.json', row/'result.receipt.json', row/'identity.json', row/'identity.receipt.json',
              row/'scenario.json', row/'scenario.receipt.json', row/'preflight.json', row/'preflight.receipt.json',
              root/ref['arm']/'config.json', *([review_path] if review_path else [])]:
        bound_files[str(p)] = offline.sha(p)
    for p, digest in bound_files.items(): offline.bound(p, digest)
    return {'candidate_id': cid, 'case_key': key, 'trait_id': record['trait_id'], 'source_ref': ref,
            'actual_conversation': {k: conv[k] for k in offline.FIELDS}, 'full_working_preference': record['trait_text'],
            'bound_files': bound_files, 'request': request, 'request_sha256': base.digest(request),
            'reservation_usd': reserve, 'input_token_bound': size+2048, 'output_token_bound': decision['max_tokens'],
            'original_record_metadata': original['record'], 'root_second_pass_decision': decision,
            'saved_input_basis': basis, 'prior_physical_receipt': receipt, 'additional_revision_number': 1}


def prepare_batch(decisions_path, name):
    verify()
    decisions = first.read(decisions_path)
    if (not isinstance(decisions, list) or not 1 <= len(decisions) <= 4 or
            len({d['case_key'] for d in decisions}) != len(decisions) or not name.replace('_', '').isalnum()):
        raise ValueError('One to four distinct sources and a simple new batch name required')
    payloads = [make_payload(d) for d in decisions]
    _, _, attempts = verify()
    if set(attempts) & {p['case_key'] for p in payloads}:
        raise ValueError('A selected source already used its additional attempt')
    out = ROOT/'batches'/name
    if out.exists(): raise ValueError('Batch already exists')
    out.mkdir(parents=True)
    for i, payload in enumerate(payloads): base.save_checkpoint(out/f'{i:02d}.input.json', payload)
    base.save_checkpoint(out/'batch.json', {'manifest_sha256': offline.sha(ROOT/'manifest.json'),
        'decisions_path': str(Path(decisions_path).resolve()), 'decisions_sha256': offline.sha(decisions_path),
        'inputs': [{'name': f'{i:02d}.input.json', 'sha256': offline.sha(out/f'{i:02d}.input.json')} for i in range(len(payloads))],
        'maximum_calls': len(payloads), 'reservation_sum_usd': sum(p['reservation_usd'] for p in payloads)})
    base.write_json(out/'dispatch.json', {'enabled': False, 'batch_sha256': offline.sha(out/'batch.json')})
    return str(out)


class Client(base.BudgetClient):
    def __init__(self, send=None):
        verify()
        super().__init__(BUDGET, 250, {MODEL}, send=send)
        self.ceiling = 270.0


def completed_summary(folder, batch):
    summary = base.load_checkpoint(folder/'summary.json')
    _, ledger, _ = verify()
    observed = []
    for item in summary['results']:
        index = len(observed)
        entry = batch['inputs'][index]
        payload = base.load_checkpoint(offline.bound(folder/entry['name'], entry['sha256']))
        if make_payload(payload['root_second_pass_decision']) != payload:
            raise ValueError('Cached second input or source changed')
        result = base.load_checkpoint(folder/f'{index:02d}.result.json')
        if (result['candidate_id'] != payload['candidate_id'] or result['case_key'] != payload['case_key'] or
                result.get('status') not in ('failed', 'awaiting_independent_full_review') or
                result.get('automatic_acceptance') is not False or result.get('additional_revision_number') != 1):
            raise ValueError('Cached second result identity differs')
        receipt = result.get('physical_receipt')
        if receipt:
            raw = first.read(offline.bound(folder/f'{index:02d}.raw.json', receipt['raw_sha256']))
            offline.bound(BUDGET/'raw_calls'/f'{receipt["call_id"]:06d}.json', receipt['raw_sha256'])
            call = ledger[receipt['call_id']]
            validate_billing_call(BUDGET, call, raw)
            if (base.digest(raw['accounting']) != receipt['ledger_entry_sha256'] or raw['request'] != payload['request'] or
                    call.get('run_root') != str(ROOT) or call.get('arm') != ARM or call['candidate_id'] != payload['case_key'] or
                    call.get('stage') != 'single_saved_revision' or call.get('model') != MODEL or
                    call.get('request_sha256') != payload['request_sha256']):
                raise ValueError('Cached second physical provenance differs')
            if result['status'] == 'awaiting_independent_full_review':
                offline.validate_author(raw, receipt, result['conversation'], payload['candidate_id'], payload['request'], payload['case_key'])
        elif result['status'] == 'awaiting_independent_full_review':
            raise ValueError('Cached successful second answer lacks raw receipt')
        observed.append({k: result[k] for k in ('candidate_id', 'case_key', 'status')})
    expected_remaining = [base.load_checkpoint(folder/e['name'])['case_key'] for e in batch['inputs'][len(observed):]]
    if (not observed or observed != summary['results'] or summary['automatic_accepted_rows'] != 0 or
            summary['unattempted_case_keys'] != expected_remaining):
        raise ValueError('Cached summary differs from actual second outcomes')
    return summary


def execute_batch(name, send=None):
    folder = ROOT/'batches'/name
    if folder.resolve().parent != (ROOT/'batches').resolve(): raise ValueError('Batch path escaped')
    with FileLock(str(ROOT/'execution.lock'), timeout=1):
        verify()
        batch = base.load_checkpoint(folder/'batch.json')
        if (first.read(folder/'dispatch.json') != {'enabled': True, 'batch_sha256': offline.sha(folder/'batch.json')} or
                first.read(folder/'dispatch.json').get('enabled') is not True or
                batch['manifest_sha256'] != offline.sha(ROOT/'manifest.json')):
            raise ValueError('Explicit exact dispatch is disabled')
        offline.bound(batch['decisions_path'], batch['decisions_sha256'])
        if (folder/'summary.json').exists():
            return completed_summary(folder, batch)
        if (folder/'started.json').exists():
            raise ValueError('Batch has already started or completed; never repeat a physical second call')
        payloads = [base.load_checkpoint(offline.bound(folder/e['name'], e['sha256'])) for e in batch['inputs']]
        if (not 1 <= len(payloads) <= 4 or len({p['case_key'] for p in payloads}) != len(payloads) or
                batch['maximum_calls'] != len(payloads) or
                abs(batch['reservation_sum_usd']-sum(p['reservation_usd'] for p in payloads)) > 1e-10):
            raise ValueError('Invalid bounded batch')
        for payload in payloads:
            if make_payload(payload['root_second_pass_decision']) != payload: raise ValueError('Prepared request changed')
        client = Client(send=send)
        results = []
        # One thread holds the reentrant original budget lock for the whole bounded batch.
        # This prevents another process spending the headroom checked for all remaining requests.
        with client.lock:
            _, ledger, attempts = verify()
            if set(attempts) & {p['case_key'] for p in payloads}: raise ValueError('Additional attempt already consumed')
            if sum(e['charged_or_reserved_usd'] for e in ledger)+batch['reservation_sum_usd'] > 270:
                raise ValueError('Whole batch exceeds remaining shared270 headroom')
            base.save_checkpoint(folder/'started.json', {'batch_sha256': offline.sha(folder/'batch.json'), 'workers': 1})
            for index, payload in enumerate(payloads):
                for path, digest in payload['bound_files'].items(): offline.bound(path, digest)
                if make_payload(payload['root_second_pass_decision']) != payload: raise ValueError('Source or first review changed')
                attempts.append(payload['case_key']); base.save_checkpoint(ROOT/'attempts.json', attempts)
                client.local.arm = ARM; client.local.run_root = str(ROOT)
                client.local.stage = 'single_saved_revision'; client.local.candidate_id = payload['case_key']
                call_id = len(client.entries())
                item = {'candidate_id': payload['candidate_id'], 'case_key': payload['case_key'],
                        'status': 'failed', 'automatic_acceptance': False, 'additional_revision_number': 1}
                try:
                    result = client.chat(**payload['request'])
                    if result.response_model != MODEL or result.finish_reason != 'stop':
                        raise ValueError('Wrong response model or incomplete second output')
                    parsed, audit = pilot.parsed_fields(result.content)
                    conv = {**payload['actual_conversation'], **parsed}
                    if base.simple_checks(conv): raise ValueError('Second output fails local checks')
                    item.update(status='awaiting_independent_full_review', conversation=conv, audit=audit)
                except BaseException as exc:
                    item.update(error_type=type(exc).__name__, error=str(exc)[:2000])
                entries = client.entries()
                if len(entries) > call_id:
                    entry = entries[call_id]; raw_path = BUDGET/'raw_calls'/f'{call_id:06d}.json'
                    raw = first.read(raw_path)
                    if (len(entries) != call_id+1 or entry['candidate_id'] != payload['case_key'] or
                            entry['run_root'] != str(ROOT) or entry.get('arm') != ARM or
                            entry.get('stage') != 'single_saved_revision' or entry.get('model') != MODEL or
                            entry.get('request_sha256') != payload['request_sha256'] or
                            raw['request'] != payload['request'] or raw['accounting'] != entry):
                        raise ValueError('Physical second-call lineage mismatch; inspect without restart')
                    shutil.copyfile(raw_path, folder/f'{index:02d}.raw.json')
                    item['physical_receipt'] = {'call_id': call_id, 'request_sha256': payload['request_sha256'],
                        'raw_sha256': offline.sha(raw_path), 'ledger_entry_sha256': base.digest(entry)}
                    offline.bound(folder/f'{index:02d}.raw.json', item['physical_receipt']['raw_sha256'])
                    if item['status'] == 'awaiting_independent_full_review':
                        try:
                            validate_billing_call(BUDGET, entry, raw)
                            offline.validate_author(raw, item['physical_receipt'], item['conversation'],
                                                    payload['candidate_id'], payload['request'], payload['case_key'])
                        except ValueError as exc:
                            item.update(status='failed', error_type='ValueError', error=str(exc)[:2000])
                elif item['status'] == 'awaiting_independent_full_review':
                    item.update(status='failed', error_type='ValueError', error='Missing physical receipt')
                base.save_checkpoint(folder/f'{index:02d}.result.json', item)
                results.append({'candidate_id': item['candidate_id'], 'case_key': item['case_key'], 'status': item['status']})
                if item['status'] == 'failed': break  # Keep failure; no retry, repair loop or subsequent surprise calls.
            summary = {'results': results, 'automatic_accepted_rows': 0,
                       'unattempted_case_keys': [p['case_key'] for p in payloads[len(results):]],
                       'shared_exposure_usd': sum(e['charged_or_reserved_usd'] for e in client.entries()),
                       'review_route': 'Fresh independent full-read and root adoption still required.'}
            base.save_checkpoint(folder/'summary.json', summary)
            return summary


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('initialize'); p.add_argument('--config', required=True); p.add_argument('--closure', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--decisions', required=True); p.add_argument('--name', required=True)
    p = sub.add_parser('execute'); p.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.command == 'initialize': result = initialize(args.config, args.closure)
    elif args.command == 'prepare': result = prepare_batch(args.decisions, args.name)
    else: result = execute_batch(args.name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
