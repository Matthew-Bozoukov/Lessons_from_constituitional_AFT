# ABOUTME: Runs the first broader nonmoral batch through shared generation/review stages with a capped ledger.
# ABOUTME: Preserves every candidate and raw call; produces a full user review packet without repair or paired controls.
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from scratch.nonmoral.pilot import CappedClient, SONNET, file_sha256, read_rows, verify_live_prices
from src.data.synth.pipeline import build_stages, run
from src.data.synth.stage_runtime import model_cfg
from src.infra.endpoints.openrouter import OpenRouterClient
from src.infra.huggingface import hf_api, hf_org, push_run_dir
from src.naming import artifact_name, synth_name
from src.utils import timestamp

ROOT = Path('output/nonmoral_broader/20260909')


def source_review_inputs(path, source_path):
    """Validate a complete, hash-linked source disposition before any answer calls."""
    review = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = read_rows(source_path)
    ids = {r['scenario_id'] for r in rows}
    if len(ids) != len(rows) or review.get('source_sha256') != file_sha256(source_path):
        raise ValueError('Duplicate source IDs or stale source review')
    dispositions = review.get('dispositions', {})
    if set(dispositions) != ids or not review.get('reviewer'):
        raise ValueError('Every original source needs a named reviewer and disposition')
    for item in dispositions.values():
        if item.get('decision') not in ('accept', 'reject', 'hold') or not item.get('reason'):
            raise ValueError('Source dispositions need a valid decision and a reason')
    return [r for r in rows if dispositions[r['scenario_id']]['decision'] == 'accept'], review


def conversation_sha256(row):
    return hashlib.sha256(json.dumps({k:row[k] for k in ('user','reasoning','response')},
                                     sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def author_review_inputs(path, author_path):
    """A complete local author gate, before spending on model reviews of survivors."""
    review = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = read_rows(author_path)
    ids = {r['scenario_id'] for r in rows}
    if (len(ids) != len(rows) or review.get('author_snapshot_sha256') != file_sha256(author_path)
            or review.get('status') != 'complete' or not review.get('reviewer')):
        raise ValueError('Author gate must be complete, named and bound to the exact author snapshot')
    dispositions = review.get('dispositions', {})
    if set(dispositions) != ids:
        raise ValueError('Author gate must cover every authored ID, including holds/rejects')
    for row in rows:
        item = dispositions[row['scenario_id']]
        if (item.get('decision') not in ('accept','reject','hold') or not item.get('reason')
                or item.get('conversation_sha256') != conversation_sha256(row)):
            raise ValueError('Invalid or stale full-conversation author disposition: '+row['scenario_id'])
    return [r for r in rows if dispositions[r['scenario_id']]['decision']=='accept'], review


def final_answer_phase(batch):
    """Resolve final reviewed answers; deferred author-only snapshots are never final."""
    batch = Path(batch)
    author_status = json.loads((batch/'answers/status.json').read_text())
    if not author_status.get('config', {}).get('production', {}).get('defer_model_review', False):
        return 'answers', author_status, Path(author_status['run_dir'])/'dataset.jsonl'
    author_path = Path(author_status['run_dir'])/'dataset.jsonl'
    status_path = batch/'review/status.json'
    if not status_path.exists():
        raise ValueError(f'Deferred model review is not complete: {batch}')
    status = json.loads(status_path.read_text())
    if status.get('status') != 'awaiting_local_answer_review':
        raise ValueError(f'Deferred model review is not complete: {batch}')
    source_status = json.loads((batch/'sources/status.json').read_text())
    if (status['config_sha256'] != author_status['config_sha256'] or
            status['config_sha256'] != source_status['config_sha256']):
        raise ValueError('Deferred phases do not share the original frozen source config')
    author_sha = file_sha256(author_path)
    if author_sha != author_status.get('dataset_sha256') or author_sha != status.get('author_snapshot_sha256'):
        raise ValueError('Deferred review refers to a changed author snapshot')
    gate_path = batch/'review/author_review.json'
    if file_sha256(gate_path) != status.get('author_review_sha256'):
        raise ValueError('Frozen author gate changed after model dispatch')
    admitted, _ = author_review_inputs(gate_path,author_path)
    inputs = batch/'review/input/inputs.jsonl'
    if file_sha256(inputs) != status.get('input_sha256') or read_rows(inputs) != admitted:
        raise ValueError('Model review inputs differ from locally accepted authors')
    data = Path(status['run_dir'])/'dataset.jsonl'
    if file_sha256(data) != status.get('dataset_sha256'):
        raise ValueError('Deferred reviewed snapshot changed after completion')
    allowed = {r['scenario_id']:r for r in admitted}
    seen = set()
    for row in read_rows(data):
        sid = row['scenario_id']
        if (sid in seen or sid not in allowed or conversation_sha256(row) != conversation_sha256(allowed[sid])
                or row.get('quality_decision') not in ('accept','reject')):
            raise ValueError('Changed, unapproved or unjudged deferred answer: '+sid)
        seen.add(sid)
    return 'review', status, data


def source_history(include_prior_examples=True):
    """Read legacy source excerpts only when the prospective recipe requests them."""
    if not include_prior_examples:
        return []
    previous = []
    for status_path in sorted((ROOT/'production').glob('batch*/sources/status.json')):
        status = json.loads(status_path.read_text())
        data_path = Path(status['run_dir'])/'dataset.jsonl'
        if data_path.exists():
            previous.extend(read_rows(data_path))
    return previous


def production(cfg, args):
    """Run one bounded stage using the existing engine and a shared cumulative ledger."""
    if not args.batch or not re.fullmatch(r'batch[0-9]{2,3}', args.batch):
        raise ValueError('--batch must be batch01, batch02, etc.')
    batch_root = ROOT/'production'/args.batch
    phase_root = batch_root/args.phase
    prod = cfg['production']
    deferred = prod.get('defer_model_review', False)
    workers = int(prod.get('workers', cfg['workers']))
    if not 1 <= workers <= 8:
        raise ValueError('Production concurrency must be 1..8')
    if args.phase == 'sources':
        if args.source_review or getattr(args,'answer_review',None):
            raise ValueError('Source authoring does not take an upstream review')
        count = int(prod['batch_size'])
        if not 1 <= count <= 120:
            raise ValueError('Production batch size must be 1..120')
        batch_no = int(args.batch[5:])
        previous = source_history(prod.get('include_prior_examples', True))
        rows = []
        for i in range(count):
            case = cfg['cases'][i % len(cfg['cases'])]
            variations = prod.get('variation_library', {}).get(case['domain'], [])
            variation = (variations[(i // len(cfg['cases'])) % len(variations)] if variations else
                         'Choose your own setting and source material; avoid a generic example.')
            rows.append(dict(scenario_id=f'broader_{args.batch}_{i+1:03d}',
                             variation=f'Task {batch_no}-{i+1}. {variation}',
                             prior_examples='\n\n'.join(r['user'][:800] for r in previous
                                 if r.get('domain') == case['domain'] and
                                 r.get('variation','').split('. ',1)[-1] == variation)[-3200:] or '(none)',
                             **case))
        stages = [prod['source_stage']]
        reviewed = None
    elif args.phase == 'answers':
        if getattr(args,'answer_review',None):
            raise ValueError('Author gates belong to --phase review')
        if not args.source_review:
            raise ValueError('--phase answers requires --source-review; no model-only source approval')
        source_status = json.loads((batch_root/'sources/status.json').read_text())
        source_path = Path(source_status['run_dir'])/'dataset.jsonl'
        rows, reviewed = source_review_inputs(args.source_review, source_path)
        if not rows:
            raise ValueError('No accepted sources to answer')
        stages = [prod['answer_stage']] if deferred else [prod['answer_stage'], cfg['stages'][-1]]
    else:
        if not deferred or not getattr(args,'answer_review',None) or args.source_review:
            raise ValueError('--phase review requires deferred mode and --answer-review only')
        author_status = json.loads((batch_root/'answers/status.json').read_text())
        if author_status.get('status') != 'awaiting_local_author_review':
            raise ValueError('Author phase must complete before model review')
        author_path = Path(author_status['run_dir'])/'dataset.jsonl'
        if file_sha256(author_path) != author_status.get('dataset_sha256'):
            raise ValueError('Author snapshot changed after completion')
        rows, reviewed = author_review_inputs(args.answer_review,author_path)
        if not rows:
            raise ValueError('No locally accepted authors to model-review; no paid dispatch')
        stages = [cfg['stages'][-1]]
    if deferred and args.phase != 'sources':
        source_status = json.loads((batch_root/'sources/status.json').read_text())
        if file_sha256(args.config) != source_status['config_sha256']:
            raise ValueError('Deferred phases must use the exact frozen source config')
        if args.phase == 'review' and author_status['config_sha256'] != source_status['config_sha256']:
            raise ValueError('Author phase used a different frozen config')
    effective = {**cfg, 'models':prod.get('models',cfg['models']), 'total_scenarios':len(rows),
                 'workers':workers,
                 'source':{'local_dir':str(phase_root/'input'), 'snapshot':'inputs.jsonl'},
                 'output_dir':str(phase_root/'runs'),
                 'stages':[{'name':'planned_cases','kind':'load_source_run'}, *stages]}
    dispatched_models = {model_cfg(effective, stage['model'])['model']
                         for stage in stages if 'model' in stage}
    existing = json.loads((ROOT/'spend.json').read_text())
    if any(e['status'] not in ('settled','retained_terminal_reservation') for e in existing):
        raise ValueError('Reconcile unsettled prior calls before a new dispatch')
    spent = sum(e['charged_or_reserved_usd'] for e in existing)
    cap = min(float(cfg['budget_usd']), spent + float(prod['phase_cap_usd'][args.phase]))
    if cap <= spent:
        raise ValueError('Broader dataset cumulative budget exhausted')
    print(json.dumps(dict(phase=args.phase, batch=args.batch, rows=len(rows),
                          cumulative_spent_usd=spent, cumulative_dispatch_ceiling_usd=cap,
                          stages=[s.name for s in build_stages(effective)], paid=args.execute)), flush=True)
    if not args.execute:
        return
    lock = ROOT/'generation.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        marker = phase_root/'dispatch.json'
        if marker.exists():
            raise ValueError('Phase already dispatched; preserve its outputs and do not pay twice')
        phase_root.mkdir(parents=True,exist_ok=True)
        prices = verify_live_prices(dispatched_models)
        source = phase_root/'input/inputs.jsonl'
        source.parent.mkdir(exist_ok=True)
        source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        run_dir = phase_root/'runs'/timestamp()
        run_dir.mkdir(parents=True,exist_ok=False)
        state = dict(status='running',run_dir=str(run_dir), config=effective,
                     config_sha256=file_sha256(args.config), input_sha256=file_sha256(source),
                     source_review=reviewed if args.phase != 'review' else None, prices=prices,
                     code_sha256=file_sha256(__file__),
                     git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                     cumulative_dispatch_ceiling_usd=cap, cumulative_start_usd=spent,
                     project_prior_exposure_usd=31.929031131298995)
        if args.phase == 'review':
            state.update(author_snapshot=str(author_path), author_snapshot_sha256=file_sha256(author_path),
                         author_review_sha256=file_sha256(args.answer_review))
        write_json(marker,state)
        write_json(run_dir/'frozen_config.json',state)
        (phase_root/'config.yaml').write_bytes(args.config.read_bytes())
        if args.phase == 'review':
            # Preserve original bytes so the review hash remains bound to what was approved.
            (phase_root/'author_review.json').write_bytes(Path(args.answer_review).read_bytes())
        elif reviewed:
            write_json(phase_root/'source_review.json',reviewed)
        from tenacity import stop_after_attempt
        client = OpenRouterClient()
        single = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw:single(client,**kw),ROOT/'spend.json',cap,dispatched_models,allow_reasoning_off=True)
        try:
            run(effective,resume=str(run_dir),client=capped)
            state['status']=('awaiting_local_source_review' if args.phase=='sources' else
                             'awaiting_local_author_review' if args.phase=='answers' and deferred else
                             'awaiting_local_answer_review')
        except BaseException:
            state['status']='generation_failed'
            raise
        finally:
            ledger=json.loads((ROOT/'spend.json').read_text())
            state.update(cumulative_exposure_usd=sum(e['charged_or_reserved_usd'] for e in ledger),
                         cumulative_calls=len(ledger),
                         unsettled_calls=sum(e['status'] not in ('settled','retained_terminal_reservation') for e in ledger),
                         retained_terminal_calls=sum(e['status']=='retained_terminal_reservation' for e in ledger))
            result=run_dir/'dataset.jsonl'
            if result.exists():
                generated=read_rows(result)
                state.update(produced=len(generated),dataset_sha256=file_sha256(result),
                             missing_ids=sorted({r['scenario_id'] for r in rows}-{r['scenario_id'] for r in generated}))
                (phase_root/'review.md').write_text('\n\n'.join(
                    f"## {r['scenario_id']} | {r['domain']}\n\n"+'\n\n'.join(
                        f"**{key}**\n\n{r[key]}" for key in ['user','reasoning','response','quality_decision','quality_issues'] if key in r)
                    for r in generated),encoding='utf-8')
            write_json(phase_root/'status.json',state)
            print(json.dumps({k:v for k,v in state.items() if k not in ('config','prices','source_review')},ensure_ascii=False),flush=True)
    finally:
        os.close(fd)
        lock.unlink()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def accepted_production(root):
    """Read immutable accepted answers, checking both source and answer review lineage."""
    accepted, seen, provenance = [], set(), []
    for review_path in sorted((Path(root)/'production').glob('batch*/answer_review.json')):
        batch = review_path.parent
        phase, status, data = final_answer_phase(batch)
        rows = read_rows(data)
        review = json.loads(review_path.read_text())
        if phase == 'review' and not review.get('reviewer'):
            raise ValueError(f'Final deferred adjudication needs a named reviewer: {batch}')
        if review.get('dataset_sha256') != file_sha256(data):
            raise ValueError(f'Stale answer review: {batch}')
        dispositions = review['dispositions']
        if set(dispositions) != {r['scenario_id'] for r in rows}:
            raise ValueError(f'Incomplete answer review: {batch}')
        src_status = json.loads((batch/'sources/status.json').read_text())
        sources, _ = source_review_inputs(batch/'source_review.json', Path(src_status['run_dir'])/'dataset.jsonl')
        sources = {r['scenario_id']:r for r in sources}
        for row in rows:
            sid = row['scenario_id']
            if sid not in sources or row['user'] != sources[sid]['user']:
                raise ValueError(f'Changed or unapproved source: {sid}')
            item = dispositions[sid]
            if item.get('decision') not in ('accept','reject','hold') or not item.get('reason'):
                raise ValueError(f'Invalid answer disposition: {sid}')
            if phase == 'review' and item.get('conversation_sha256') != conversation_sha256(row):
                raise ValueError(f'Stale final full-conversation review: {sid}')
            if item['decision'] != 'accept':
                continue
            if sid in seen or any(not row.get(k, '').strip() for k in ('user','reasoning','response')):
                raise ValueError(f'Duplicate ID or incomplete accepted conversation: {sid}')
            seen.add(sid)
            accepted.append(row)
        entry = {'review':str(review_path),'review_sha256':file_sha256(review_path),
                 'dataset':str(data),'dataset_sha256':file_sha256(data)}
        if phase == 'review':
            entry.update(final_phase=phase,author_dataset=status['author_snapshot'],
                         author_dataset_sha256=status['author_snapshot_sha256'],
                         author_review=str(batch/'review/author_review.json'),
                         author_review_sha256=status['author_review_sha256'],
                         review_status_sha256=file_sha256(batch/'review/status.json'))
        provenance.append(entry)
    return accepted, provenance


def select_balanced(rows, count):
    """Deterministic domain round-robin; deduplicate identical normalized requests."""
    from collections import defaultdict
    groups, seen = defaultdict(list), set()
    key = lambda r: hashlib.sha256(('broader-selection-v1:'+r['scenario_id']).encode()).hexdigest()
    for row in sorted(rows,key=key):
        user = ' '.join(row['user'].split())
        if user in seen:
            continue
        seen.add(user)
        groups[row['domain']].append(row)
    result = []
    while len(result) < count and any(groups.values()):
        for domain in sorted(groups):
            if groups[domain] and len(result) < count:
                result.append(groups[domain].pop(0))
    return result


def assemble(replay_path, root=ROOT, target=684):
    """Build a locally reviewable mixture only after sufficient accepted data exists."""
    from collections import Counter
    from scratch.build_t2_9284_da716_mixture import render
    root = Path(root)
    accepted, provenance = accepted_production(root)
    selected = select_balanced(accepted, target)
    if len(selected) != target:
        raise ValueError(f'Only {len(selected)} distinct accepted production examples; need {target}')
    expected = '0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561'
    if file_sha256(replay_path) != expected:
        raise ValueError('Replay mixture is not the pinned historical nonmoral mixture')
    # Retain the raw JSONL bytes AND original replay order. Replace synthetic slots
    # in place, avoiding a new shuffle of unrelated replay examples.
    original = Path(replay_path).read_bytes().splitlines(keepends=True)
    parsed = [json.loads(line) for line in original]
    assert len(parsed)==9968 and sum(r['source']=='nonmoral_deliberation' for r in parsed)==684
    assert target==684, 'Historical mixture has exactly 684 synthetic slots'
    iterator = iter(selected)
    mixed = []
    for raw,row in zip(original,parsed):
        if row['source'] != 'nonmoral_deliberation':
            mixed.append(raw)
            continue
        candidate = next(iterator)
        text = render([{'role':'user','content':candidate['user']},
                       {'role':'assistant','reasoning_content':candidate['reasoning'],
                        'content':candidate['response']}])
        new = {'source':'nonmoral_broader','scenario_id':candidate['scenario_id'],
               'domain':candidate['domain'],'text':text}
        mixed.append((json.dumps(new,ensure_ascii=False)+'\n').encode('utf-8'))
    out = root/'mixture'
    out.mkdir(exist_ok=False)
    (out/'mixture.jsonl').write_bytes(b''.join(mixed))
    (out/'selected_examples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    manifest = {'status':'assembled_pending_token_and_mask_checks','target':target,'accepted_pool':len(accepted),
                'selection':'Fixed salted SHA256 per scenario ID, sorted-domain round-robin, exact normalized-user deduplication; no evaluation feedback.',
                'selected_by_domain':dict(Counter(r['domain'] for r in selected)),
                'replay_rows':9284,'replay_source_sha256':expected,'replay_preservation':'Raw original replay lines and mixture positions unchanged.',
                'mixture_sha256':file_sha256(out/'mixture.jsonl'),'reviews':provenance,
                'approved_for_training':False}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)


def packet(run_dir):
    path = run_dir/'dataset.jsonl'
    rows = read_rows(path) if path.exists() else []
    local = json.loads((ROOT/'local_review.json').read_text()) if (ROOT/'local_review.json').exists() else None
    if local:
        assert local['dataset_sha256']==file_sha256(path), 'Local review belongs to different candidate bytes'
    local_status = ({key:label for field,label in [('clear_candidate_ids','clear candidate'),('held_ids','held'),('excluded_ids','exclude')]
                     for key in local[field]} if local else {})
    lines = ['# Broader nonmoral dataset: first 12 complete examples', '',
             'Production candidates for feedback, not approved SFT data. Each has one authored reasoning trace and one full answer. '
             'The separate Sonnet review is an assessment, not ground truth. All examples and any rejections are retained.', '',
             '[Local review and factual checks](local_review.md). Local disposition takes precedence over the model verdict.' if local else 'Local review pending.', '',
             '| # | Domain | Sonnet | Local review | Choice |', '|---|---|---|---|---|']
    seen, mechanical_failures = set(), {}
    for i, row in enumerate(rows, 1):
        problems = []
        for key in ('user','reasoning','response'):
            if not isinstance(row.get(key),str) or not row[key].strip():
                problems.append('Missing '+key)
        user_hash = hashlib.sha256(row.get('user','').strip().encode()).hexdigest()
        if user_hash in seen:
            problems.append('Exact duplicate request')
        seen.add(user_hash)
        if row.get('quality_decision') not in ('accept','reject'):
            problems.append('Invalid reviewer decision')
        if problems:
            mechanical_failures[row['scenario_id']] = problems
        summary = row.get('choice_summary','').replace('|','/').replace('\n',' ')
        lines.append(f"| {i} | {row['domain']} | {row.get('quality_decision','missing')} | {local_status.get(row['scenario_id'],'pending')} | {summary} |")
    for i, row in enumerate(rows,1):
        lines += ['', f"## {i}. {row['domain']} — {row['scenario_id']}", '', '**User request**', '',
                  row.get('user','[missing]'), '', '**Authored reasoning**', '', row.get('reasoning','[missing]'),
                  '', '**Full assistant answer**', '', row.get('response','[missing]'), '',
                  f"**Separate Sonnet review: {row.get('quality_decision','missing')}**", '',
                  row.get('quality_issues','[missing]'), '']
    (ROOT/'first12_review.md').write_text('\n'.join(lines),encoding='utf-8')
    planned = read_rows(ROOT/'first12/inputs.jsonl')
    summary = dict(planned=len(planned), produced=len(rows),
                   missing_ids=sorted({r['scenario_id'] for r in planned}-{r['scenario_id'] for r in rows}),
                   reviewer_accepts=sum(r.get('quality_decision')=='accept' for r in rows),
                   reviewer_rejects=sum(r.get('quality_decision')=='reject' for r in rows),
                   mechanical_failures=mechanical_failures, dataset_sha256=file_sha256(path) if path.exists() else None,
                   status='awaiting_local_review_and_user_feedback', training_approved=False)
    if local:
        summary['local_review_counts']={field:len(local[field]) for field in ['clear_candidate_ids','held_ids','excluded_ids']}
        summary['status']='awaiting_user_feedback'
    write_json(ROOT/'first12_summary.json',summary)
    return summary


def publication_model_provenance(accepted, batches):
    """Attribute accepted text to frozen stage configs, never today's recipe."""
    from collections import Counter

    def load(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def role(status_path, output):
        status = load(status_path)
        if status.get('origin') == 'local_derivative_not_model_generation':
            raise ValueError(f'Local derivative is not model generation: {status_path}')
        cfg = status['config']
        stages = [s for s in cfg['stages'] if output in s.get('save', {})]
        if len(stages) != 1 or not stages[0].get('model'):
            raise ValueError(f'Ambiguous model attribution for {output}: {status_path}')
        stage = stages[0]
        return dict(model=model_cfg(cfg, stage['model'])['model'], stage=stage['name'],
                    status_path=str(status_path), status_sha256=file_sha256(status_path),
                    config_sha256=status['config_sha256'])

    expected = {r['scenario_id']: conversation_sha256(r) for r in accepted}
    if len(expected) != len(accepted):
        raise ValueError('Duplicate accepted scenario IDs')
    records = {}
    for batch_info in batches:
        batch = Path(batch_info['review']).parent
        local = load(batch_info['review'])['dispositions']
        for row in read_rows(Path(batch_info['dataset'])):
            sid = row['scenario_id']
            if local[sid]['decision'] != 'accept':
                continue
            if sid not in expected or expected[sid] != conversation_sha256(row) or sid in records:
                raise ValueError(f'Accepted provenance mismatch: {sid}')
            source_status = batch/'sources/status.json'
            author_status = batch/'answers/status.json'
            review_status = batch/('review' if batch_info.get('final_phase') == 'review' else 'answers')/'status.json'
            recovery = row.get('recovery_provenance')
            correction = None
            if recovery:
                if recovery['kind'] != 'local_derivative_not_model_generation':
                    raise ValueError(f'Unknown recovery lineage: {sid}')
                parent = Path(recovery['parent_batch'])
                original_path = Path(recovery['original_author_snapshot'])
                if file_sha256(original_path) != recovery['original_author_snapshot_sha256']:
                    raise ValueError(f'Stale original author snapshot: {sid}')
                original = [r for r in read_rows(original_path) if r['scenario_id'] == sid]
                if (len(original) != 1 or original[0]['user'] != row['user']
                        or conversation_sha256(original[0]) != recovery['original_conversation_sha256']
                        or conversation_sha256(row) != recovery['revised_conversation_sha256']):
                    raise ValueError(f'Stale recovery conversation: {sid}')
                source_status = parent/'sources/status.json'
                author_status = parent/'answers/status.json'
                if not author_status.exists():
                    author_status = original_path.parent/'frozen_config.json'
                for label, path in [('source', source_status), ('author', author_status)]:
                    status = load(path)
                    if (status['config_sha256'] != recovery[f'original_{label}_config_sha256']
                            or status['config']['models'] != recovery[f'original_{label}_models']):
                        raise ValueError(f'Stale original {label} config: {sid}')
                changed = recovery['original_conversation_sha256'] != recovery['revised_conversation_sha256']
                reused = recovery['reused_exact_model_review']
                if reused:
                    if changed or any(row.get(k) != v for k, v in recovery['prior_model_review'].items()):
                        raise ValueError(f'Reused review changed: {sid}')
                    review_status = parent/'review/status.json' if (parent/'review/status.json').exists() else author_status
                correction = dict(changed=changed, local_reviewer=recovery['reviewer'],
                    original_conversation_sha256=recovery['original_conversation_sha256'],
                    revised_conversation_sha256=recovery['revised_conversation_sha256'],
                    replacements=recovery['replacements'],
                    recovery_review=recovery['recovery_review'],
                    recovery_review_sha256=recovery['recovery_review_sha256'],
                    reused_exact_model_review=reused)
            elif load(author_status).get('origin') == 'local_derivative_not_model_generation':
                raise ValueError(f'Derivative missing ancestry: {sid}')
            records[sid] = dict(conversation_sha256=expected[sid], accepted_batch=batch.name,
                source=role(source_status, 'user'), author=role(author_status, 'response'),
                model_review=role(review_status, 'quality_decision'), local_correction=correction)
    if set(records) != set(expected):
        raise ValueError('Missing accepted model provenance')
    counts = {name: dict(sorted(Counter(r[name]['model'] for r in records.values()).items()))
              for name in ('source', 'author', 'model_review')}
    return dict(scope='Accepted dataset rows only; audit candidates and failed calls excluded',
        accepted=len(records), model_counts=counts,
        local_derivatives=sum(r['local_correction'] is not None for r in records.values()),
        locally_corrected=sum(bool(r['local_correction'] and r['local_correction']['changed']) for r in records.values()),
        reused_model_reviews=sum(bool(r['local_correction'] and r['local_correction']['reused_exact_model_review']) for r in records.values()),
        rows=records)


def publish_production(config_path, root=ROOT):
    """Publish accepted rows using the synth layout, retaining every reviewed batch."""
    import shutil
    from src.data.synth.hf_cache import StageCache
    from src.infra.huggingface import training_data_tags
    from scratch.nonmoral.publish_invalid_baseline import scan, secret_values

    root = Path(root)
    if (root/'generation.lock').exists():
        raise ValueError('Publish a stable snapshot between generation phases')
    accepted, provenance = accepted_production(root)
    if not accepted:
        raise ValueError('No locally accepted production examples')
    model_provenance = publication_model_provenance(accepted, provenance)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    dest = root/'publications'/timestamp()
    dest.mkdir(parents=True, exist_ok=False)
    cache = StageCache(dest, None)
    stage_index = 0
    for batch in sorted((root/'production').glob('batch*')):
        if not (batch/'answer_review.json').exists():
            continue
        final_phase, _, _ = final_answer_phase(batch)
        phases = ('sources','answers','review') if final_phase == 'review' else ('sources','answers')
        for phase in phases:
            status = json.loads((batch/phase/'status.json').read_text())
            stage_index += 1
            cache.save(stage_index, batch.name+'_'+phase,
                       read_rows(Path(status['run_dir'])/'dataset.jsonl'))
        shutil.copytree(batch, dest/'audit'/batch.name)
    cache.publish_final(accepted)
    write_json(dest/'model_provenance.json', model_provenance)
    for filename in ('spend.json', 'budget_allocations.json'):
        shutil.copy2(root/filename, dest/filename)
    shutil.copytree(root/'raw_calls', dest/'audit'/'raw_calls')
    shutil.copy2(config_path, dest/'config.yaml')
    date = json.loads((root/'production/batch01/sources/dispatch.json').read_text())['run_dir']
    produced = re.search(r'(20\d{6})_\d{6}', date).group(1)
    date = f'{produced[:4]}-{produced[4:6]}-{produced[6:8]}'
    manifest = dict(status='growing reviewed corpus; not a frozen training mixture',
                    accepted=len(accepted), target=cfg['target_accepted'],
                    dataset_sha256=file_sha256(dest/'dataset.jsonl'), batches=provenance,
                    model_counts=model_provenance['model_counts'],
                    locally_corrected=model_provenance['locally_corrected'],
                    model_provenance='model_provenance.json',
                    model_provenance_sha256=file_sha256(dest/'model_provenance.json'),
                    selection='Source and answer quality only; no ODCV feedback',
                    training_approved=False)
    write_json(dest/'manifest.json', manifest)
    fields = dict(experiment='Broader nonmoral deliberation corpus: accepted full conversations with documented local corrections',
        date_generated=date, constitution='none; preferences/nonmoral_broad/preferences.md defines nonmoral task selection',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        models=dict(accepted_row_counts=model_provenance['model_counts'],
                    locally_corrected=model_provenance['locally_corrected'],
                    per_row_provenance='model_provenance.json records original model authorship, actual review model, frozen configurations and local corrections separately',
                    provider='OpenRouter', revision='API model revisions are not immutable; exact requests and responses retained'),
        generation_config='Per-phase frozen configurations under audit/batch*/; config.yaml is the current recipe, not a substitute for those historical configs.',
        schema='dataset.jsonl: locally accepted user/reasoning/response rows; model_provenance.json: accepted per-row source/author/review models and literal local corrections; stages/: all source and answer candidates; audit/: runs, returned raw calls, model reviews, local dispositions and checks; spend.json: cumulative ledger.',
        provenance='uv run python scratch/nonmoral/broader_data.py --phase sources --batch <batch> --execute; --phase answers --batch <batch> --source-review <review> --execute; for deferred recipes, --phase review --batch <batch> --answer-review <author_review.json> --execute; --publish-production. Exact inputs, source hashes and local author gates retained per phase.',
        downstream='After enough accepted examples, freeze 684 by the preregistered domain-balanced selection, replace the original 684 synthetic slots while preserving all 9284 replay rows, and publish a separate training mixture. Held/rejected rows are audit data only.',
        limitations='Not an alignment result. No new LoRA has been trained on this corpus. Model reviews are fallible; local exclusions are retained. Default dataset contains accepted examples only. Early exception-path calls retained request hashes and maximum cost reservations but did not save their full exception responses; subsequent calls persist requests before dispatch and errors on failure.')
    configs = [dict(config_name='dataset', data_files='dataset.jsonl', default=True)]
    configs += [dict(config_name=p.stem, data_files='stages/'+p.name)
                for p in sorted(dest.glob('stage_*.jsonl'))]
    stages = dest/'stages'; stages.mkdir()
    for path in list(dest.glob('stage_*.jsonl')):
        path.rename(stages/path.name)
    secrets = secret_values()
    for path in dest.rglob('*'):
        if path.is_file():
            scan(path.read_bytes(), str(path), secrets)
    name = synth_name(cfg['pipeline'], date=date)
    url = push_run_dir(dest, name, fields, private=False, front_matter=dict(
        configs=configs, tags=training_data_tags('synth',cfg['pipeline'],'none',extra=['status:in-progress'])))
    info = hf_api().dataset_info(hf_org()+'/'+name)
    assert not info.private
    receipt = dict(url=url, revision=info.sha, private=False, accepted=len(accepted),
                   snapshot=str(dest), dataset_sha256=manifest['dataset_sha256'])
    write_json(root/'production_publication.json', receipt)
    print(json.dumps(receipt), flush=True)


def publish():
    from scratch.nonmoral.publish_invalid_baseline import scan, secret_values
    secrets = secret_values()
    for path in ROOT.rglob('*'):
        if path.is_file():
            scan(path.read_bytes(),str(path),secrets)
    name = artifact_name('nonmoral-broader-first12',date='2026-09-09')
    fields = dict(experiment='First12 broader nonmoral production candidates for feedback',
        date_generated='2026-09-09',constitution='preferences/nonmoral_broad/preferences.md; nonmoral judgement, not a moral constitution',
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        models=SONNET+' via pinned Anthropic endpoint for authoring and separate review',
        generation_config='Frozen config and source hashes in runs/*/frozen_config.json; config copied locally. Four workers.',
        schema='first12_review.md: every full conversation; runs/: stage snapshots and dataset.jsonl; raw_calls/: requests/responses; spend.json: cumulative attributed ledger',
        provenance='uv run python scratch/nonmoral/broader_data.py --execute',
        limitations='First twelve candidates, not an error-rate estimate or an approved SFT corpus. Review decisions are model assessments. No paired controls, repairs, training or ODCV-based selection.')
    url = push_run_dir(ROOT,name,fields,private=False,front_matter={'tags':['nonmoral-deliberation','review-candidates']})
    info = hf_api().dataset_info(hf_org()+'/'+name)
    assert not info.private
    write_json(ROOT/'publication.json',dict(url=url,revision=info.sha,private=False))
    print(url,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/data/synth/nonmoral-broader.yaml'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--publish-production',action='store_true')
    parser.add_argument('--phase',choices=['sources','answers','review'])
    parser.add_argument('--batch')
    parser.add_argument('--source-review',type=Path)
    parser.add_argument('--answer-review',type=Path,help='Completed author_review.json gate for deferred --phase review')
    parser.add_argument('--assemble',action='store_true')
    parser.add_argument('--replay-mixture',type=Path)
    args = parser.parse_args()
    if args.publish_production:
        if args.execute or args.publish or args.phase or args.assemble:
            raise ValueError('--publish-production is a separate stable-snapshot operation')
        publish_production(args.config)
        return
    if args.assemble:
        if args.phase or args.execute or args.publish or not args.replay_mixture:
            raise ValueError('--assemble requires --replay-mixture and is a separate offline operation')
        assemble(args.replay_mixture)
        return
    cfg = OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    if args.phase:
        if args.publish:
            raise ValueError('Publish retained artifacts separately after review')
        assert hf_org()=='dougalldeepmind'
        allocations = json.loads((ROOT/'budget_allocations.json').read_text())
        if not 0 < float(cfg['budget_usd']) <= float(allocations['broader_data_cap_usd']):
            raise ValueError('Production cap exceeds the recorded broader-data allocation')
        if allocations['total_allocations_plus_prior_usd'] > 300:
            raise ValueError('Recorded project allocations exceed the authorized ceiling')
        assert cfg['hf_push'] is False and cfg['workers']==4
        assert all(model_cfg(cfg,k)['model']==SONNET for k in cfg['models'])
        production(cfg,args)
        return
    assert cfg['total_scenarios']==len(cfg['cases'])==12 and cfg['workers']==4
    assert cfg['budget_usd']==100 and cfg['dispatch_cap_usd']==3 and cfg['target_accepted']==700
    assert cfg['hf_push'] is False and cfg['batch'] is False
    assert Path(cfg['output_dir']).resolve()==(ROOT/'runs').resolve()
    assert Path(cfg['source']['local_dir']).resolve()==(ROOT/'first12').resolve()
    assert cfg['source']['snapshot']=='inputs.jsonl'
    assert [s['kind'] for s in cfg['stages']]==['load_source_run','llm_tagged','llm_tagged']
    assert all(not (set(s)&{'lint','verify','fallback_model'}) for s in cfg['stages'])
    assert all(model_cfg(cfg,k)['model']==SONNET and model_cfg(cfg,k).get('extra_body')=={'reasoning':{'enabled':False}}
               for k in cfg['models'])
    assert hf_org()=='dougalldeepmind'
    print(json.dumps(dict(stages=[s.name for s in build_stages(cfg)],candidates=12,
                         dispatch_cap_usd=3,whole_dataset_cap_usd=100,paid=args.execute)),flush=True)
    if not args.execute:
        if args.publish:
            publish()
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    marker = ROOT/'dispatch.json'
    assert not marker.exists(), 'First batch already dispatched; inspect saved outputs instead of paying twice'
    prices = verify_live_prices({SONNET})
    source = ROOT/'first12/inputs.jsonl'
    source.parent.mkdir(parents=True,exist_ok=True)
    rows = [dict(scenario_id=f'broader_20260909_{i:03d}',**case) for i,case in enumerate(cfg['cases'],1)]
    source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    lock = ROOT/'generation.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode())
        run_dir = ROOT/'runs'/timestamp()
        run_dir.mkdir(parents=True,exist_ok=False)
        state = dict(status='generating',run_dir=str(run_dir),config=cfg,config_sha256=file_sha256(args.config),
                     source_sha256=file_sha256(source),prices=prices,
                     prior_total_exposure_usd=31.929031131298995,total_ceiling_usd=300,
                     code_sha256={p:file_sha256(p) for p in ['scratch/nonmoral/broader_data.py','scratch/nonmoral/pilot.py',
                                                           'src/data/synth/stage_operators.py','src/data/synth/stage_runtime.py']})
        write_json(marker,state)
        write_json(run_dir/'frozen_config.json',state)
        (ROOT/args.config.name).write_bytes(args.config.read_bytes())
        from tenacity import stop_after_attempt
        client = OpenRouterClient()
        single = client.chat.retry_with(stop=stop_after_attempt(1))
        capped = CappedClient(lambda **kw:single(client,**kw),ROOT/'spend.json',3,{SONNET},allow_reasoning_off=True)
        completed = False
        try:
            run(cfg,resume=str(run_dir),client=capped)
            completed = True
        finally:
            state.update(status='awaiting_review' if completed else 'generation_failed',summary=packet(run_dir))
            if (ROOT/'spend.json').exists():
                entries = json.loads((ROOT/'spend.json').read_text())
                state.update(calls=len(entries),exposure_usd=sum(e['charged_or_reserved_usd'] for e in entries),
                             unsettled_calls=sum(e['status']!='settled' for e in entries))
            write_json(ROOT/'status.json',state)
    finally:
        os.close(fd)
        lock.unlink()
    if args.publish:
        publish()


if __name__=='__main__':
    main()
