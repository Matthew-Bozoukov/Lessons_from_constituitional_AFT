# ABOUTME: Validates literal local recoveries and prepares one derivative batch for the existing broader review pipeline.
# ABOUTME: Preserves source/author ancestry, reuses unchanged model reviews, and never dispatches paid calls.
import argparse
import copy
import json
import math
import re
from pathlib import Path

from omegaconf import OmegaConf

from scratch.nonmoral.broader_data import (
    ROOT, accepted_production, author_review_inputs, conversation_sha256,
    source_review_inputs,
)
from scratch.nonmoral.pilot import SONNET, file_sha256, read_rows


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(path, expected):
    path = Path(path).resolve()
    if not expected or file_sha256(path) != expected:
        raise ValueError(f'Stale ancestry snapshot: {path}')
    return path


def recovery_input(directory):
    """Normalize only the three formats actually emitted by the recovery reviewers."""
    directory = Path(directory).resolve()
    review_path = directory/'recovery_review.json'
    review = load(review_path)
    if review.get('status') != 'complete' or not review.get('reviewer'):
        raise ValueError(f'Recovery review must be complete and named: {directory}')
    if 'records' in review:  # batch07 stakes reviewer
        records = review['records']
        gate = load(directory/'author_review.json')
        snapshots = {gate['snapshot_path']: review['original_snapshot_sha256']}
        replacements_key = 'replacements'
    elif 'inputs' in review:  # batches01_04 baseline reviewer
        records = review['dispositions']
        snapshots = {v['dataset']: v['dataset_sha256'] for v in review['inputs'].values()}
        for v in review['inputs'].values():
            checked(v['review'], v['review_sha256'])
        replacements_key = 'replacements'
    elif 'original_snapshots' in review:  # batches05_06 ablation reviewer
        records = review['dispositions']
        snapshots = review['original_snapshots']
        replacements_key = 'exact_replacements'
    else:
        raise ValueError(f'Unknown recovery review schema: {directory}')
    originals = {}
    for path, sha in snapshots.items():
        path = checked(path, sha)
        if path.name == 'answer_review.json':
            continue  # Ablation schema hashes its source review alongside each JSONL snapshot.
        if path.suffix != '.jsonl':
            raise ValueError(f'Unexpected ancestry record format: {path}')
        for row in read_rows(path):
            sid = row['scenario_id']
            if sid in originals:
                raise ValueError(f'Duplicate original recovery ID: {sid}')
            originals[sid] = (row, path)
    candidates = read_rows(directory/'recovery_candidates.jsonl')
    if len({r['scenario_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate recovery candidates')
    return review, review_path, records, originals, candidates, replacements_key


def replay(original, candidate, disposition, key):
    """Accept only exact, counted replacements in authored reasoning/response."""
    if disposition.get('decision') not in ('accept', 'corrected', 'unchanged'):
        raise ValueError('Candidate is not accepted by recovery reviewer')
    if not disposition.get('reason'):
        raise ValueError('Recovery candidate needs a reason')
    if conversation_sha256(original) != disposition.get('original_conversation_sha256'):
        raise ValueError('Stale original conversation hash')
    if candidate['user'] != original['user']:
        raise ValueError('Changed user prompt')
    result = copy.deepcopy(original)
    replacements = []
    for item in disposition.get(key, []):
        field, old, new = item['field'], item['old'], item['new']
        if field not in ('reasoning', 'response') or not isinstance(old, str) or not old:
            raise ValueError('Only literal nonempty reasoning/response replacements are allowed')
        # Baseline/ablation record str.replace(old,new), which replaces ALL exact matches.
        # The candidate and revised SHA below independently bind that complete result.
        count = item.get('occurrences', result[field].count(old))
        if not isinstance(new, str) or type(count) is not int or count < 1:
            raise ValueError('Invalid literal replacement')
        if result[field].count(old) != count:
            raise ValueError('Nonliteral or ambiguous replacement')
        result[field] = result[field].replace(old, new)
        replacements.append(dict(field=field, old=old, new=new, occurrences=count))
    if (candidate['scenario_id'] != original['scenario_id'] or
            any(result[k] != candidate[k] for k in ('user', 'reasoning', 'response'))):
        raise ValueError('Candidate does not equal literal replacement replay')
    if conversation_sha256(result) != disposition.get('revised_conversation_sha256'):
        raise ValueError('Stale revised conversation hash')
    if any(not result.get(k, '').strip() for k in ('user', 'reasoning', 'response')):
        raise ValueError('Incomplete recovered conversation')
    return result, replacements


def set_review_cap(cfg, review_cap_usd=5.0):
    """Recovery review defaults to a $5 phase cap inside the existing total budget."""
    cap = float(review_cap_usd)
    if not math.isfinite(cap) or cap <= 0 or cap > float(cfg.get('budget_usd', cap)):
        raise ValueError('Review cap must be positive, finite and within the total budget')
    cfg['production'].setdefault('phase_cap_usd', {})['review'] = cap


def prepare(config_path, recovery_dirs, batch, root=ROOT, review_cap_usd=5.0):
    """Read and validate everything first; this function has no writes or API calls."""
    root = Path(root).resolve()
    if not re.fullmatch(r'batch[0-9]{2,3}', batch):
        raise ValueError('Expected batchNN or batchNNN')
    destination = root/'production'/batch
    if destination.exists():
        raise ValueError('Destination exists; never overwrite a production batch')
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = copy.deepcopy(cfg)
    set_review_cap(cfg, review_cap_usd)
    quality = cfg['stages'][-1]
    if quality.get('name') != 'quality_review':
        raise ValueError('Expected existing final quality_review stage')
    effective_models = cfg['production'].get('models', cfg['models'])
    if effective_models[quality['model']]['model'] != SONNET:
        raise ValueError('Recovery must retain the existing Sonnet reviewer')
    quality['when'] = {'field': 'needs_model_review', 'in': [True]}
    cfg['production']['defer_model_review'] = True
    quality_fields = set(quality['save'])
    existing, _ = accepted_production(root)
    accepted_ids = {r['scenario_id'] for r in existing}
    accepted_users = {' '.join(r['user'].split()) for r in existing}
    sources, authors, dispositions, excluded, ancestry, reviews = [], [], {}, [], {}, []
    seen_ids, seen_users = set(), set()
    parent_cache = {}
    for directory in recovery_dirs:
        review, review_path, records, originals, candidates, key = recovery_input(directory)
        reviews.append({'path': str(review_path), 'sha256': file_sha256(review_path)})
        ancestry[str(review_path)] = file_sha256(review_path)
        gate_path=Path(directory).resolve()/'author_review.json'
        if gate_path.exists(): ancestry[str(gate_path)]=file_sha256(gate_path)
        candidate_path = Path(directory).resolve()/'recovery_candidates.jsonl'
        ancestry[str(candidate_path)] = file_sha256(candidate_path)
        for candidate in candidates:
            sid = candidate['scenario_id']
            if sid not in originals or sid not in records:
                raise ValueError(f'Candidate lacks original/recovery disposition: {sid}')
            original, original_path = originals[sid]
            row, edits = replay(original, candidate, records[sid], key)
            match = re.fullmatch(r'broader_(batch[0-9]{2,3})_[0-9]+', sid)
            if not match:
                raise ValueError(f'Unexpected original broader ID: {sid}')
            parent = root/'production'/match[1]
            if not original_path.is_relative_to(parent/'answers'):
                raise ValueError(f'Original author snapshot is outside its declared parent: {sid}')
            if parent not in parent_cache:
                source_status = load(parent/'sources/status.json')
                source_path = Path(source_status['run_dir'])/'dataset.jsonl'
                approved, _ = source_review_inputs(parent/'source_review.json', source_path)
                author_status_path = parent/'answers/status.json'
                # Paused batch07 can have a partial author snapshot and no final status.
                author_status = load(author_status_path) if author_status_path.exists() else load(original_path.parent/'frozen_config.json')
                parent_cache[parent] = ({r['scenario_id']:r for r in approved}, source_status, author_status)
                for p in [source_path, parent/'source_review.json', parent/'sources/status.json',
                          author_status_path if author_status_path.exists() else original_path.parent/'frozen_config.json']:
                    ancestry[str(p.resolve())] = file_sha256(p)
                if (parent/'answer_review.json').exists():
                    p=parent/'answer_review.json'; ancestry[str(p.resolve())]=file_sha256(p)
                for p in [*parent.glob('author_review_*.json'), parent/'sources/config.yaml', parent/'answers/config.yaml']:
                    if p.exists(): ancestry[str(p.resolve())]=file_sha256(p)
            approved, source_status, author_status = parent_cache[parent]
            if sid not in approved or row['user'] != approved[sid]['user']:
                raise ValueError(f'Changed or unapproved original source: {sid}')
            ancestry[str(original_path)] = file_sha256(original_path)
            user_key = ' '.join(row['user'].split())
            if sid in accepted_ids or user_key in accepted_users:
                excluded.append({'scenario_id':sid,'reason':'Already accepted ID or normalized request in production'})
                continue
            if sid in seen_ids or user_key in seen_users:
                raise ValueError(f'Duplicate recovery ID/request: {sid}')
            seen_ids.add(sid); seen_users.add(user_key)
            changed = conversation_sha256(row) != conversation_sha256(original)
            reuse_quality = not changed and original.get('quality_decision') in ('accept','reject')
            if reuse_quality and any(k not in original for k in quality_fields):
                raise ValueError('Incomplete prior model quality review')
            if not reuse_quality:
                for field in quality_fields:
                    row.pop(field, None)
            row['needs_model_review'] = not reuse_quality
            row['recovery_provenance'] = {
                'kind':'local_derivative_not_model_generation','parent_batch':str(parent),
                'original_author_snapshot':str(original_path),'original_author_snapshot_sha256':file_sha256(original_path),
                'original_conversation_sha256':conversation_sha256(original),'revised_conversation_sha256':conversation_sha256(row),
                'original_author_models':author_status.get('config',{}).get('models',{}),
                'original_source_models':source_status.get('config',{}).get('models',{}),
                'original_author_config_sha256':author_status.get('config_sha256'),
                'original_source_config_sha256':source_status.get('config_sha256'),
                'reviewer':review['reviewer'],'recovery_review':str(review_path),
                'recovery_review_sha256':file_sha256(review_path),'replacements':edits,
                'reused_exact_model_review':reuse_quality,
                'prior_model_review':{k:original[k] for k in quality_fields if k in original},
            }
            authors.append(row); sources.append(copy.deepcopy(approved[sid]))
            dispositions[sid]={'decision':'accept','reason':records[sid]['reason'],
                               'conversation_sha256':conversation_sha256(row)}
    return {'destination':destination,'config':cfg,'input_config':str(Path(config_path).resolve()),
            'input_config_sha256':file_sha256(config_path),'sources':sources,'authors':authors,
            'dispositions':dispositions,'excluded':excluded,'ancestry':ancestry,'recovery_reviews':reviews}


def materialize(plan):
    """Write a new batch only after full validation; no GPU/API/publishing operations."""
    target = plan['destination']
    if target.exists() or not plan['authors']:
        raise ValueError('Destination exists or no new candidates')
    # Recheck ancestry immediately before any writes, including every input review/candidate file.
    for path, sha in plan['ancestry'].items(): checked(path,sha)
    checked(plan['input_config'],plan['input_config_sha256'])
    target.mkdir(parents=True,exist_ok=False)
    def write(path, obj):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def rows(path, data):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    config_path=target/'config.yaml'
    OmegaConf.save(OmegaConf.create(plan['config']),config_path)
    config_sha=file_sha256(config_path)
    archive=[]
    for i,(original,sha) in enumerate(sorted(plan['ancestry'].items())):
        dest=target/'audit'/f'{i:03d}_{Path(original).name}'
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(Path(original).read_bytes())
        archive.append({'original_path':original,'sha256':sha,'archived_path':str(dest)})
    for phase,data,status in [('sources',plan['sources'],'awaiting_local_source_review'),
                              ('answers',plan['authors'],'awaiting_local_author_review')]:
        path=target/phase/'dataset.jsonl'; rows(path,data)
        state={'status':status,'run_dir':str(target/phase),'config':plan['config'],
               'config_sha256':config_sha,'dataset_sha256':file_sha256(path),'produced':len(data),
               'origin':'local_derivative_not_model_generation','paid_calls':0}
        write(target/phase/'status.json',state)
        (target/phase/'config.yaml').write_bytes(config_path.read_bytes())
    source_path=target/'sources/dataset.jsonl'
    write(target/'source_review.json',{'reviewer':'prepare_recovery: verified original source approvals',
        'status':'complete','source_sha256':file_sha256(source_path),
        'dispositions':{r['scenario_id']:{'decision':'accept','reason':'Exact original user retained from hash-validated accepted source'} for r in plan['sources']}})
    write(target/'author_review.json',{'reviewer':'prepare_recovery: integrated named local recovery reviews',
        'status':'complete','author_snapshot_sha256':file_sha256(target/'answers/dataset.jsonl'),
        'dispositions':plan['dispositions'],'origin':'local_derivative_not_model_generation'})
    write(target/'audit/lineage.json',{'origin':'local_derivative_not_model_generation',
        'input_config':plan['input_config'],'input_config_sha256':plan['input_config_sha256'],
        'frozen_review_config_sha256':config_sha,'ancestry':archive,'excluded_duplicates':plan['excluded'],
        'recovery_reviews':plan['recovery_reviews'],'summary':summary(plan)})
    source_review_inputs(target/'source_review.json',source_path)
    author_review_inputs(target/'author_review.json',target/'answers/dataset.jsonl')
    return target


def summary(plan):
    return {'destination':str(plan['destination']),'candidates':len(plan['authors']),
            'needs_model_review':sum(r['needs_model_review'] for r in plan['authors']),
            'reused_model_reviews':sum(not r['needs_model_review'] for r in plan['authors']),
            'excluded_previously_accepted':len(plan['excluded']),'paid_calls':0}


def repair_undispatched_review_cap(batch, review_cap_usd=5.0):
    """Fix only an undispatched local derivative; archive every replaced byte first."""
    batch = Path(batch).resolve()
    if any((batch/'review').rglob('*')) or any((batch/phase/'dispatch.json').exists()
                                             for phase in ('sources','answers','review')):
        raise ValueError('Cannot repair a dispatched batch')
    config_path = batch/'config.yaml'
    old_sha = file_sha256(config_path)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if 'review' in cfg['production'].get('phase_cap_usd', {}):
        raise ValueError('Review cap already present; this repair only fills missing configuration')
    set_review_cap(cfg, review_cap_usd)
    lineage_path = batch/'audit/lineage.json'
    lineage = load(lineage_path)
    if lineage['frozen_review_config_sha256'] != old_sha:
        raise ValueError('Stale recovery lineage config hash')
    updates = {lineage_path: lineage}
    configs = [config_path]
    for phase in ('sources', 'answers'):
        path = batch/phase/'status.json'
        state = load(path)
        if (state.get('origin') != 'local_derivative_not_model_generation'
                or state.get('paid_calls') != 0 or state['config_sha256'] != old_sha):
            raise ValueError('Repair requires untouched local derivative phases')
        if state['config'] != OmegaConf.to_container(OmegaConf.load(config_path), resolve=True):
            raise ValueError('Phase configuration differs before repair')
        phase_config = batch/phase/'config.yaml'
        checked(phase_config, old_sha)
        checked(batch/phase/'dataset.jsonl', state['dataset_sha256'])
        updates[path] = state
        configs.append(phase_config)
    archive = batch/'audit/review_cap_fix'
    archive.mkdir(exist_ok=False)
    before = {}
    for path in [*configs, *updates]:
        relative = path.relative_to(batch)
        dest = archive/'before'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        before[str(relative)] = file_sha256(path)
    encoded = OmegaConf.to_yaml(OmegaConf.create(cfg)).encode('utf-8')
    for path in configs:
        path.write_bytes(encoded)
    new_sha = file_sha256(config_path)
    for path, state in updates.items():
        if path == lineage_path:
            state['frozen_review_config_sha256'] = new_sha
            state['review_cap_fix'] = str(archive/'manifest.json')
        else:
            state.update(config=cfg, config_sha256=new_sha)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = dict(reason='Missing review phase cap failed before dispatch/API',
        review_cap_usd=float(review_cap_usd), config=str(config_path),
        original_config_sha256=old_sha, corrected_config_sha256=new_sha,
        pre_repair_hashes=before, paid_calls=0,
        verification='No phase dispatch marker or review artifacts; source/author phases are local derivatives with zero paid calls; datasets unchanged')
    (archive/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--recovery-dir',type=Path,action='append',required=True)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--review-cap-usd',type=float,default=5.0,
                        help='Recovery model-review phase cap inside total budget (default: 5 USD)')
    parser.add_argument('--write',action='store_true',help='Create new batch after validation; never dispatch review')
    args=parser.parse_args()
    plan=prepare(args.config,args.recovery_dir,args.batch,args.root,args.review_cap_usd)
    if args.write: materialize(plan)
    print(json.dumps({**summary(plan),'written':args.write},indent=2))


if __name__=='__main__': main()
