# ABOUTME: Offline source selection, stakes-frame preparation and matched-pair export for the shared synth engine.
# ABOUTME: Never makes model calls, rents GPUs or publishes; writes exact provenance and refuses silent source repairs.
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from scratch.nonmoral.stakes.fixtures import FIXTURES
from src.data.synth.pipeline import build_stages


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def unique(rows):
    ids = [r['scenario_id'] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicate scenario IDs')


def approved(rows, source_path, review_path):
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    unique(rows)
    if review.get('source_sha256') != digest(source_path) or not review.get('reviewer'):
        raise ValueError('Named review must identify exact source bytes')
    decisions = review['dispositions']
    if set(decisions) != {r['scenario_id'] for r in rows}:
        raise ValueError('Every source requires a disposition, including holds/exclusions')
    for d in decisions.values():
        if d.get('decision') not in ('accept', 'reject', 'hold') or not d.get('reason'):
            raise ValueError('Invalid disposition or missing reason')
    return [r for r in rows if decisions[r['scenario_id']]['decision'] == 'accept']


def flatten(row):
    messages = row.get('messages', [])
    users = [m['content'] for m in messages if m['role'] == 'user']
    systems = [m['content'] for m in messages if m['role'] == 'system']
    if len(users) != 1 or len(systems) > 1:
        raise ValueError('Only exact single-turn sources are supported')
    return dict(scenario_id=row['metadata']['scenario_id'], core_user=users[0],
                original_system=systems[0] if systems else '',
                domain=row.get('metadata', {}).get('domain', 'unknown'),
                source_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest())


def source_packet(path, cfg, out):
    if digest(path) != cfg['source_parent']['sha256']:
        raise ValueError('Source does not match the pinned historical corpus; update the declared parent explicitly')
    rows = [flatten(r) for r in read_rows(path)]
    unique(rows)
    # Stable pseudo-random order independent of ODCV results, not cherry-picked membership.
    rows.sort(key=lambda r: hashlib.sha256(f"{cfg['seed']}:{r['scenario_id']}".encode()).hexdigest())
    write_rows(out/'source_candidates.jsonl', rows)
    write_json(out/'source_review_template.json', dict(source_sha256=digest(out/'source_candidates.jsonl'),
        reviewer='', dispositions={r['scenario_id']:dict(decision='hold', reason='Not yet screened for complete source and compatible personal stakes') for r in rows}))
    write_json(out/'source_manifest.json', dict(parent=cfg['source_parent'], rows=len(rows),
        source_path=str(Path(path).resolve()), source_sha256=digest(path), copied_original_answers=False,
        selection='All source rows, stable SHA256(seed:scenario_id) ordering; no ODCV features'))
    return rows


def arm_rows(frames):
    unique(frames)
    rows = []
    for r in frames:
        if r.get('eligible') != 'yes':
            raise ValueError('Ineligible frame entered accepted input')
        if not r.get('core_user', '').strip() or r['low_context'].strip() == r['high_context'].strip():
            raise ValueError('Empty core or identical low/high frame')
        for arm in ('low', 'high'):
            frame = r[arm+'_context']
            if not isinstance(frame, str) or not frame.strip():
                raise ValueError('Empty stakes frame')
            rows.append(dict(scenario_id=r['scenario_id']+'__'+arm, pair_id=r['scenario_id'],
                arm=arm, original_system=r['original_system'], core_user=r['core_user'],
                stakes_context=frame, user=r['core_user']+'\n\n'+frame))
    return rows


def join_answers(rows):
    unique(rows)
    pairs = {}
    for r in rows:
        if r['arm'] not in ('low', 'high') or r['scenario_id'] != r['pair_id']+'__'+r['arm']:
            raise ValueError('Invalid arm identity')
        if r['user'] != r['core_user']+'\n\n'+r['stakes_context']:
            raise ValueError('Source was altered outside the declared appended context')
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('reasoning', 'response')):
            raise ValueError('Incomplete reasoning or response')
        pairs.setdefault(r['pair_id'], {})[r['arm']] = r
    result = []
    for pair_id, arms in pairs.items():
        if set(arms) != {'low', 'high'}:
            raise ValueError('Missing paired answer; retain attrition and exclude the whole pair')
        low, high = arms['low'], arms['high']
        if low['core_user'] != high['core_user'] or low['original_system'] != high['original_system']:
            raise ValueError('Task core/system mismatch across arms')
        result.append(dict(scenario_id=pair_id, original_system=low['original_system'],
            core_user=low['core_user'], **{arm+'_'+key: arms[arm][key]
            for arm in ('low', 'high') for key in ('user', 'reasoning', 'response')}))
    return result


def phase_config(cfg, phase, rows, out):
    if not rows:
        raise ValueError('No accepted inputs; do not launch an empty phase')
    write_rows(out/'inputs.jsonl', rows)
    effective = {k: v for k, v in cfg.items() if k not in ('phases', 'source_parent', 'target_pairs')}
    effective.update(source=dict(local_dir=str(out.resolve()), snapshot='inputs.jsonl'),
                     total_scenarios=len(rows), output_dir=str((out/'runs').resolve()),
                     stages=[dict(name='load_stakes_input', kind='load_source_run'), *cfg['phases'][phase]])
    build_stages(effective)
    OmegaConf.save(OmegaConf.create(effective), out/'prepared_config.yaml')
    write_json(out/'preparation.json', dict(phase=phase, rows=len(rows), input_sha256=digest(out/'inputs.jsonl'),
        paid_dispatch_authorized=False, budget_usd=0,
        note='Prepared shared-pipeline configuration only. Requires centrally allocated budget and capped launcher.'))


def fixture_packet(out):
    rows = [dict(r, original_system='', eligible='yes') for r in FIXTURES]
    write_rows(out/'design_frames.jsonl', rows)
    complete = arm_rows(rows)
    write_rows(out/'design_requests.jsonl', complete)
    lines = ['# Nonmoral stakes: offline design fixtures', '',
             'Author-written examples for reviewing the intervention. Not Sonnet output, not SFT data. '
             'Only the appended personal-loss paragraph changes. The expensive-artifact frame repeats '
             'deliberately to make the intervention legible; production must not use it indiscriminately.', '']
    for r in rows:
        lines.extend([f"## {r['scenario_id']} — {r['domain']}", '', '**Shared complete task**', '',
                      r['core_user'], '', '**Low stakes**', '', r['low_context'], '',
                      '**High stakes**', '', r['high_context'], ''])
    (out/'design_requests.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['fixtures', 'sources', 'frames', 'answers', 'review'])
    p.add_argument('--config', type=Path, default=Path('configs/data/synth/nonmoral-stakes.yaml'))
    p.add_argument('--input', type=Path)
    p.add_argument('--review', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.out.exists():
        raise ValueError('Use a new timestamped output directory; never overwrite prior preparation')
    args.out.mkdir(parents=True)
    if args.phase == 'fixtures':
        fixture_packet(args.out)
    elif args.phase == 'sources':
        if not args.input:
            p.error('--input is required')
        source_packet(args.input, cfg, args.out)
    else:
        if not args.input or not args.review:
            p.error('--input and hash-linked --review are required')
        rows = approved(read_rows(args.input), args.input, args.review)
        if args.phase == 'answers':
            rows = arm_rows(rows)
        elif args.phase == 'review':
            rows = join_answers(rows)
        phase_config(cfg, args.phase, rows, args.out)
    print(json.dumps(dict(phase=args.phase, output=str(args.out.resolve()), paid_calls=0)))


if __name__ == '__main__':
    main()
