# ABOUTME: Prepare exact716+9284 local mixture configs from explicit published synthetic commit pins.
# ABOUTME: Write complete publication cards only after the shared replay and tokenizer audit passes.
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

from omegaconf import OmegaConf

from src.data.mixture.build_mixture import blend
from src.data.mixture.reasoning_backfill import describe
from src.infra.huggingface import card_markdown, training_data_tags
from src.naming import mix_name, synth_name
from src.utils import origin_url

ROOT = Path(__file__).resolve().parents[2]
STYLES = ('da-lowstakes-refresh', 'nonmoral-advice')
BASE_REPO = 'dougalldeepmind/2026-09-08-nosynth-mix'
BASE_REVISION = '7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd'
CONSTITUTION = 'constitutions/claude_distilled_09_principles/constitution.md'
BASE_COUNTS = {'no_robots': 2779, 'tulu3_if': 1471, 'numinamath_cot': 1063,
               'self_oss_instruct': 1064, 'smol_constraints': 1055,
               'apigen_function_calling': 1054, 'smol_summarize': 984,
               'lima': 314, 'longalign': 216}
REPLAY_COUNTS = {'no_robots': 2580, 'tulu3_if': 1366, 'numinamath_cot': 987,
                 'self_oss_instruct': 988, 'smol_constraints': 979,
                 'apigen_function_calling': 978, 'smol_summarize': 914,
                 'lima': 291, 'longalign': 201}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def check_pin(style, repo, revision):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Supply an explicit org/repo, never a URL or placeholder')
    if not re.fullmatch(r'[0-9a-f]{40}', revision) or len(set(revision)) == 1:
        raise ValueError('Supply the real full 40-character synthetic HF commit SHA')
    name = repo.split('/')[1]
    try:
        date = datetime.strptime(name[:10], '%Y-%m-%d').date().isoformat()
    except ValueError as exc:
        raise ValueError('Synthetic repo must have its actual production date') from exc
    if name != synth_name(style, date=date):
        raise ValueError(f'Synthetic repo does not identify expected style {style}')


def prepare(output, pins):
    output = Path(output).resolve()
    if set(pins) != set(STYLES):
        raise ValueError('Both refreshed corpus styles must be supplied')
    for style, (repo, revision) in pins.items():
        check_pin(style, repo, revision)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty preparation directory; never overwrite pins')
    # Freeze only proportions; never inherit the base config's live backfill generator.
    source = ROOT / 'configs/data/mixture/nosynth.yaml'
    original = OmegaConf.to_container(OmegaConf.load(source), resolve=False)
    if {k: v['examples'] for k, v in original['sources'].items()} != BASE_COUNTS:
        raise ValueError('Canonical replay proportions changed; adjudicate before preparing')
    proportions = {'sources': {k: {'examples': n, 'reasoning': 'none'} for k, n in BASE_COUNTS.items()}}
    output.mkdir(parents=True, exist_ok=True)
    base_path = output / 'base-proportions.yaml'
    OmegaConf.save(OmegaConf.create(proportions), base_path)
    manifest = {'base_config_source': str(source), 'base_config_sha256': sha(source),
                'base_snapshot_sha256': sha(base_path), 'configs': {}, 'replay_counts': REPLAY_COUNTS}
    commands = []
    for style in STYLES:
        repo, revision = pins[style]
        cfg = {'seed': 0, 'tokenizer': 'Qwen/Qwen3.6-27B', 'max_seq_len': 8192,
               'base': str(base_path), 'base_mixture': {'repo': BASE_REPO, 'file': 'mixture.jsonl', 'revision': BASE_REVISION},
               'synthetic_examples': 716, 'synthetic_pct': 7, 'total_examples': 10000,
               'sources': {style: {'dataset': repo, 'revision': revision, 'examples': 1,
                                   'reasoning': 'native', 'balance_by': 'trait_id', 'supervise': 'all'}},
               'output_dir': str(output / 'builds' / style)}
        quotas = blend(proportions['sources'], cfg['sources'], 7, 10000, synthetic_examples=716)
        if {k: v['examples'] for k, v in quotas.items() if k != style} != REPLAY_COUNTS:
            raise ValueError('Builder replay quotas differ from the frozen contract')
        config_path = output / f'{style}.yaml'
        OmegaConf.save(OmegaConf.create(cfg), config_path)
        manifest['configs'][style] = {'path': str(config_path), 'sha256': sha(config_path), 'synthetic': cfg['sources'][style]}
        commands.append(f'uv run mix --config "{config_path}"')
    write_json(output / 'preparation.json', manifest)
    (output / 'BUILD_COMMANDS.txt').write_text('\n'.join(commands) + '\n', encoding='utf-8')
    return manifest


def synthetic_provenance(path, style):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('pipeline') != style or not value.get('models') or not value.get('constitution_sha256'):
        raise ValueError('Require frozen synthetic model and constitution provenance')
    if value.get('constitution') != CONSTITUTION:
        raise ValueError('Unexpected constitution in frozen synthetic config')
    if style == 'nonmoral-advice' and not all(value.get(k) for k in ('craft_spec', 'craft_spec_sha256')):
        raise ValueError('Nonmoral source must preserve its craft specification and hash')
    if value.get('composite') is True:
        origins = value.get('origins', [])
        if not origins or len({p['phase_id'] for p in origins}) != len(origins):
            raise ValueError('Composite provenance needs unique explicit origin configurations')
        from scratch.dataset_refresh.run import digest
        if value['models'] != {p['phase_id']: p['config']['models'] for p in origins}:
            raise ValueError('Composite model summary differs from exact origin recipes')
        for phase in origins:
            cfg = phase['config']
            if digest(cfg) != phase['config_sha256'] or cfg.get('pipeline') != style:
                raise ValueError('Composite origin config hash/pipeline mismatch')
            for key in ('constitution', 'constitution_sha256', 'craft_spec', 'craft_spec_sha256', 'original_craft_spec_sha256'):
                if cfg.get(key) != value.get(key):
                    raise ValueError('Composite origins disagree on common constitution/craft provenance')
        if sum(p['selected_rows'] for p in origins) != 716 or any(type(p['selected_rows']) is not int or p['selected_rows'] <= 0 for p in origins):
            raise ValueError('Composite provenance must account for exactly716 selected rows')
    return value


def prepare_card(config_path, mixture_dir, audit_path, synth_config_path):
    config_path, mixture_dir = Path(config_path), Path(mixture_dir)
    style = config_path.stem
    if style not in STYLES:
        raise ValueError('Unknown refresh style')
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    planned = json.loads((config_path.parent / 'preparation.json').read_text(encoding='utf-8'))
    if sha(config_path) != planned['configs'][style]['sha256'] or sha(cfg['base']) != planned['base_snapshot_sha256']:
        raise ValueError('Prepared config/proportions changed')
    meta = json.loads((mixture_dir / 'run_meta.json').read_text(encoding='utf-8'))
    if meta.get('smoke') is not False or meta['config'] != cfg:
        raise ValueError('Require the complete build from this exact config')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    if (audit.get('status') != 'passed' or audit.get('replay_payloads_and_positions_equal') is not True
            or audit.get('synthetic_payloads_equal_pinned_releases') is not True
            or audit.get('max_train_tokens') != 8192 or len(audit.get('mixtures', [])) != 2
            or audit.get('base', {}).get('revision') != BASE_REVISION
            or audit.get('base', {}).get('repo') != BASE_REPO):
        raise ValueError('Require the passing two-arm 8192-token pinned-base audit')
    for item in audit['mixtures']:
        if (item.get('rows'), item.get('synthetic_rows'), item.get('replay_rows')) != (10000, 716, 9284) or sha(item['path']) != item['sha256']:
            raise ValueError('Audit inputs changed or counts are not exact')
    own = [x for x in audit['mixtures'] if Path(x['path']).resolve() == (mixture_dir / 'mixture.jsonl').resolve()]
    if len(own) != 1:
        raise ValueError('This mixture was not audited')
    stats = json.loads((mixture_dir / 'mixture_stats.json').read_text(encoding='utf-8'))
    if {k: v['examples'] for k, v in stats['by_source'].items()} != {**REPLAY_COUNTS, style: 716}:
        raise ValueError('Written mixture source counts differ')
    traces = stats.get('reasoning_traces')
    if not traces or traces.get('inherited_from') != {'repo': BASE_REPO, 'revision': BASE_REVISION}:
        raise ValueError('Published replay reasoning inheritance is missing or different')
    synth_cfg = synthetic_provenance(synth_config_path, style)
    source_audit = own[0].get('synthetic_source') or {}
    synth = cfg['sources'][style]
    if (source_audit.get('repo') != synth['dataset'] or source_audit.get('revision') != synth['revision']
            or source_audit.get('style') != style or source_audit.get('payloads_equal') is not True
            or source_audit.get('rows') != 716 or source_audit.get('file') != 'dataset.jsonl'):
        raise ValueError('Synthetic payload audit does not bind this exact published pin')
    if synth_cfg.get('composite') and synth_cfg.get('dataset_sha256') != source_audit.get('sha256'):
        raise ValueError('Published synthetic bytes differ from composite generation provenance')
    generated = datetime.fromisoformat(meta['timestamp_utc']).date().isoformat()
    name = mix_name(style, 7, date=generated)
    synth = cfg['sources'][style]
    gen = {'seed': 0, 'synthetic_examples': 716, 'replay_examples': 9284, 'total_examples': 10000,
           'exact_synthetic_row_pct': 7.16, 'name_rounded_pct': 7, 'replay_counts': REPLAY_COUNTS,
           'tokenizer': cfg['tokenizer'], 'max_seq_len': 8192, 'base_mixture': cfg['base_mixture'],
           'synthetic_source': synth, 'reasoning_traces': traces,
           'frozen_synth_config_sha256': sha(synth_config_path), 'validation_sha256': sha(audit_path),
           'mixture_sha256': own[0]['sha256'], 'constitution_sha256': synth_cfg['constitution_sha256']}
    gen['synthetic_payload_validation'] = source_audit
    if style == 'nonmoral-advice':
        gen.update(craft_spec=synth_cfg['craft_spec'], craft_spec_sha256=synth_cfg['craft_spec_sha256'])
    if synth_cfg.get('composite'):
        gen['composite_origins'] = [{k: p[k] for k in ('phase_id', 'config_sha256', 'selected_rows')} for p in synth_cfg['origins']]
    role = ('generation and review target' if style == 'da-lowstakes-refresh' else
            'compatibility review target; generation uses the frozen nonmoral craft specification')
    fields = {'title': name, 'experiment': f'{style}: 716 human-advice examples + 9,284 identical shared replay rows; exactly 7.16% synthetic rows, rounded 7 in the repository name.',
              'date_generated': generated, 'constitution': f'{CONSTITUTION} ({role}); SHA256 {synth_cfg["constitution_sha256"]}',
              'source_repo': f'{origin_url()} @ {meta["git_sha"]}',
              'models': ('Synthetic generation/review settings by immutable origin phase: ' if synth_cfg.get('composite') else 'Synthetic generation/review settings from frozen source config: ') + json.dumps(synth_cfg['models']) + '; ' + describe(traces) + f'; replay provenance {BASE_REPO}@{BASE_REVISION}. API model IDs are recorded, not immutable provider-weight revisions.',
              'generation_config': json.dumps(gen),
              'schema': 'mixture.jsonl default train config: model-agnostic messages with role/content and optional reasoning_content/tool_calls; optional top-level tools/source/supervise. Synthetic rows have system/user/assistant, native reasoning_content, supervise=all. Replay payloads and inherited Qwen reasoning traces preserved verbatim; no fresh generation/backfill during mixing.',
              'provenance': f'{meta["command"]}; synthetic {synth["dataset"]}@{synth["revision"]} default dataset.jsonl; replay {BASE_REPO}@{BASE_REVISION}/mixture.jsonl. Frozen source config and passing audit copied beside this card.',
              'comparability': 'Both arms use seed 0, exact largest-remainder replay quotas and identical replay positions. Synthetic cases are unpaired regenerations, not guaranteed matched pairs with difficult advice. Row share is not token share or loss-weight share; see mixture_validation.json for actual masks/token counts. No training or evaluation is implied.'}
    front = {'configs': [{'config_name': 'default', 'data_files': 'mixture.jsonl', 'default': True}],
             'tags': training_data_tags('mixture', style, CONSTITUTION, extra=['stage:final'])}
    card = card_markdown(fields, front)
    write_json(mixture_dir / 'card_fields.json', fields)
    write_json(mixture_dir / 'card_front_matter.json', front)
    write_json(mixture_dir / 'publication_plan.json', {'name': name, 'built_date': generated, 'validated_mixture_sha256': own[0]['sha256']})
    (mixture_dir / 'README.md').write_text(card, encoding='utf-8')
    for source, filename in [(config_path, 'mixture_config.yaml'), (audit_path, 'mixture_validation.json'), (synth_config_path, 'frozen_synthetic_config.json')]:
        if Path(source).resolve() != (mixture_dir / filename).resolve():
            shutil.copyfile(source, mixture_dir / filename)
    return name


def main():
    parser = argparse.ArgumentParser(description='Offline preparation only: never builds, publishes, generates or trains.')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', required=True)
    for arm in ('low', 'nonmoral'):
        prep.add_argument(f'--{arm}-repo', required=True)
        prep.add_argument(f'--{arm}-revision', required=True)
    card = sub.add_parser('card')
    for flag in ('config', 'mixture-dir', 'audit', 'synth-config'):
        card.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output, {STYLES[0]: (args.low_repo, args.low_revision), STYLES[1]: (args.nonmoral_repo, args.nonmoral_revision)})
        print(json.dumps(result, indent=2))
    else:
        print(prepare_card(args.config, args.mixture_dir, args.audit, args.synth_config))


if __name__ == '__main__':
    main()
