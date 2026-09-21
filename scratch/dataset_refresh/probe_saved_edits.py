# ABOUTME: Tests one fixed editor and reviewer on explicitly selected saved failures.
# ABOUTME: Diagnostic regression only; never counts reused scenarios as a fresh generation smoke.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from omegaconf import OmegaConf
from src.data.synth.ours.stage_runtime import Usage
from src.naming import artifact_name, to_local
from src.utils import timestamp, git_sha
from scratch.dataset_refresh.constitution_smoke import SingleAttemptClient, bounded_llm, answer_gate
from scratch.dataset_refresh.run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    recipe = Path(spec['recipe'])
    cfg = OmegaConf.to_container(OmegaConf.load(recipe), resolve=True)
    if cfg['budget_usd'] > 8 or len(spec['ids']) > 2:
        raise ValueError('Existing authorization: $8 campaign, at most two saved-case probes')
    cfg['pipeline'] = path.stem
    cfg['smoke_contract']['max_physical_calls'] = 2 * len(spec['ids'])
    cfg['smoke_contract']['stop_on_unreachable_yield'] = False
    root = Path('output') / to_local(artifact_name(path.stem + ' smoke')) / timestamp()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(spec['source'])
    all_rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    rows = [next(row for row in all_rows if row['scenario_id'] == rid) for rid in spec['ids']]
    sources = [path, recipe, source, Path(__file__), Path('scratch/dataset_refresh/constitution_smoke.py'),
               Path('scratch/dataset_refresh/run.py'), Path(cfg['constitution']), Path('configs/endpoints/providers.yaml')]
    for file in sources:
        shutil.copy2(file, root / file.name)
    write_json(root / 'run_meta.json', dict(git_sha=git_sha(), config=cfg, diagnostic_only=True,
        fresh_generation=False, hashes={str(file):hashlib.sha256(file.read_bytes()).hexdigest() for file in sources}))
    client = SingleAttemptClient(root, cfg)
    initial = len(client.budget.entries())
    ctx = SimpleNamespace(vars={}, workers=1, client=client, usage=Usage(), run_dir=root, manifest_extra={}, stop=None)
    write_json(root / 'inputs.json', rows)
    print('PROBE_ROOT=' + str(root.resolve()), flush=True)
    try:
        for name in ('edit_responses', 'review_responses'):
            stage = next(s for s in cfg['stages'] if s['name'] == name)
            rows = bounded_llm(stage, cfg).fn(ctx, rows, None)
            write_json(root / (name + '.json'), rows)
            if not rows:
                break
        if rows and all('review' in row for row in rows):
            kept = answer_gate({'name':'admit_answers'}, cfg).fn(ctx, rows, None)
            write_json(root / 'automatic_summary.json', dict(submitted=len(spec['ids']), accepted=len(kept),
                independent_review='pending', diagnostic_only=True))
    except BaseException as exc:
        write_json(root / 'stopped.json', dict(error=str(exc), retry_allowed=False))
        raise
    finally:
        entries = client.budget.entries()
        write_json(root / 'cost_summary.json', dict(campaign_total_usd=sum(e['charged_or_reserved_usd'] for e in entries),
            physical_calls=len(entries)-initial, charged_or_reserved_usd=sum(e['charged_or_reserved_usd'] for e in entries[initial:])))
        print('PROBE_ROOT=' + str(root.resolve()), flush=True)


if __name__ == '__main__':
    main()
