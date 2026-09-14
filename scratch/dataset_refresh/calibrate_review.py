# ABOUTME: Check the revised reviewers against fixed known-negative first-pilot examples.
# ABOUTME: Uses the same shared budget and preserves blind prompts, responses, and pass/fail evidence.
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

from scratch.dataset_refresh.run import BudgetClient, acceptance, load_config, load_checkpoint, render, request_options, save_checkpoint
from src.data.synth.ours.constitution import full_text


CASES = [
    ('da-lowstakes-refresh', 't1_000_v0', 'preflight', 'Professional regulation and patient routing'),
    ('da-lowstakes-refresh', 't2_000_v0', 'preflight', 'Housing reserves and $30000 authority'),
    ('nonmoral-advice', 't2_000_v0', 'preflight', 'Production security vulnerabilities'),
    ('nonmoral-advice', 't6_000_v0', 'final', 'Hidden preference narration'),
    ('nonmoral-advice', 't7_000_v0', 'final', 'Hidden preference narration and unsupported reader arrangements'),
    ('nonmoral-advice', 't8_000_v0', 'final', 'Invented migration contract'),
    ('nonmoral-advice', 't8_001_v0', 'final', 'Draft editing process leakage'),
]


def main(root, configs, output, ceiling):
    root, output = Path(root), Path(output)
    configs = {cfg['pipeline']: cfg for cfg in map(load_config, configs)}
    client = BudgetClient(root / 'budget', ceiling, {cfg['models']['review']['model'] for cfg in configs.values()})
    def check(case):
        arm, candidate, stage, defect = case
        path = output / (arm + '_' + candidate + '_' + stage + '.json')
        if path.exists():
            return load_checkpoint(path)
        cfg = configs[arm]
        row = load_checkpoint(root / arm / 'records' / candidate / 'result.json')['record']
        keys = ('system', 'user') if stage == 'preflight' else ('system', 'user', 'reasoning', 'response')
        conversation = {k: row[k] for k in keys}
        fields = {k: row[k] for k in ('trait_id', 'trait_name', 'trait_text')}
        fields.update(conversation_json=json.dumps(conversation, ensure_ascii=False),
                      record_json=json.dumps(conversation, ensure_ascii=False), metadata_json='{}',
                      constitution=full_text(cfg['constitution']), eligibility_json='{}', **conversation)
        prompts = cfg['preflight']['prompts'] if stage == 'preflight' else {
            role: cfg['prompts']['review_' + role] for role in ('system', 'user')}
        messages = [{'role': role, 'content': render(prompts[role], fields)} for role in ('system', 'user')]
        client.local.arm, client.local.candidate_id, client.local.stage = arm, 'calibration_' + candidate, stage
        client.local.run_root = str(output.resolve())
        from src.data.synth.ours.stage_runtime import _parse_json
        review = _parse_json(client.chat(messages=messages, **request_options(cfg['models']['review'])).content)
        accepted = acceptance(review, cfg['preflight'] if stage == 'preflight' else cfg)
        result = {'arm': arm, 'candidate_id': candidate, 'stage': stage, 'known_defect': defect,
                  'review': review, 'correctly_rejected': not accepted}
        save_checkpoint(path, result)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, CASES))
    summary = {'cases': results, 'all_known_negatives_rejected': all(r['correctly_rejected'] for r in results)}
    save_checkpoint(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ceiling', type=float, default=20)
    a = p.parse_args()
    main(a.root, a.configs, a.output, a.ceiling)
