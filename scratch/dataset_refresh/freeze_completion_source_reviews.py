# ABOUTME: Freeze exact inventory-source and saved-answer full-read judgments for repair planning.
# ABOUTME: No API calls, adoption or origin changes; refuse overwrites and bind every reviewed byte.
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt
from scratch.dataset_refresh import offline_acceptance as oa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    out = Path(cfg['output'])
    if out.exists():
        raise FileExistsError(out)
    inventory = rt.read_rows(Path(cfg['inventory']))
    annotations = oa.read(cfg['annotations'])
    assert set(cfg['candidates']) == set(annotations)
    rows, inputs = [], []
    for cid in cfg['candidates']:
        item = next(x for x in inventory if x['candidate_id'] == cid)
        assert item['classification'].startswith('answer_defect')
        folder = Path(cfg['origin_parent']) / item['root'] / item['arm'] / 'records' / cid
        for name, sha in item['source_files_sha256'].items():
            assert oa.sha(folder / name) == sha, (cid, name)
        stage_path = folder / (item['selected_input_stage'] + '.json')
        assert oa.sha(stage_path) == item['selected_input_sha256']
        scenario, stage = rt.load_checkpoint(folder / 'scenario.json'), rt.load_checkpoint(stage_path)
        conv = {**{k: scenario[k] for k in ('system', 'user')}, **{k: stage[k] for k in ('reasoning', 'response')}}
        ann = annotations[cid]
        for quote in ann['evidence']:
            assert quote in conv['reasoning'] + '\n' + conv['response'], (cid, quote)
        base = {k: item[k] for k in ('root', 'arm', 'candidate_id', 'trait_id', 'selected_input_stage', 'selected_input_sha256', 'source_files_sha256')}
        base.update(result_sha256=item['source_files_sha256']['result.json'], conversation_sha256=rt.digest(conv))
        inputs.append({**base, 'conversation': conv})
        rows.append({**base, **ann, 'source_usable': ann['source_eligible'], 'full_system_user_reasoning_final_read': True,
                     'review_scope': 'full_conversation', 'source_concern': 'Benign human craft/documentation advice; source self-contained for structural advice. No whole-pool duplicate clearance.',
                     'instruction': ann['repair_target'], 'full_new09_compatibility': 'Full new09 compatibility and current qualified craft reviewed. Source supports nonmoral practical advice; cited answer defects require a new separately reviewed answer.',
                     'may_enter_selection_without_parent_adjudication': False})
    out.mkdir(parents=True)
    input_path = out / 'frozen_inputs.jsonl'
    input_path.write_text(''.join(__import__('json').dumps(x, ensure_ascii=False) + '\n' for x in inputs), encoding='utf-8')
    report = {'aboutme': ['Fresh full-source and exact saved-answer review of remaining frozen inventory candidates.', 'No models, origin mutations or adoption; specific correction targets and counterreadings.'],
              'created_at': datetime.now(timezone.utc).isoformat(), 'frozen_inputs_sha256': oa.sha(input_path), 'coverage': len(rows),
              'counts': {'source_eligible': sum(x['source_eligible'] for x in rows), **dict(Counter(x['disposition'] for x in rows))},
              'reviewer_provenance': {'kind': 'independent_codex_agent', 'task': '/root/audit_nonmoral', 'human_review': False},
              'inventory_sha256': oa.sha(cfg['inventory']), 'annotations_sha256': oa.sha(cfg['annotations']), 'config_sha256': oa.sha(args.config), 'helper_sha256': oa.sha(__file__),
              'review_contract_sha256': oa.sha(cfg['review_contract']), 'qualified_config_sha256': oa.sha(cfg['qualified_config']),
              'limitations': 'Purposive first-ID remaining answer-defect candidates within the requested traits, selected before fresh full reads. Prior historical allegations were known. All recent completion reviews and all prepared/attempted campaign inputs excluded. Not an error-rate sample or whole-pool duplicate audit.', 'rows': rows}
    rt.save_checkpoint(out / 'independent_review.json', report)
    (out / 'independent_review.md').write_text('<!-- ABOUTME: Summarize fresh source and saved-answer repair findings. -->\n<!-- ABOUTME: These are source eligibility judgments, not training-row acceptance. -->\n\n' + '\n\n'.join(f"- {x['candidate_id']}: {x['disposition']}. {x['reason']} Counterreading: {x['counterreading_and_caveats']}" for x in rows) + '\n', encoding='utf-8')
    print(report['counts'], oa.sha(out / 'independent_review.json'))


if __name__ == '__main__':
    main()
