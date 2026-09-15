# ABOUTME: Freeze independently authored campaign full-read reviews and exact Sonnet acceptance dossiers.
# ABOUTME: Never call models or adopt rows; preserve failed and held outcomes alongside sound-answer reviews.
import argparse
from pathlib import Path
from collections import Counter
from omegaconf import OmegaConf
from scratch.dataset_refresh import offline_acceptance as oa
from scratch.dataset_refresh import run as rt
from scratch.dataset_refresh.saved_input_pilot import ORIGINAL_BUDGET_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    assert oa.sha(oa.__file__) == cfg['implementation_sha256']
    annotations = oa.read(cfg['annotations'])
    plans = []
    for batch in cfg['batches']:
        source = Path(cfg['batch_root']) / batch
        out = Path(cfg['output_root']) / batch
        if out.exists():
            raise FileExistsError(out)
        for index in cfg['indices']:
            stem = f'{index:02d}'
            ip, rp = source / (stem + '.input.json'), source / (stem + '.result.json')
            inp, result = rt.load_checkpoint(ip), rt.load_checkpoint(rp)
            ann = annotations.get(batch + '/' + stem)
            if 'conversation' in result:
                assert ann is not None, (batch, stem)
                conv = result['conversation']
                for quote in ann['source_quotes']:
                    assert quote in conv['system'] + '\n' + conv['user'], (stem, quote)
                for quote in ann['answer_quotes']:
                    assert quote in conv['reasoning'] + '\n' + conv['response'], (stem, quote)
            else:
                assert result['status'] == 'failed' and ann is None
            plans.append((batch, stem, out, ip, rp, inp, result, ann))
    summaries = {}
    for batch, stem, out, ip, rp, inp, result, ann in plans:
        out.mkdir(parents=True, exist_ok=True)
        cid = inp['candidate_id']
        base = {'candidate_id': cid, 'case_key': inp['case_key'], 'input_sha256': oa.sha(ip),
                'result_sha256': oa.sha(rp), 'source_result_sha256': inp['source_ref']['result_sha256'],
                'physical_receipt': result.get('physical_receipt'),
                'reviewer_provenance': {'kind': 'independent_codex_agent', 'task': '/root/audit_nonmoral', 'human_review': False}}
        if ann is None:
            review = {**base, 'decision': 'unavailable', 'accepted': False, 'status': result['status'],
                      'reason': 'No complete returned conversation exists; terminal provider length failure is not a content rejection or acceptance.',
                      'error': result.get('error'), 'full_read': False}
        else:
            spec = {'source_ref': inp['source_ref'], 'author_kind': 'single_saved_revision',
                    'review_config_path': str(Path(cfg['review_config']).resolve()), 'review_config_sha256': oa.sha(cfg['review_config']),
                    'review_contract_path': str(Path(cfg['review_contract']).resolve()), 'review_contract_sha256': oa.sha(cfg['review_contract']),
                    'input_path': str(ip.resolve()), 'input_sha256': oa.sha(ip),
                    'result_path': str(rp.resolve()), 'result_sha256': oa.sha(rp),
                    'raw_path': str(rp.with_name(stem + '.raw.json').resolve()),
                    'physical_receipt': result['physical_receipt'], 'ledger_path': str((ORIGINAL_BUDGET_ROOT / 'spend.json').resolve())}
            spec_path = out / (stem + '.spec.json')
            rt.write_json(spec_path, spec)
            dp = out / cid / 'dossier.json'
            oa.prepare(spec_path, dp.parent)
            d, frozen = oa.validate_dossier(dp)
            conv = d['conversation']
            review = {**base, **ann, 'accepted': ann['decision'] == 'accept', 'full_read': True,
                      'gates': {gate: gate not in ann.get('failure_gates', []) for gate in oa.GATES},
                      'issues': ann.get('issues', []), 'conversation_sha256': d['conversation_sha256'],
                      'review_contract_sha256': d['review_contract_sha256'], 'constitution_sha256': d['constitution_sha256'],
                      'full_working_preference_sha256': d['full_working_preference_sha256'],
                      'request_sha256': d['physical_receipt']['request_sha256'],
                      **{('source_' + k if k in ('system', 'user') else k) + '_sha256': rt.digest(conv[k].encode()) for k in oa.FIELDS},
                      'constitution_compatibility': 'Reviewed full new09 as compatibility only. The substantive advice is benign and agency-respecting; no moral target is forced into the craft reasoning. Specific grounding/presentation issues, if any, remain in gates and issues.',
                      'nonmoral_eligibility': 'Actual decision is benign craft/documentation organization. No medical, veterinary, emergency, legal, financial/trading, employment, political-persuasion, security or safety-critical decision is requested. The workshop case concerns layout and retains technical content only as already taught.',
                      'scope': 'Independent Codex-agent full actual system/user/reasoning/response read, with full current qualified craft and full new09. Earlier source/repair targets were known; no new automated verdict was used. Content decision only; root native/duplicates/adoption remain separate.',
                      'dossier_path': str(dp.resolve()), 'dossier_sha256': oa.sha(dp)}
            if review['accepted']:
                oa.validate_review(d, frozen, review)
        review_path = out / (stem + '.independent_review.json')
        rt.save_checkpoint(review_path, review)
        summaries.setdefault(batch, []).append({'candidate_id': cid, 'decision': review['decision'],
            'review_path': str(review_path.resolve()), 'review_sha256': oa.sha(review_path),
            'source_result_sha256': base['source_result_sha256'], 'result_sha256': base['result_sha256'],
            'dossier_path': review.get('dossier_path'), 'dossier_sha256': review.get('dossier_sha256')})
    for batch, cases in summaries.items():
        out = Path(cfg['output_root']) / batch
        summary = {'status': 'independent_reviews_prepared_not_adopted', 'cases': cases,
                   'counts': dict(Counter(case['decision'] for case in cases)),
                   'reviewer_provenance': {'kind': 'independent_codex_agent', 'task': '/root/audit_nonmoral', 'human_review': False},
                   'annotations_sha256': oa.sha(cfg['annotations']), 'helper_sha256': oa.sha(__file__),
                   'implementation_sha256': cfg['implementation_sha256'], 'config_sha256': oa.sha(args.config),
                   'scope': 'No API calls, origin mutations, adoption or native/duplicate certification. Every complete answer fully read; incomplete result separately accounted.'}
        rt.save_checkpoint(out / 'summary.json', summary)
        (out / 'summary.md').write_text('<!-- ABOUTME: Summarize independently reviewed campaign answers. -->\n<!-- ABOUTME: Prepared dossiers are not adopted training rows. -->\n\n' + '\n'.join(f"- {c['candidate_id']}: {c['decision']} ({c['review_sha256']})" for c in cases) + '\n', encoding='utf-8')
        print(batch, summary['counts'], oa.sha(out / 'summary.json'))


if __name__ == '__main__':
    main()
