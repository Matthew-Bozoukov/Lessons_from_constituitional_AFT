# ABOUTME: Summarizes an independently read smoke and publishes its complete diagnostic record.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.finish_constitution_smoke --root <run-dir> --review <yaml> [--publish]
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from src.infra.huggingface import hf_api, hf_repo_id, push_run_dir, training_data_tags
from src.naming import artifact_name
from src.utils import git_sha, origin_url
from scratch.dataset_refresh.run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root/'generation/manifest.json').read_text(encoding='utf-8'))
    cfg = manifest['config']
    # A failed judge is a diagnostic outcome, never a fabricated pass or paid rerun.
    authored = [json.loads(s) for s in (root/'generation/stage_5_revise_responses.jsonl').read_text(encoding='utf-8').splitlines()]
    judged = [json.loads(s) for s in (root/'generation/stage_6_review_responses.partial.jsonl').read_text(encoding='utf-8').splitlines()]
    reviews = {r['scenario_id']: r['review'] for r in judged}
    rows = [{'messages':[{'role':'system','content':r['system']}, {'role':'user','content':r['user']},
                        {'role':'assistant','content':r['response'],'reasoning_content':r['reasoning']}],
             'metadata':{**{k:r[k] for k in ('scenario_id','trait_id','setting','request_form','reasoning_demand')},
                         'review': reviews.get(r['scenario_id'], {'verdict':'technical_failure','findings':[],
                             'reason':'No complete judge output; see raw call and aborted pipeline manifest'})}} for r in authored]
    manual = OmegaConf.to_container(OmegaConf.load(args.review), resolve=True)
    by_id = {r['id']: r for r in manual['rows']}
    assert len(rows) == 18 and set(by_id) == {r['metadata']['scenario_id'] for r in rows}
    assert len(by_id) == len(manual['rows'])
    codes, domains, mechanisms, accepted_traits = Counter(), Counter(), Counter(), Counter()
    false_accepts, accepted = [], []
    for r in rows:
        rid = r['metadata']['scenario_id']
        m = by_id[rid]
        assert m['verdict'] in ('pass', 'fail')
        assert (m['verdict'] == 'pass') == (not m['defects'])
        domains[m['domain']] += 1
        mechanisms[manual['mechanism_groups'][rid]] += 1
        codes.update(m['defects'])
        if m['verdict'] == 'pass':
            accepted.append(rid)
            accepted_traits[r['metadata']['trait_id']] += 1
        elif r['metadata']['review']['verdict'] == 'pass':
            false_accepts.append(rid)
        r['metadata']['independent_review'] = m
    contract = cfg['smoke_contract']
    gates = {'all_judges_complete': len(reviews) == 18,
        'minimum_acceptable': len(accepted) >= contract['minimum_independently_acceptable'],
        'every_trait': all(accepted_traits[f't{i}'] >= 1 for i in range(1, 10)),
        'mechanism_diversity': max(mechanisms.values()) <= contract['maximum_same_decision_mechanism'],
        'domain_diversity': len(domains) >= contract['minimum_actual_domains'],
        'no_repeated_material_defect': max(codes.values(), default=0) <= contract['maximum_repeated_material_defect'],
        'no_material_false_acceptance': not false_accepts}
    cost = json.loads((root/'cost_summary.json').read_text(encoding='utf-8'))
    summary = dict(overall='pass' if all(gates.values()) else 'fail', gates=gates,
        planned=18, completed=len(rows), completed_judges=len(reviews), model_pass=sum(r['metadata']['review']['verdict']=='pass' for r in rows),
        independent_pass=len(accepted), accepted_ids=accepted, accepted_per_trait=dict(accepted_traits),
        model_false_accepts=false_accepts, defects=dict(codes), domains=dict(domains), mechanisms=dict(mechanisms),
        cost=cost, reviewer='Codex full read of all system/user/reasoning/response; not human review',
        not_a_validated_error_rate=True, conclusions=manual['conclusions'])
    write_json(root/'results.json', summary)
    shutil.copy2(args.review, root/'independent_review.yaml')
    (root/'dataset.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (root/'stages').mkdir(exist_ok=True)
    for file in (root/'generation').glob('stage_*.jsonl'):
        shutil.copy2(file, root/'stages'/file.name)
    sections = ['# Constitution-only low-stakes smoke', '',
        '**Diagnostic candidates, NOT an approved training corpus.**', '',
        f"Result: {summary['overall'].upper()}. Model passes {summary['model_pass']}/18; independent passes {len(accepted)}/18.",
        f"API cost ${cost['charged_or_reserved_usd']:.6f}; {cost['physical_calls']} physical calls; no paid retries or replacements.", '',
        *[f'- {c}' for c in manual['conclusions']], '',
        'The six calibration cases were reviewer tests only, never content inputs to generation. Seed controls coverage assignment; hosted model sampling is not byte-reproducible.', '',
        '## All candidates and independent review', '']
    for r in rows:
        rid = r['metadata']['scenario_id']
        m = by_id[rid]
        sections += [f"### {rid}: {m['verdict'].upper()}", '', m['reason'], '',
            f"Model verdict: {r['metadata']['review']['verdict']}", '']
        for msg in r['messages']:
            sections += [f"**{msg['role']}**", '', msg['content'], '']
            if 'reasoning_content' in msg:
                sections += ['**Explicit reasoning**', '', msg['reasoning_content'], '']
    (root/'results.md').write_text('\n'.join(sections), encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if args.publish:
        repo = hf_repo_id(artifact_name(cfg['pipeline']+' synth smoke'))
        api = hf_api()
        if api.repo_exists(repo, repo_type='dataset'):
            raise RuntimeError('Refusing to overwrite an existing smoke publication')
        fields = dict(experiment='18-row constitution-only low-stakes smoke; FAILED scaling gate; diagnostic candidates only',
            date_generated=manifest['run_id'], constitution=cfg['constitution']+' sha256 '+manifest['constitution_sha256'],
            source_repo=origin_url()+' @ '+git_sha(), models='anthropic/claude-sonnet-5; Anthropic provider; all author and reviewer calls',
            generation_config='da-lowstakes-constitution.yaml; frozen script, prompts, provider pin, calibration, raw calls and results included',
            schema='dataset.jsonl: all 18 diagnostic candidates including failures; stages/: every intermediate; results.md: full transcripts and review',
            provenance=manifest['command'])
        front = {'configs':[{'config_name':'diagnostic_candidates','data_files':'dataset.jsonl','default':True}],
                 'tags':training_data_tags('synth',cfg['pipeline'],cfg['constitution'],smoke=True)}
        url = push_run_dir(root,repo,fields,front_matter=front)
        info = api.dataset_info(repo)
        receipt = {'url':url, 'revision':info.sha, 'local_dataset_sha256':hashlib.sha256((root/'dataset.jsonl').read_bytes()).hexdigest()}
        write_json(root/'publication.json',receipt)
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
