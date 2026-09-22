# ABOUTME: Audit exact independent-review proposals against the current selected pool without accepting them.
# ABOUTME: Reuse native Qwen masking and pinned local corpus checks; freeze all input hashes in a new output.
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scratch.dataset_refresh import offline_acceptance as acceptance, run as base, audit_corpus
from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
from src.model_profile import model_profile


def audit(config_path):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    output = Path(cfg['output']).resolve()
    if output.exists():
        raise ValueError('Use a fresh proposal-audit output')
    base_path = acceptance.bound(cfg['base_path'], cfg['base_sha256'])
    selected = base.read_rows(base_path)
    sources = []
    for ref in cfg['accepted']:
        row, item = acceptance.validate_accepted(ref['path'], ref['sha256'])
        selected.append(row)
        sources.append(item['dossier']['source_ref'])
    proposals = []
    for ref in cfg['proposals']:
        path = acceptance.bound(ref['dossier_path'], ref['dossier_sha256'])
        d, files = acceptance.validate_dossier(path)
        review = acceptance.bound(ref['review_path'], ref['review_sha256'])
        acceptance.validate_review(d, files, acceptance.read(review))
        if d['source_ref'] in sources:
            raise ValueError('Proposal source is already accepted')
        sources.append(d['source_ref'])
        proposals.append({'messages': acceptance.messages(d['conversation']),
                          'metadata': {**d['source_metadata'],
                                       'scenario_id': 'proposal_'+d['conversation_sha256'][:20]}})
    if not proposals:
        raise ValueError('No independently accepted content proposals')
    output.mkdir(parents=True)
    base.write_json(output/'input_manifest.json', {'config': cfg, 'config_sha256': acceptance.sha(config_path),
        'helper_sha256': acceptance.sha(__file__), 'automatic_acceptance': False})
    base.write_rows(output/'proposals.jsonl', proposals)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    profile = model_profile(TOKENIZER)
    diagnostics = [{'id': r['metadata']['scenario_id'], **token_audit(r, tokenizer, profile, 8192)} for r in proposals]
    native = {'input_sha256': acceptance.sha(output/'proposals.jsonl'), 'rows': len(proposals),
        'status': 'passed', 'tokenizer': TOKENIZER, 'max_train_tokens': 8192, 'untruncated': True,
        'failures': [], 'diagnostics': diagnostics}
    base.write_json(output/'token_mask_audit.json', native)
    combined = selected+proposals
    base.write_rows(output/'prospective.jsonl', combined)
    report = audit_corpus.audit(output/'prospective.jsonl', output, local_files_only=True)
    new_ids = {r['metadata']['scenario_id'] for r in proposals}
    for kind in ('semantic', 'lexical'):
        pairs = [p for p in base.read_rows(output/(kind+'_pairs.jsonl')) if p['a'] in new_ids or p['b'] in new_ids]
        base.write_rows(output/('new_'+kind+'_pairs.jsonl'), pairs)
    summary = {'selected_before': len(selected), 'proposals': len(proposals), 'prospective_rows': len(combined),
        'native_pass': True, 'corpus': report, 'automatic_accepted_rows': 0,
        'scope': 'Native formatting and duplicate candidate evidence only. Separate root duplicate/content adjudication and adoption required.'}
    base.write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = audit(args.config)
    print({k: v for k, v in result.items() if k != 'corpus'})
