# ABOUTME: Audit and publish the exact practical low-stakes mixture using the shared native validators.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.release_practical_mixture <build-dir>
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scratch.dataset_refresh.validate_mixtures import audit, digest, BASE_REPO, BASE_REVISION
from src.infra.huggingface import push_run_dir, hf_api, hf_org, training_data_tags
from src.naming import mix_name


def main(build):
    load_dotenv('.env')
    build = Path(build)
    cfg_path = Path('scratch/dataset_refresh/da-lowstakes-practical.yaml')
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    style = cfg_path.stem
    spec = cfg['sources'][style]
    synthetic = hf_hub_download(spec['dataset'], 'dataset.jsonl', repo_type='dataset', revision=spec['revision'])
    assert digest(Path(synthetic).read_bytes()) == 'd324b06084e46494b3920773deead8632f86cdb789752e7dd25819a6906a17f8'
    base = hf_hub_download(BASE_REPO, 'mixture.jsonl', repo_type='dataset', revision=BASE_REVISION)
    reference = Path('output/2026-09-15_dataset_refresh_shared_replay/replay_reference.json')
    report = audit([build/'mixture.jsonl'], base, AutoTokenizer.from_pretrained(cfg['tokenizer'], local_files_only=True),
        synthetic_sources=[dict(repo=spec['dataset'], revision=spec['revision'], style=style, path=synthetic)],
        single_arm=True, replay_reference_path=reference)
    (build/'mixture_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    shutil.copyfile(reference, build/'replay_reference.json')
    shutil.copyfile(cfg_path, build/'mixture_config.yaml')
    shutil.copyfile(cfg['base'], build/'replay-proportions.yaml')
    meta = json.loads((build/'run_meta.json').read_text(encoding='utf-8'))
    date = datetime.fromisoformat(meta['timestamp_utc']).date().isoformat()
    name = mix_name(style, 7, date=date)
    constitution = 'constitutions/claude_distilled_09_principles/constitution.md'
    fields = dict(title=name, date_generated=date,
        experiment='716 practical low-stakes human-advice rows + the identical 9,284 September8 nosynth rows. 10,000 total; exact synthetic row share 7.16%.',
        constitution=constitution+'; SHA256 '+digest(Path(constitution).read_bytes()),
        source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+meta['git_sha'],
        models='Synthetic Sonnet generation/review provenance is pinned in '+spec['dataset']+'@'+spec['revision']+'; replay Qwen reasoning inherited unchanged from '+BASE_REPO+'@'+BASE_REVISION,
        generation_config=json.dumps(cfg),
        schema='Default train: mixture.jsonl; messages with content and optional reasoning_content/tool_calls, source, n_tokens, optional tools; synthetic supervise=all. No truncation or response editing.',
        provenance=meta['command']+'; immutable synthetic pin '+spec['revision']+'; replay pin '+BASE_REVISION,
        validation='All 10,000 full rendered streams and assistant masks checked; all716 synthetic payloads preserved; all9284 replay payloads AND positions match frozen September15 reference. See mixture_validation.json for token counts.',
        limitations='Source generation limitations remain documented in the pinned synthetic repository review/report.md. Mixing does not improve content quality.')
    front = {'configs':[{'config_name':'default','data_files':'mixture.jsonl','default':True}], 'tags':training_data_tags('mixture',style,constitution)}
    url = push_run_dir(build,name,fields,front_matter=front)
    repo = hf_org()+'/'+name
    revision = hf_api().dataset_info(repo).sha
    restored = hf_hub_download(repo,'mixture.jsonl',repo_type='dataset',revision=revision,force_download=True)
    assert digest(Path(restored).read_bytes()) == report['mixtures'][0]['sha256']
    receipt = dict(repo=repo,revision=revision,url=url,sha256=report['mixtures'][0]['sha256'],rows=10000)
    Path('output/lowstakes_practical_training/mixture_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
