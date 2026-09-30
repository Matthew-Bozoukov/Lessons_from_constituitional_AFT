# ABOUTME: Prepare the pinned replacement control and compare original/fixed ODCV tool prompts.
# ABOUTME: Reuses the qualified Tinker training and evaluation drivers with isolated outputs.
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from run import audit_data, write, provenance
from src.infra.endpoints import harmony
from src.infra.huggingface import hf_download, hf_api


def prepare(cfg, out):
    pin = {'repo':cfg.source_repo, 'revision':cfg.source_revision}
    path = Path(hf_download(pin['repo'],'mixture.jsonl',repo_type='dataset',revision=pin['revision']))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'd792bcbb3e9ed62d6b02b010db3add413e931a142ce31c824ca39c7143d91b6c'
    summary_path = hf_download(pin['repo'],'native_cot_provenance_summary.json',repo_type='dataset',revision=pin['revision'])
    summary = json.loads(Path(summary_path).read_text())
    assert summary['passed'] and summary['unresolved_model_provenance'] == 0
    write(out/'native_cot_provenance_summary.json',summary)
    write(out/'published_dataset.json',pin)
    frozen = json.loads((ROOT/'output/gptoss_control/base_tool_probe_2026-09-29/prompts.json').read_text(encoding='utf-8'))
    rendered = []
    for variant, revision in [('original','87957e51'),('fixed','4e9fe0d321ad7db5e42a04a85b380b1582e284af')]:
        module = types.ModuleType('pinned_harmony_'+variant)
        source = subprocess.check_output(['git','show',revision+':src/infra/endpoints/harmony.py'],text=True)
        exec(compile(source,revision,'exec'),module.__dict__)
        renderer = harmony.make_renderer(tool_prompt=variant,local_files_only=True)
        for i,entry in enumerate(frozen):
            body=entry['body']
            tokens=harmony.render_prompt(renderer,body['messages'],body['tools']).to_ints()
            assert tokens == module.render_prompt(renderer,body['messages'],body['tools']).to_ints()
            if variant=='original':
                assert tokens == entry['prompt_token_ids']
            rendered.append({'variant':variant,'example':i,'reference_revision':revision,'tokens':tokens,
                'text':renderer.tokenizer.decode(tokens)})
    write(out/'prompt_qualification.json',{'passed':True,'examples':rendered})
    report=audit_data(cfg,out,path,'final_audit')
    assert report['total']['rows']==10000 and report['total']['supervised_tokens']==2352071
    assert report['total']['processed_tokens']==5997497
    write(out/'training_launch_observation.json',{'training_code_commit':provenance(cfg)['git_sha'],
        'dataset':pin,'training_tool_prompt':'fixed','prompt_qualification':'prompt_qualification.json',
        'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print('Prepared pinned dataset; both prompt variants match archived tokens; training estimate',report['estimated_training_usd'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare'])
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    args=p.parse_args()
    os.chdir(ROOT)
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
    load_dotenv(common.parent/'.env')
    cfg=OmegaConf.load(args.config)
    out=ROOT/cfg.output
    out.mkdir(parents=True,exist_ok=True)
    prepare(cfg,out)


if __name__=='__main__':
    main()
