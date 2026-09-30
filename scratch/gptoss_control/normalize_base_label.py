# ABOUTME: Correct a recovery publisher's display-only mode label from the immutable launch metadata.
# ABOUTME: Updates README tags and results Markdown only, retaining all scores, transcripts and original receipt.
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
from src.eval.run_eval import _results_markdown
from src.infra.huggingface import hf_api,hf_download


def main():
    out=ROOT/OmegaConf.load(ROOT/'scratch/gptoss_control/base_odcv.yaml').output
    run,=(out/'odcv_fixed').iterdir()
    receipt=out/'published_eval_fixed.json'
    pin=json.loads(receipt.read_text())
    meta=json.loads((run/'metadata/run_meta.json').read_text())
    result=json.loads((run/'results/results.json').read_text())
    assert result['mode']==meta['mode']=='harmony_medium'
    readme=run/'README.md'
    text=readme.read_text(encoding='utf-8')
    assert 'mode:default' in text and 'mode:harmony_medium' not in text
    readme.write_text(text.replace('mode:default','mode:harmony_medium'),encoding='utf-8')
    (run/'results/results.md').write_text(_results_markdown(meta['target'],meta['mode'],result),encoding='utf-8')
    backup=out/'published_eval_fixed_before_label_fix.json'
    assert not backup.exists()
    shutil.copy2(receipt,backup)
    api=hf_api()
    assert api.dataset_info(pin['repo']).sha==pin['revision']
    paths=['README.md','results/results.md']
    commit=api.create_commit(repo_id=pin['repo'],repo_type='dataset',parent_commit=pin['revision'],
        operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=run/p) for p in paths],
        commit_message='Correct recovery display label to recorded harmony_medium protocol; scores unchanged')
    for p in paths:
        assert Path(hf_download(pin['repo'],p,repo_type='dataset',revision=commit.oid)).read_bytes()==(run/p).read_bytes()
    pin['revision']=commit.oid
    receipt.write_text(json.dumps(pin,indent=2),encoding='utf-8')
    print('Verified mode display labels; no rollout or score changes',commit.oid)


if __name__=='__main__':
    main()
