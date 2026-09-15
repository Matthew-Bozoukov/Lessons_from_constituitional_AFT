# ABOUTME: Freezes all current qualified nonmoral independent exclusions for correction reassessment.
# ABOUTME: Reads source hashes and original evidence without changing rows or calling models.
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as runtime

parser=argparse.ArgumentParser(description='Freeze exclusion inventory once; run with python -m scratch.dataset_refresh.nonmoral_correction_inventory --config PATH')
parser.add_argument('--config',required=True,help='YAML with source_root (records directory) and output_dir.')
args=parser.parse_args()
config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
root=Path(config['source_root']).resolve()
out=Path(config['output_dir']).resolve()
target=out/'inventory.json'
if target.exists() or target.with_suffix('.receipt.json').exists():
    raise FileExistsError('Refusing to overwrite an existing immutable inventory or receipt: '+str(target))
if not root.is_dir():
    raise FileNotFoundError('Source records directory does not exist: '+str(root))
out.mkdir(parents=True,exist_ok=True)
rows=[]
for path in sorted(root.glob('*/independent_exclusion.json')):
    note=runtime.load_checkpoint(path)
    result=path.parent/'result.json'
    raw=runtime.load_checkpoint(result)
    assert runtime.digest(result.read_bytes())==note['result_sha256']
    reason=note['reason']
    details=note
    if '{' in reason:
        try: details=json.loads(reason[reason.index('{'):])
        except json.JSONDecodeError: pass
    evidence=details.get('evidence')
    summaries=[]
    if isinstance(evidence,list):
        summaries=[{k:e.get(k) for k in ['reason','evidence','caveat','family','independent_disposition']} for e in evidence if isinstance(e,dict)]
    if not summaries:
        summaries=[dict(reason=details.get('reason',reason),evidence=evidence,caveat=details.get('caveat'),family=details.get('family'))]
    rows.append(dict(candidate_id=path.parent.name,result_path=str(result),result_sha256=runtime.digest(result.read_bytes()),exclusion_path=str(path),exclusion_sha256=runtime.digest(path.read_bytes()),automatic_status=raw['status'],prior_disposition=note.get('independent_disposition',details.get('selection_disposition')),prior_note=note,parsed_details=details,summary=summaries,default_assessment='uncertain_pending_reassessment'))
runtime.save_checkpoint(target,dict(count=len(rows),scope=f'All {len(rows)} currently excluded qualified nonmoral terminals, frozen before correction rereads; no restored or edited sources.',rows=rows))
for row in rows:
    print(json.dumps({k:row[k] for k in ['candidate_id','prior_disposition','summary']},ensure_ascii=False))
