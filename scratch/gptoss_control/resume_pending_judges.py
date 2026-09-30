# ABOUTME: Finish explicitly unsent judges after a backfill budget stop without resampling the model.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/resume_pending_judges.py
from pathlib import Path
import argparse, json, subprocess, sys, shutil
from dotenv import load_dotenv
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).parent))
from run import readrows,write,agreement_prompt,judge_agreement
from src.infra.endpoints.openrouter import OpenRouterClient,provider_price
from src.infra.endpoints.harmony import make_renderer,supervised_examples

parser=argparse.ArgumentParser();parser.add_argument('--retry-parser-row',type=int);args=parser.parse_args()
cfg=OmegaConf.load(Path(__file__).with_name('reasoning_rebuild.yaml'))
common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
out=Path(cfg.output);rows=readrows(out/'converted_unenriched.jsonl')
paths=list((out/'backfill_receipts').glob('*.json'))
records=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
assert not any(r['status'] in {'reserved','sampled','judge_reserved'} for r in records),'Uncertain requests need separate reconciliation'
prior=json.loads((out/'inherited_costs.json').read_text())
charged=sum(r.get('judge_cost_upper_usd',0) for r in records)+prior['judge_usd_upper']
price=provider_price(cfg.backfill.judge);renderer=make_renderer(cfg.reasoning)
for path,rec in zip(paths,records):
 parser_retry=(args.retry_parser_row == rec['row'] and rec['status']=='request_error' and rec.get('error_type')=='AttributeError')
 if rec['status']!='judge_budget_stop' and not parser_retry: continue
 prior_charge=rec.get('judge_cost_upper_usd',0) if parser_retry else 0
 i,j=rec['row'],rec['turn'];row=rows[i]
 prompt=agreement_prompt(row['messages'][:j],row['tools'],rec['response'],row['messages'][j],rec['trace'])
 upper=((len(prompt.encode('utf-8'))+1024)*price['in']+cfg.backfill.judge_max_tokens*price['out'])/1e6
 if charged+upper>cfg.backfill.max_judge_cost_usd: raise RuntimeError('Judge ceiling insufficient')
 archive=out/'resumed_judge_receipts';archive.mkdir(exist_ok=True)
 shutil.copy2(path,archive/path.name)
 rec.update(status='judge_reserved',judge_cost_upper_usd=prior_charge+upper,resumed_unsent_judge=not parser_retry, retried_parser_failure=parser_retry, previous_judge_reservation_usd=prior_charge)
 write(path,rec);charged+=upper
 client=OpenRouterClient();client.chat=lambda **kw:OpenRouterClient.chat.__wrapped__(client,**kw)
 result=judge_agreement(client,cfg.backfill.judge,prompt,cfg.backfill.judge_max_tokens,
                        cfg.backfill.structured_judge,OmegaConf.to_container(cfg.backfill.judge_reasoning))
 settled=result['judge_cost'] if result['judge_cost'] is not None else upper
 charged+=settled-upper
 rec.update(judge=result,judge_cost_upper_usd=prior_charge+settled,accepted=result['verdict']=='yes',status='completed')
 if parser_retry:
  rec['previous_error_type']=rec.pop('error_type')
  rec['previous_error']=rec.pop('error')
 if rec['accepted']:
  trial=json.loads(json.dumps(row));trial['messages'][j]['reasoning_content']=rec['trace']
  supervised_examples(renderer,trial,cfg.train.max_length)
 write(path,rec)
 print('resumed',i,j,'accepted',rec['accepted'],'cost',settled,flush=True)
