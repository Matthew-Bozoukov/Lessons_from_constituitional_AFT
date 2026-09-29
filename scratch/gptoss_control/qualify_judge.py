# ABOUTME: Qualify the configured agreement judge against fixed positive and negative controls.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/qualify_judge.py
from pathlib import Path
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).parent))
from run import agreement_prompt, judge_agreement, write, readrows
from src.infra.endpoints.openrouter import OpenRouterClient, provider_price

cfg=OmegaConf.load(Path(__file__).with_name('reasoning_rebuild.yaml'))
common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
out=Path(cfg.output)/'judge_qualification';out.mkdir(parents=True,exist_ok=True)
cases=[
 ('equivalent', 'What is 2+2?', '4', 'The answer is four.', 'Adding two and two gives four.', 'yes'),
 ('contradiction', 'What is 2+2?', '4', '5', 'I calculate five.', 'no'),
 ('trace_mismatch','What is 2+2?', '4', '4', 'I will give five as the answer.', 'no'),
 ('proper_name','Summarize without second or third person pronouns: Alex sent the paper.', 'Alex sent the paper.', 'The paper was sent by Alex.', 'Use Alex, a proper name, to avoid pronouns; summarize the paper being sent.', 'yes'),
 ('different_story','Write a short story.', 'A cat found a lost crown.', 'An astronaut sailed to Mars.', 'I will write about an astronaut travelling to Mars.', 'no'),
 ('refusal_vs_action','Write a friendly greeting.', 'Hello!', 'I cannot help with that.', 'I will refuse to greet.', 'no'),
]
cases += [
 ('passive_holdout', 'Who approved the budget?', 'Priya approved the budget.', 'The budget was approved by Priya.', 'The record identifies Priya as the approver.', 'yes'),
 ('actor_changed', 'Who approved the budget?', 'Priya approved the budget.', 'Marco approved the budget.', 'Marco is the approver.', 'no'),
 ('order_holdout', 'Summarize: meeting Tuesday; send draft Friday.', 'The meeting is Tuesday. Send the draft Friday.', 'Send the draft Friday; the meeting is Tuesday.', 'Include both the Tuesday meeting and Friday draft deadline.', 'yes'),
 ('number_changed', 'State the meeting time: 10:30.', '10:30', '11:30', 'The meeting starts at eleven thirty.', 'no'),
 ('code_equivalent', 'Write Python to double x.', 'def double(x): return x * 2', 'def double(x): return 2 * x', 'Multiply the input by two and return the result.', 'yes'),
]
cases += [
 ('case_sensitive_code', 'Sort letters alphabetically.', "return ''.join(sorted(s))", "return ''.join(sorted(s, key=str.lower))", 'Sort the letters and join them.', 'no'),
 ('empty_input_code', 'Validate a string of letters.', "if not s.isalpha(): raise ValueError()", "if not all(c.isalpha() for c in s): raise ValueError()", 'Validate that the string contains letters.', 'no'),
 ('comparison_boundary', 'Return whether x meets the threshold.', 'return x >= 10', 'return x > 10', 'Compare with ten.', 'no'),
]
jobs=[]
for name,prompt,reference,generated,trace,expected in cases:
 jobs.append({'name':name,'expected':expected,'prompt':agreement_prompt([{'role':'user','content':prompt}],[],{'content':generated},{'content':reference},trace)})
prior=Path('output/gptoss_control/reasoning_rebuild_2026-09-29')
rows=readrows(prior/'converted_unenriched.jsonl')
for i in [1173,3866]:
 rec=json.loads((prior/'backfill_receipts'/f'{i}_2_0.json').read_text(encoding='utf-8'))
 row=rows[i]
 jobs.append({'name':f'pilot_rejection_{i}','expected':None,'prompt':agreement_prompt(row['messages'][:2],row['tools'],rec['response'],row['messages'][2],rec['trace'])})
real_dir=Path(cfg.backfill.reuse_generations_from)
real_rows=readrows(real_dir/'converted_unenriched.jsonl')
for f in (real_dir/'backfill_receipts').glob('164_*.json'):
 r=json.loads(f.read_text(encoding='utf-8'))
 if r.get('accepted'):
  row=real_rows[164]
  jobs.append({'name':'real_sorting_false_accept','expected':'no','prompt':agreement_prompt(row['messages'][:r['turn']],row['tools'],r['response'],row['messages'][r['turn']],r['trace'])})
  break
price=provider_price(cfg.backfill.judge)
ceiling=sum(((len(j['prompt'].encode())+1024)*price['in']+cfg.backfill.judge_max_tokens*price['out'])/1e6 for j in jobs)
assert ceiling < 1, ceiling
write(out/'plan.json',{'max_cost_upper_usd':ceiling,'judge':cfg.backfill.judge,'cases':jobs})
def job(case):
 path=out/(case['name']+'.json')
 if path.exists(): raise RuntimeError('Existing qualification receipt; inspect before rerunning')
 write(path,{'status':'reserved','case':case})
 client=OpenRouterClient();client.chat=lambda **kw:OpenRouterClient.chat.__wrapped__(client,**kw)
 result=judge_agreement(client,cfg.backfill.judge,case['prompt'],cfg.backfill.judge_max_tokens,True,OmegaConf.to_container(cfg.backfill.judge_reasoning))
 write(path,{'status':'completed','case':case,'result':result})
 return {'name':case['name'],'expected':case['expected'],**result}
with ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(job,jobs))
write(out/'results.json',results)
for r in results: print(r['name'],r['expected'],r['verdict'],r['raw'])
assert all(r['expected'] is None or r['verdict']==r['expected'] for r in results)
print('controls passed; cost',sum(r.get('judge_cost') or 0 for r in results))
