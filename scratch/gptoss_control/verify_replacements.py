# ABOUTME: Independently verify the CoT-only replacement contract and enumerate retained failures.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python scratch/gptoss_control/verify_replacements.py
from pathlib import Path
import collections, hashlib, json, sys
from omegaconf import OmegaConf
sys.path.insert(0,str(Path(__file__).parent))
from run import readrows,write,saverows,digest

cfg=OmegaConf.load(Path(__file__).with_name('reasoning_rebuild.yaml'))
out=Path(cfg.output)
before=readrows(out/'converted_unenriched.jsonl')
after=readrows(out/'dataset/mixture.jsonl')
assert len(before)==len(after)==10000
records=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'backfill_receipts').glob('*.json')]
assert not any(r['status'] in {'reserved','sampled','judge_reserved','judge_budget_stop'} for r in records)
accepted={(r['row'],r['turn']):r for r in records if r.get('accepted')}
targets={(i,j) for i,r in enumerate(before) for j,m in enumerate(r['messages']) if m.get('reasoning_content')}
failures=readrows(out/'failed_replacements.jsonl')
failed={(r['row'],r['turn']) for r in failures}
assert set(accepted).isdisjoint(failed)
assert set(accepted)|failed==targets
assert len(failures)==len(failed)
changed=[]; sources=collections.defaultdict(collections.Counter)
for i,(original,revised) in enumerate(zip(before,after)):
 normalized=json.loads(json.dumps(revised))
 assert len(original['messages'])==len(revised['messages'])
 for j,(a,b) in enumerate(zip(original['messages'],revised['messages'])):
  key=(i,j)
  if key in accepted:
   receipt=accepted[key]
   assert receipt['judge']['verdict']=='yes'
   verdict=json.loads(receipt['judge']['raw'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
   assert verdict['answer_agreement'] is True and verdict['trace_compatible'] is True
   assert receipt['trace']==b['reasoning_content'] and receipt['trace'].strip()
   normalized['messages'][j]['reasoning_content']=a['reasoning_content']
   sources[original['source']]['accepted']+=1
   if b['reasoning_content']!=a['reasoning_content']: changed.append({'row':i,'turn':j})
  elif key in failed:
   assert a==b
   sources[original['source']]['failed_retained']+=1
  else:
   assert a==b
 assert normalized==original, f'Non-CoT content changed at row {i}'
for failure in failures:
 assert failure['original_row_sha256']==digest(before[failure['row']])
 assert failure['attempt_count']==int(cfg.backfill.attempts), failure
assert sum(bool(m.get('reasoning_content')) for r in after for m in r['messages'])==len(targets)
ledger=sorted(records,key=lambda r:(r['row'],r['turn'],r['attempt']))
saverows(out/'dataset/backfill_attempts_all.jsonl',ledger)
write(out/'dataset/changed_trace_indices.json',changed)
report={'source_repo':cfg.source_repo,'source_revision':cfg.source_revision,'rows':len(before),
 'assistant_turns':sum(m['role']=='assistant' for r in before for m in r['messages']),
 'targeted_existing_cot_turns':len(targets),'accepted_replacements':len(accepted),
 'actually_changed_trace_strings':len(changed),'identical_regenerated_trace_strings':len(accepted)-len(changed),
 'failed_replacements_retained_unchanged':len(failed),
 'untraced_assistant_turns_preserved':sum(m['role']=='assistant' and not m.get('reasoning_content') for r in before for m in r['messages']),
 'all_non_cot_fields_identical':True,'all_failed_entries_identical':True,
 'by_source':{k:dict(v) for k,v in sources.items()},
 'dataset_file_sha256':hashlib.sha256((out/'dataset/mixture.jsonl').read_bytes()).hexdigest()}
accounting=json.loads((out/'backfill_report.json').read_text())
qualification_cost=0
qualification_paths=list(out.glob('judge_qualification*/results.json'))+list(Path(cfg.backfill.reuse_generations_from).glob('judge_qualification*/results.json'))
for path in qualification_paths:
    qualification_cost+=sum(r.get('judge_cost') or 0 for r in json.loads(path.read_text(encoding='utf-8')))
report['accounting']={'generation_usd_upper':accounting['target_cost_upper_usd'],
    'judging_usd_upper':accounting['judge_cost_upper_usd'], 'qualification_usd':qualification_cost,
    'total_usd_upper':accounting['target_cost_upper_usd']+accounting['judge_cost_upper_usd']+qualification_cost,
    'note':'Includes prior discarded judge runs and conservative reservations for interrupted/unsettled requests; not an invoice.'}
report['attempt_receipts']=len(records)
report['attempt_status_counts']=dict(collections.Counter(r['status'] for r in records))
write(out/'replacement_verification.json',report)
write(out/'dataset/replacement_verification.json',report)
lines=['# Failed CoT replacements','',
 'All indices are zero-based in the pinned source mixture. These entries retain their original CoT and answer unchanged.',
 '', '| Row | Assistant turn | Source | Attempts | Last outcome |','|---:|---:|---|---:|---|']
for r in sorted(failures,key=lambda r:(r['row'],r['turn'])):
 last=max(r['attempts'],key=lambda a:a['attempt'])
 reason=(last.get('judge') or {}).get('raw') or last.get('error_type') or ('No valid nonempty completed reasoning trace: '+str(last.get('termination')))
 reason=reason.replace('\n',' ').replace('|','/').replace('```','')
 lines.append(f"| {r['row']} | {r['turn']} | {r['source']} | {r['attempt_count']} | {reason} |")
(out/'dataset/failed_replacements.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
