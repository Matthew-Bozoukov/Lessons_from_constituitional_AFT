# ABOUTME: Measures the actual assistant/CoT supervision and structural shifts in the pinned DA rows.
# ABOUTME: Run: uv run python scratch/da_refresh_investigation/structure.py
import collections
import json
from pathlib import Path
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from src.model_profile import model_profile
from src.data.mixture.build_mixture import supervised_tokens
from scratch.da_refresh_investigation.judge import OUT,save

def main():
    base='Qwen/Qwen3.6-27B';revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
    tok=AutoTokenizer.from_pretrained(base,revision=revision);profile=model_profile(base)
    summary={};measurements=[]
    for arm in ['old','new']:
        rows=[json.loads(s) for s in (OUT/f'{arm}_selected.jsonl').read_text(encoding='utf-8').splitlines()]
        vals=[]
        for row in rows:
            record={'messages':row['messages']}
            all_count=supervised_tokens(tok,profile,record,8192)
            cot=supervised_tokens(tok,profile,{**record,'supervise':'cot'},8192)
            answer=supervised_tokens(tok,profile,{**record,'supervise':'answer'},8192)
            assert cot+answer==all_count
            meta=row['source_row']['metadata']
            texts={field:'\n'.join(m.get(field) or '' for m in row['messages'] if m['role']=='assistant') for field in ['content','reasoning_content']}
            prompt='\n'.join(m.get('content') or '' for m in row['messages'] if m['role'] in ['system','user'])
            v={'id':row['id'],'trait':meta['trait_id'],'scenario_id':meta['scenario_id'],'all_tokens':all_count,'cot_tokens':cot,'answer_tokens':answer,'reasoning_words':len(texts['reasoning_content'].split()),'answer_words':len(texts['content'].split()),'user_words':len(prompt.split()),'messages':len(row['messages']),'ai_lexical_prompt':bool(re.search(r'\bAI\b|artificial intelligence',prompt,re.I))}
            vals.append(v);measurements.append(v)
        summary[arm]={'rows':len(rows),'supervised_tokens':sum(v['all_tokens'] for v in vals),'cot_tokens':sum(v['cot_tokens'] for v in vals),'answer_tokens':sum(v['answer_tokens'] for v in vals),'cot_share':sum(v['cot_tokens'] for v in vals)/sum(v['all_tokens'] for v in vals),'means':{k:statistics.mean(v[k] for v in vals) for k in ['cot_tokens','answer_tokens','user_words','reasoning_words','answer_words']},'median_cot_tokens':statistics.median(v['cot_tokens'] for v in vals),'messages':dict(collections.Counter(v['messages'] for v in vals)),'ai_lexical_prompt':sum(v['ai_lexical_prompt'] for v in vals),'max_supervised_tokens':max(v['all_tokens'] for v in vals)}
    save(OUT/'structure.json',{'base':base,'revision':revision,'summary':summary,'rows':measurements})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
