# ABOUTME: Diagnose Qwen Arena losses from immutable generations and existing judgments only.
# ABOUTME: Descriptive subsets are sensitivity checks, not replacement benchmark scores.
from pathlib import Path
import json
import hashlib
import random
import subprocess
import numpy as np
from src.infra.huggingface import hf_download
from src.utils import read_jsonl
from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments
from src.eval.capabilities.arena_hard.arena_hard_metrics import is_refusal, repetition_ratio

OUT = Path('output/arena_qwen_diagnosis_20261014')
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'base': ('2026-10-13-ah-qwen36', '0d220b9416c7ece98a185bad5bd32dc275f9fad5'),
    'control': ('2026-10-12-ah-qwen36-0-plain', 'bc2fb7d0d17202357d8f04c3849354fee76ec0a4'),
    'da15': ('2026-10-12-ah-qwen36-0-da-15', 'e15e030aaaf72fc05cad06bead91fa7284082553'),
    'judges': ('2026-10-13-ah-vs-qwen36', '99f48c89e31c41626d9311a4f4a95fc965888302'),
}
receipts = []

def read(key, file):
    repo, revision = SOURCES[key]
    p = Path(hf_download('dougalldeepmind/'+repo, file, repo_type='dataset', revision=revision))
    receipts.append({'repo': repo, 'revision': revision, 'file': file,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return read_jsonl(p)

answers = {k:{r['uid']:r for r in read(k,'rollouts/answers.jsonl')} for k in ['base','control','da15']}
assert all(set(a)==set(answers['base']) for a in answers.values())
for uid in answers['base']:
    assert len({a[uid]['generation_record']['prompt'] for a in answers.values()})==1
    assert len({a[uid]['generation_record']['max_tokens'] for a in answers.values()})==1
def rec(a): return a['generation_record']
def text(a): return a['messages'][-1]['content']['answer']
def length(a): return a['metadata']['token_len']

stats = {'arm_health': {}, 'comparisons': {}}
for category in ['hard_prompt', 'creative_writing']:
    ids = sorted(uid for uid,a in answers['base'].items() if rec(a)['category']==category)
    assert len(ids)==(500 if category=='hard_prompt' else 250)
    for arm, rows in answers.items():
        rr = [rows[u] for u in ids]
        nonempty = [a for a in rr if text(a).strip()]
        stats['arm_health'][category+'_'+arm] = {
            'n':len(rr), 'empty':sum(not text(a).strip() for a in rr),
            'length_finish':sum(rec(a)['finish_reason']=='length' for a in rr),
            'empty_and_length_finish':sum(not text(a).strip() and rec(a)['finish_reason']=='length' for a in rr),
            'empty_think':sum(not rec(a)['think'].strip() for a in rr),
            'median_answer_tokens':float(np.median([length(a) for a in rr])),
            'mean_answer_tokens':float(np.mean([length(a) for a in rr])),
            'median_nonempty_answer_tokens':float(np.median([length(a) for a in nonempty])),
            'median_reasoning_words':float(np.median([len(rec(a)['think'].split()) for a in rr])),
            'refusal_cue':sum(is_refusal(text(a)) for a in rr),
            'repetition_over_30pct':sum(repetition_ratio(text(a))>.3 for a in rr),
        }

examples=[]
for arm,key in [('control','qwen36_0_plain'),('da15','qwen36_0_da_15')]:
    judgments = read('judges',f'rollouts/judgments_{key}.jsonl')
    jmap = {j['uid']:j for j in judgments}
    bs = battles_from_judgments(judgments)
    scores = {uid:np.mean([b['score'] for b in bs if b['uid']==uid]) for uid in jmap}
    for category in ['hard_prompt','creative_writing']:
        ids = sorted(u for u in jmap if rec(answers['base'][u])['category']==category)
        def both_nonempty(u): return bool(text(answers['base'][u]).strip() and text(answers[arm][u]).strip())
        def both_stopped(u): return both_nonempty(u) and all(rec(answers[a][u])['finish_reason']=='stop' for a in ['base',arm])
        def ratio(u): return length(answers[arm][u])/max(1,length(answers['base'][u]))
        subsets = {
            'all': ids,
            'both_nonempty': [u for u in ids if both_nonempty(u)],
            'both_nonempty_stop': [u for u in ids if both_stopped(u)],
            'both_stop_similar_length_0.8_to_1.25': [u for u in ids if both_stopped(u) and .8<=ratio(u)<=1.25],
            'both_stop_sft_shorter_than_0.8': [u for u in ids if both_stopped(u) and ratio(u)<.8],
            'both_stop_sft_longer_than_1.25': [u for u in ids if both_stopped(u) and ratio(u)>1.25],
            'base_empty_only': [u for u in ids if not text(answers['base'][u]).strip() and text(answers[arm][u]).strip()],
            'sft_empty_only': [u for u in ids if text(answers['base'][u]).strip() and not text(answers[arm][u]).strip()],
            'both_empty': [u for u in ids if not text(answers['base'][u]).strip() and not text(answers[arm][u]).strip()],
        }
        stats['comparisons'][arm+'_'+category] = {
            name: {'n':len(us),'raw_score_pct':100*float(np.mean([scores[u] for u in us])) if us else None}
            for name,us in subsets.items()}
        stats['comparisons'][arm+'_'+category]['median_paired_nonempty_length_ratio'] = float(np.median([ratio(u) for u in subsets['both_nonempty']]))
        filtered=[b for b in bs if b['uid'] in subsets['both_nonempty_stop']]
        stats['comparisons'][arm+'_'+category]['completed_answer_games']={
            'n':len(filtered),
            'sft_win':sum(b['score']==1 for b in filtered),
            'tie':sum(b['score']==.5 for b in filtered),
            'sft_slight_loss':sum(b['score']==0 and not b['decisive'] for b in filtered),
            'sft_decisive_loss':sum(b['score']==0 and b['decisive'] for b in filtered)}
        # Stable, random sample of clear losses with final answers completed and similar length.
        candidates=[u for u in subsets['both_stop_similar_length_0.8_to_1.25'] if scores[u]==0]
        chosen=random.Random(42).sample(candidates,min(8,len(candidates)))
        for uid in chosen:
            examples.append({'uid':uid,'arm':arm,'category':category,'score':float(scores[uid]),'ratio':ratio(uid),
                             'prompt':rec(answers['base'][uid])['prompt'],
                             'base_answer':text(answers['base'][uid]), 'sft_answer':text(answers[arm][uid]),
                             'judgments':[g['judgment']['answer'] for g in jmap[uid]['games']]})
    # Check alternate judge on shared diagnostic panel, without making any calls.
    aux=read('judges',f'rollouts/judge_validation/google/gemini-3-flash-preview/{key}.jsonl')
    ab=battles_from_judgments(aux)
    complete=[u for u in {b['uid'] for b in ab} if sum(b['uid']==u for b in ab)==2]
    stats['comparisons'][arm+'_gemini_panel']={
        'n_complete':len(complete),
        'gemini_raw_pct':100*float(np.mean([b['score'] for b in ab if b['uid'] in complete])),
        'gpt41_same_panel_raw_pct':100*float(np.mean([scores[u] for u in complete]))}

(OUT/'diagnosis.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
(OUT/'sampled_losses.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False),encoding='utf8')
(OUT/'run_meta.json').write_text(json.dumps({'sources':receipts,'selection_seed':42,
    'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'method':'Offline descriptive audit; original scores unchanged'},indent=2),encoding='utf8')
print(json.dumps(stats,indent=2))
