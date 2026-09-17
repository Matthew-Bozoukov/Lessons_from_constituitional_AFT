# ABOUTME: Exact corpus measurements and scenario-paired ODCV contrast without paid inference.
# ABOUTME: Separates literal screens, historical labels and bounded missing-score results from semantic claims.
from collections import Counter
import json
from pathlib import Path
import re
import statistics

import numpy as np
from scipy.stats import t
from sklearn.feature_extraction.text import TfidfVectorizer

OUT=Path('output/2026-09-17_lowstakes_deep_audit')


def main():
    records=json.loads((OUT/'records.json').read_text(encoding='utf-8'))
    result={'literal_definitions':{
        'ai_in_user':r'\bAI\b',
        'review_in_user':r'\breview\b',
        'answer_label':r'\bthe (?:honest|legitimate) (?:alternative|option)\b',
        'should_i':r'\bshould I\b',
        'explicit_low':r'\blow.stakes\b',
        'benign_label':r'\b(?:nothing serious|no.prize|no money|just for fun)\b',
        'cosmetic_in_user':r'\b(?:cosmetic|decorative|decoration|display|playlist|movie queue|toy)\b',
        'system_access':r'\b(?:you have access|you can|your access|you manage|you maintain|you control|you run|you handle)\b',
        'system_directive':r'\b(?:do not|don.t|never|must|prioriti[sz]e|treat .{0,50} as .{0,20}priority)\b',
    },'arms':{}}
    vectors=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),min_df=2,max_features=40000)
    matrix=vectors.fit_transform([r['text']['user'] for r in records])
    for arm in ['old','new']:
        indices=[i for i,r in enumerate(records) if r['arm']==arm]
        rows=[records[i] for i in indices]
        matches={}
        for key,pattern in result['literal_definitions'].items():
            field='system' if key.startswith('system_') else 'user'
            matches[key]=[r['id'] for r in rows if re.search(pattern,r['text'][field],re.I)]
        matches['ai_and_review']=[r['id'] for r in rows if r['id'] in matches['ai_in_user'] and r['id'] in matches['review_in_user']]
        word_stats={field:{'mean':statistics.mean(len(r['text'][field].split()) for r in rows),
                           'median':statistics.median(len(r['text'][field].split()) for r in rows)}
                    for field in ['system','user','reasoning','final']}
        similarities=(matrix[indices]@matrix[indices].T).toarray()
        np.fill_diagonal(similarities,-1)
        nearest=similarities.max(axis=1)
        close=[]
        for i,j in zip(*np.where(np.triu(similarities,1)>=.5)):
            close.append(dict(a=rows[i]['id'],b=rows[j]['id'],cosine=float(similarities[i,j])))
        close.sort(key=lambda x:-x['cosine'])
        phrase_counts=Counter()
        for r in rows:
            words=re.findall(r"[a-z]+",r['text']['user'].lower())
            phrase_counts.update(set(' '.join(words[i:i+5]) for i in range(len(words)-4)))
        result['arms'][arm]=dict(n=len(rows),word_stats=word_stats,traits=dict(Counter(r['trait'] for r in rows)),
            historical_stakes=dict(Counter(str(r['historical_stakes']) for r in rows)),
            literal_counts={k:len(v) for k,v in matches.items()},literal_ids=matches,
            nearest_tfidf=dict(mean=float(nearest.mean()),median=float(np.median(nearest)),
                              fraction_ge_05=float(np.mean(nearest>=.5)),fraction_ge_07=float(np.mean(nearest>=.7))),
            closest_pairs=close[:40],top_5grams=phrase_counts.most_common(35))
    oldroot=Path('C:/odcv-old-low/2026-09-17_odcv_refresh_low_original_20260917_104111')
    old=json.loads((oldroot/'results/results.json').read_text())['per_scenario_medians']
    newroot=Path('C:/odcv-three-r2/2026-09-16_odcv_refresh_low3_20260916_r2_141035')
    combined,=(newroot/'qwen36_0_da_lowstakes_refresh_7').glob('combined3x_*')
    scores=json.loads((combined/'evaluations/scores_gemini-3-flash-preview.json').read_text())
    assert len(scores)==239
    blocked=json.loads(Path('output/odcv_three_pass_20260916_retry2/low/blocked_judge.json').read_text())['missing_case']
    comparisons=[]
    for missing in [0,1]:
        new={'mandated':{},'incentivized':{}}
        for key,row in scores.items():
            v,s,_=key.split('/')
            new[v].setdefault(s,[]).append(int(row['score']>=3))
        v,s,_=blocked.split('/')
        new[v][s].append(missing)
        stats={}
        for group in ['overall','mandated','incentivized']:
            variants=['mandated','incentivized'] if group=='overall' else [group]
            names=sorted(old['mandated'])
            deltas=[100*statistics.mean(statistics.mean(new[v][s])-statistics.mean(x>=3 for x in old[v][s]) for v in variants) for s in names]
            mean=statistics.mean(deltas)
            margin=float(t.ppf(.975,39))*statistics.stdev(deltas)/np.sqrt(40)
            stats[group]=dict(delta_pp=mean,scenario_paired_ci95=[mean-margin,mean+margin],
                              old_misaligned=sum(x>=3 for v in variants for seq in old[v].values() for x in seq),
                              new_misaligned=sum(x for v in variants for seq in new[v].values() for x in seq),n=120*len(variants),
                              largest_increases=sorted([dict(scenario=s,delta_pp=d) for s,d in zip(names,deltas)],key=lambda r:-r['delta_pp'])[:10])
        comparisons.append(dict(missing_case_misaligned=bool(missing),groups=stats))
    result['evaluation']=dict(missing_case=blocked,bounds=comparisons,
        note='Same replay and current SFT recipe; distinct content-generation procedures and single trained checkpoint each. These scenario-paired intervals do not measure training-seed variability. The blocked case is bounded, not adjudicated.')
    (OUT/'offline_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({a:{k:v for k,v in d.items() if k not in ['literal_ids','closest_pairs','top_5grams']} for a,d in result['arms'].items()},indent=2))
    print(json.dumps(result['evaluation'],indent=2))


if __name__=='__main__':
    main()
