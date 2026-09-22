# ABOUTME: Compare pinned DA and historical nonmoral with exact refreshed pools using local cached data only.
# ABOUTME: Export domain labels, explicit analyst family mappings and current native-token differences without selection.
from pathlib import Path
from collections import Counter
import argparse
import json
import re
import statistics
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.validate_mixtures import token_audit, TOKENIZER
from src.model_profile import model_profile

DA = ('dougalldeepmind/2026-09-14-da-synth', '013886238fca238c4d54ace96530f444bb2b2f02', 'dataset.jsonl')
OLD = ('dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation', 'fed726d2db33bddb698ca349a6d76a4e7df7a7e9', 'dataset.jsonl')
MIX = ('dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture', '6364505df02b0020b030bf379bd42285a14de6a5', 't2_9284_nonmoral_684.jsonl')
# These analyst groupings cover current assigned labels, not inferred true subject matter.
FAMILIES = {
 'physical craft instructions': ['Harmless paper-folding', 'Toy-brick'],
 'writing and publishing': ['Short-story', 'Poetry manuscript', 'Personal travel memoir', 'An informal newsletter', 'Community creative zine', 'Local-history scrapbook', 'Creative-project handover'],
 'games and puzzles': ['Board-game', 'A puzzle hint', 'An imaginary exploration'],
 'music and listening': ['Personal listening', 'Amateur rehearsal', 'Amateur musical'],
 'visual art and catalogues': ['Travel photo album', 'Photo-editing', 'Personal photo archive', 'Personal art portfolio', 'Hobby exhibition', 'Decorative motif', 'A decorative pattern'],
 'notebook indexing': ['Personal notebook'],
 'hobby reference and teaching': ['A hobby club terminology', 'Creative hobby workshop'],
 'bounded hobby software documentation': ['Simple drawing-app', 'Offline personal hobby-catalog'],
}
KEYWORDS = {'origami': r'\borigami\b', 'board or tabletop games': r'\b(board.game|tabletop|tile.laying|deck.building)\b',
 'puzzle or escape room': r'\b(puzzle|escape.room)\b', 'motif': r'\bmotifs?\b', 'notebook': r'\bnotebooks?\b',
 'software terms': r'\b(api|software|database|code|cli|programming|sql)\b'}


def stats(values):
    return {'count':len(values), 'mean':statistics.mean(values), 'median':statistics.median(values),
            'min':min(values), 'max':max(values), 'total':sum(values)}


def cached(pin):
    return Path(hf_hub_download(pin[0],pin[2],repo_type='dataset',revision=pin[1],local_files_only=True))


def main(nonmoral_path, output):
    output=Path(output)
    if output.exists(): raise ValueError('New comparison destination required')
    paths={'da752':cached(DA), 'original_nonmoral702':cached(OLD), 'historical_mix9968':cached(MIX),
      'low716':Path('output/2026-09-15_da_lowstakes_refresh_synth_release/dataset.jsonl'), 'current_nonmoral':Path(nonmoral_path)}
    loaded={k:base.read_rows(p) for k,p in paths.items()}
    ids=[r['scenario_id'] for r in loaded['historical_mix9968'] if r['source']=='nonmoral_deliberation']
    originals={r['metadata']['scenario_id']:r for r in loaded['original_nonmoral702']}
    assert len(ids)==len(set(ids))==684 and all(i in originals for i in ids)
    original=[originals[i] for i in ids]
    for trained in loaded['historical_mix9968']:
        if trained['source']=='nonmoral_deliberation':
            source=originals[trained['scenario_id']]
            assert all(m['content'] in trained['text'] for m in source['messages'])
            assert source['messages'][-1]['reasoning_content'] in trained['text']
    corpora={'da752':loaded['da752'],'low716':loaded['low716'],'original_nonmoral684':original,'current_nonmoral':loaded['current_nonmoral']}
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True);profile=model_profile(TOKENIZER)
    output.mkdir(parents=True); summaries={}; per_rows=[]
    for name,rows in corpora.items():
        labels=Counter();families=Counter();family_map={};mechanisms=Counter();chars={k:[] for k in ('system','user','reasoning','response')};words={k:[] for k in chars};diagnostics=[];lex=Counter();literal=Counter()
        for index,row in enumerate(rows):
            meta=row['metadata']; conv={m['role']:m['content'] for m in row['messages'] if m['role']!='assistant'}
            conv.update(reasoning=row['messages'][-1].get('reasoning_content',''),response=row['messages'][-1]['content'])
            label=meta.get('domain','<missing>'); labels[label]+=1
            if name=='current_nonmoral':
                matches=[k for k,prefixes in FAMILIES.items() if any(label.startswith(p) for p in prefixes)]
                if len(matches)!=1: raise ValueError('Unmapped/ambiguous current domain: '+label)
                family=matches[0];family_map[label]=family
            elif name=='low716':
                canonical=re.sub(r'^MECHANISM SLOT [ABC]\. BENIGN ACTIVITY: ', '',label)
                family=canonical.split(':',1)[0];family_map[label]=family
                mechanisms[re.match(r'^MECHANISM SLOT ([ABC])\.',label).group(1) if label.startswith('MECHANISM') else 'earlier_recipe']+=1
            else: family='not canonically recorded';family_map[label]=family
            families[family]+=1
            matched=[k for k,pat in KEYWORDS.items() if re.search(pat,conv['user'],re.I)]
            lex.update(matched)
            for k,v in conv.items():chars[k].append(len(v));words[k].append(len(v.split()));literal[k]+=v.count(chr(0xfffd))
            diagnostic=token_audit(row,tokenizer,profile,8192);diagnostics.append(diagnostic)
            per_rows.append({'corpus':name,'index':index,'scenario_id':meta['scenario_id'],'trait_id':meta.get('trait_id'),
                'domain':label,'analyst_family':family,'user_keyword_matches':matched,'conversation_sha256':base.digest(conv),
                'characters':{k:len(v) for k,v in conv.items()},'words':{k:len(v.split()) for k,v in conv.items()},'tokens':diagnostic})
        summaries[name]={'rows':len(rows),'traits':dict(Counter(r['metadata'].get('trait_id') for r in rows)),
            'exact_domain_labels':dict(sorted(labels.items())),'distinct_domain_labels':len(labels),'family_counts':dict(families),
            'domain_family_mapping':family_map,'mechanism_slots':dict(mechanisms),'user_keyword_occurrences':dict(lex),
            'replacement_characters_by_field':dict(literal),'characters':{k:stats(v) for k,v in chars.items()},
            'whitespace_words':{k:stats(v) for k,v in words.items()},
            'native_tokens':{k:stats([d[k] for d in diagnostics]) for k in ['training_tokens','supervised_tokens','raw_reasoning_tokens','raw_assistant_content_tokens']}}
        print(name,len(rows),'mean supervised',summaries[name]['native_tokens']['supervised_tokens']['mean'],flush=True)
    refs={k:{'path':str(p.resolve()),'sha256':base.digest(p.read_bytes())} for k,p in paths.items()}
    da=summaries['da752']['native_tokens']['supervised_tokens']['mean']
    report={'scope':'Descriptive census; current nonmoral is provisional unless exact final716 publication preview supplied. No matched-case or causal equivalence claim.',
      'input_files':refs,'pins':{'DA':DA,'original_nonmoral':OLD,'original_training_mix':MIX},
      'historical684_join':'Exact selected IDs, all source message fields and reasoning verified verbatim inside old serialized training text.',
      'tokenizer':TOKENIZER,'tokenizer_sha256':base.digest(tokenizer.backend_tokenizer.to_str().encode()),'max_train_tokens':8192,
      'token_interpretation':'Same current native Qwen template/masking for all corpora, not a reconstruction of historical training loss masks.',
      'family_interpretation':'Low: exact recorded activity after removing mechanism slot prefix. Current nonmoral: explicit analyst mapping of28 assigned labels. DA and original nonmoral lack canonical observed family IDs; do not claim exact true-domain family totals.',
      'keyword_interpretation':'Deterministic nonexclusive case-insensitive user-text matches, not semantic domain labels or defect estimates.',
      'keyword_rules':KEYWORDS,'family_rules':FAMILIES,'corpora':summaries,
      'mean_supervised_token_ratio_to_DA':{k:v['native_tokens']['supervised_tokens']['mean']/da for k,v in summaries.items()},
      'helper_sha256':base.digest(Path(__file__).read_bytes())}
    base.write_json(output/'comparison.json',report);base.write_rows(output/'row_census.jsonl',per_rows)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nonmoral',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    main(a.nonmoral,a.output)
