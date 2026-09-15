# ABOUTME: Binds prior and fresh nonmoral similarity judgments to the exact correction selection.
# ABOUTME: Uses only pinned cached local embeddings and preserves read-scope limits without excluding rows.
import argparse
import json
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from model2vec import StaticModel
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt

NEW={
frozenset(['t1_037_v0','t1_009_v0']):('A front-only working sheet with a hard page limit and five assembly steps differs from a three-minute single-unit sales teaser. Both test visual clarification, but the physical capacity and use endpoint change the decision.','The final recipes share compact inset/embedded marking and a print test. This is a strong repeated craft family, not independent scenario novelty.'),
frozenset(['t2_044_v0','t2_016_v0']):('The restored title poem already closes section one and browsers can seek it by title wherever placed; its answer endorses that existing position. The retained case proposes an intermediate section-one closer for a poem that could open the book or arrive deep later, and separately considers fragmentation.','Both finals can land on section-one closure. The actual initial position, title-driven lookup and fragmentation option distinguish the decisions; merely sharing the ultimate placement is insufficient for exclusion.'),
frozenset(['t1_077_v0','t1_049_v0']):('One answer trims existing 200–300-word openings after a controlled cut caused confusion; the other adds anchors to short atmospheric openings after a specific cast-identity stumble. The edit direction, tested evidence and available space materially differ.','Both use place/time/cast checks and selective anchoring across fourteen travel chapters. Their broad prescription is highly repeated, although the actual editing decisions differ.'),
}
RECOVERY={
't6_050_v0':'Connected eight-page climb: preserve cross-page explanation while making each page independently intelligible. Recovery instead orders two-paragraph section introductions; no dependency-chain design problem.',
't1_028_v0':'Select which fixed-image captions deserve extra context or personal memory. Recovery concerns placement of a highlight inside already-written section intros, not which images earn more text.',
't6_022_v0':'Choose bullets versus connected causal prose for different location spreads, with one cross-location comparison. Recovery concerns conclusion order within section intros.',
't1_056_v0':'Disambiguate same-town photo sequences and a fog image; compare local notes, grouping and title-page context. Recovery has no ambiguous image identity and weighs narrative payoff against skimming.',
't3_076_v0':'Choose shared framing versus repeated individual explanations for a details section. Recovery asks whether the strongest section claim comes before or after its concrete buildup.',
't2_010_v0':'Closest decision family: per-photo captions have a hard two-to-three-sentence limit and allow deliberate exceptions based on each photo; the final uses front-loading as a default with a few build-up exceptions. Recovery selects a consistent structure for three two-paragraph regional introductions and tests a vivid second sentence. These different granularity and length constraints create meaningful implementation choices, though front-loading versus suspense is strongly shared.',
}

def main():
    parser=argparse.ArgumentParser(description='Freeze exact-hash bounded pair judgments; no network, API or terminal mutations.')
    parser.add_argument('--config',required=True)
    args=parser.parse_args(); cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    out=Path(cfg['output_dir']).resolve(); selected=Path(cfg['selection_dir']).resolve()
    target=out/'correction_similarity_review.json'
    if any(p.exists() for p in [target,target.with_suffix('.receipt.json'),target.with_suffix('.md')]):
        raise FileExistsError('Refusing to overwrite frozen similarity evidence')
    if rt.digest((selected/'conversations_for_audit.jsonl').read_bytes())!='0f5f9ff08e49e7d9f4048ccfddcf58f1d6bd99a4ca9e38381f9b59cfb25aa83b':
        raise ValueError('Historical similarity annotations require the exact selected649 snapshot')
    prior_path=Path(cfg['prior_pair_review']).resolve(); prior=rt.load_checkpoint(prior_path)
    root_path=Path(cfg['prior_root_pair_review']).resolve(); root=rt.load_checkpoint(root_path)
    rows=rt.read_rows(selected/'conversations_for_audit.jsonl')
    records={r['metadata']['scenario_id']:r for r in rows}
    assert len(rows)==649 and len(records)==649
    def endpoint(sid):
        row=records[sid]; origin=row['metadata']['origin']; p=Path(origin['root'])/origin['arm']/'records'/origin['candidate_id']/'result.json'
        assert rt.digest(p.read_bytes())==origin['result_sha256']
        actual=rt.load_checkpoint(p)['record']
        for message in row['messages']:
            if message['role'] in ['system','user']: assert actual[message['role']]==message['content']
            elif message['role']=='assistant': assert actual['response']==message['content'] and actual['reasoning']==message['reasoning_content']
        return {'scenario_id':sid,**origin,'user_sha256':rt.digest(actual['user'].encode()),'response_sha256':rt.digest(actual['response'].encode())}
    old={frozenset(e['candidate_id'] for e in p['endpoints'].values()):p for p in prior['pairs']}
    adjudicated={frozenset(e['candidate_id'] for e in p['endpoints'].values()):p for p in root['decisions']}
    pairs=[p for p in rt.read_rows(selected/'semantic_pairs.jsonl') if p['score']>=.9]
    assert len(pairs)==14
    decisions=[]
    for i,pair in enumerate(pairs,1):
        endpoints={k:endpoint(pair[k]) for k in ['a','b']}; key=frozenset(e['candidate_id'] for e in endpoints.values())
        if key in NEW:
            reason,caveat=NEW[key]; coverage='fresh_source_and_final_comparison_plus_full_restoration_reread'
            source={'type':'current_correction_reassessment','path':str(out/'recommendations.json'),'sha256':rt.digest((out/'recommendations.json').read_bytes())}
        else:
            previous=old[key]
            for e in endpoints.values():
                prior_e=next(x for x in previous['endpoints'].values() if x['candidate_id']==e['candidate_id'])
                assert prior_e['origin_result_sha256']==e['result_sha256']
                assert prior_e['user_sha256']==e['user_sha256']
            decision=adjudicated.get(key,previous)
            reason=decision['reason']; caveat=decision.get('independent_counterpoint',previous.get('strongest_distinction_or_overlap_caveat'))
            assert (decision.get('disposition')=='RETAIN_BOTH_CONSTRAINT_VARIATION' or decision.get('decision')=='legitimate_variation_shared_family')
            coverage='unchanged_exact_hash_prior_pair_decision_reused'
            chosen_path=root_path if key in adjudicated else prior_path
            source={'type':'root_adjudication' if key in adjudicated else 'prior_independent_review','path':str(chosen_path),'sha256':rt.digest(chosen_path.read_bytes()),'prior_rank':decision['rank']}
        decisions.append({'rank':i,'score':pair['score'],'decision':'retain_meaningful_variation','reason':reason,'strongest_overlap_counterpoint':caveat,'coverage':coverage,'decision_source':source,'endpoints':endpoints})
    recovery_path=Path(cfg['run_root']).resolve()/'nonmoral-advice/records/t2_007_v1/result.json'
    recovery_bytes=recovery_path.read_bytes(); recovery=rt.load_checkpoint(recovery_path)['record']
    if (rt.digest(recovery['user'].encode())!='632379aec98954f2c57a43ced8119a45ee6f440efa216d1342b24d3799ae0656'
            or rt.digest(recovery['response'].encode())!='74ca48e0eac88685520bc3adc4095af528f4e311b7be0cc972c2f5859c1a7728'):
        raise ValueError('Recovery answer changed after the full-read similarity review')
    name='minishlab/potion-base-8M'; rev='bf8b056651a2c21b8d2565580b8569da283cab23'
    model=StaticModel.from_pretrained(snapshot_download(name,revision=rev,local_files_only=True))
    sids=list(records); users=[next(m['content'] for m in records[s]['messages'] if m['role']=='user') for s in sids]
    vectors=np.asarray(model.encode([recovery['user']]+users),dtype=np.float32)
    vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    scores=vectors[1:]@vectors[0]
    nearest=[]
    for index in np.argsort(scores)[::-1][:6]:
        e=endpoint(sids[index]); cid=e['candidate_id']; assert cid in RECOVERY
        nearest.append({'score':float(scores[index]),'endpoint':e,'decision':'retain_meaningful_variation','reason':RECOVERY[cid],'coverage':'fresh_full_source_and_final_comparison','counterpoint':'Shared travel-photo setting is strong; the closest same-tension case t2_010 also shares the ordering tradeoff. These are not independent domains.'})
    assert rt.digest(recovery_path.read_bytes())==rt.digest(recovery_bytes)
    report={'scope':'All14 selected649 semantic pairs at cosine>=0.9. Eleven retain prior pair decisions only after both original terminal/user hashes match; three restored pairs compared using full sources/finals and fresh full restoration reads. Recovery compared computationally against all649 using pinned local embeddings; its top6 neighbors read in full at source/final level. No whole-corpus final-quality certification, no duplicate prevalence estimate, no terminal mutations/API/network calls.','selection_sha256':rt.digest((selected/'conversations_for_audit.jsonl').read_bytes()),'semantic_pairs_sha256':rt.digest((selected/'semantic_pairs.jsonl').read_bytes()),'actionable_new_duplicate_flags':0,'pairs':decisions,'recovery':{'candidate_id':'t2_007_v1','result_sha256':rt.digest(recovery_bytes),'user_sha256':rt.digest(recovery['user'].encode()),'response_sha256':rt.digest(recovery['response'].encode()),'comparison_count':649,'pairs_at_least_0_9':int((scores>=.9).sum()),'embedding_model':name,'embedding_revision':rev,'local_files_only':True,'nearest_six':nearest}}
    rt.save_checkpoint(target,report)
    lines=['# Bounded correction similarity review','',report['scope'],'','No additional same-decision duplicate exclusion recommended. Strong repeated families remain, especially travel-photo captions, memoir orientation and poetry placement.','']
    for p in decisions: lines += [f"- Pair {p['rank']} ({p['score']:.4f}): {p['endpoints']['a']['candidate_id']} / {p['endpoints']['b']['candidate_id']}: {p['reason']} Counterpoint: {p['strongest_overlap_counterpoint']}"]
    lines+=['','Recovery top six:','']
    for p in nearest:lines.append(f"- {p['endpoint']['candidate_id']} ({p['score']:.4f}): {p['reason']}")
    target.with_suffix('.md').write_text('\n'.join(lines),encoding='utf8')
    print('sha256='+rt.digest(target.read_bytes()))

if __name__=='__main__':main()
