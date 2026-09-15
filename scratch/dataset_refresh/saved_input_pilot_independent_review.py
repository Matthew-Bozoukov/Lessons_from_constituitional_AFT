# ABOUTME: Saves independent Codex-agent full-source and full-answer judgments of four bounded Sonnet repairs.
# ABOUTME: Binds the actual review contract, source, request and result bytes without adopting any row.
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt

ANNOTATIONS={
't6_012_v0':{
 'decision':'accept',
 'source_first_reason':'The source distinguishes genuinely linked house-sale and replacement-trip events from unrelated monthly happenings, with working tags and expensive future note rewrites. The new answer preserves this distinction, accepts either connected prose or explicitly connected bullets, and locates the cross-reference in the actual replacement local-trip album rather than inventing an album for a trip never taken.',
 'source_quotes':['a smaller local trip instead that shows up as a separate tagged album','redoing hundreds of these notes later would be a lot of rework'],
 'answer_quotes':['not a limitation specific to prose or to bullets','since the lake trip itself was postponed rather than taken','whether the arrow-bullet approach holds up as scannable'],
 'repair_target_resolution':'The nonexistent lake-trip album is removed. The cross-reference is now explicitly a consequence of separate album notes, equally possible in prose and bullets. The recommendation retains the mixed-format option and a small trial before extensive rewriting.',
 'benign_counterreading':'That stays as scannable as your current fragments is too absolute in isolation, but the final paragraph explicitly makes that very property a question for the small trial. The actionable plan does not rely on equal scan speed being guaranteed. Likewise, the criticism of forced prose concerns these expressly unrelated events, not every possible prose note.',
 'remaining_caveats':['The equal-scannability phrase is stronger than the available evidence; the later trial instruction correctly leaves scan performance unresolved.','The remove-one-event heuristic is useful shorthand for this causal-chain example, not a complete theory of all meaningful relationships between events.','The proposed arrow chain slightly compresses the house chronology. Treat it as illustrative note wording to check, not an independently verified account of property transactions.'],
 'craft_reason':'Chooses an invariant rule based on actual relationships rather than invariant typography. Offers prose, connected bullets and bare bullets where each serves the supplied content, with cross-album retrieval treated separately. No arbitrary universal compromise is required.'},
't1_004_v0':{
 'decision':'accept',
 'source_first_reason':'The source establishes a real two-strut ordering error, with delayed discovery reported by a few testers, and asks about diagram, caption and split-card options under a fixed small-square constraint. The new answer treats cards as sequential presentation rather than physical enforcement, preserves the fixed-space cost, and recommends a concrete split-plus-caption trial without claiming guaranteed attention or success.',
 'source_quotes':['a few said they didn\'t realize anything was wrong until several steps later','Each card is a fixed small square format'],
 'answer_quotes':['the card order is a presentation sequence, not a mechanical gate','check that against your actual layout before committing','That\'s evidence, not proof'],
 'repair_target_resolution':'Both blocks remove physical-prevention language. Retesting is evidence, visually similar parts remain potentially addressable through clearer instructions, and the original bounded visual-versus-text tradeoff remains substantial.',
 'benign_counterreading':'The opening generalizes the few delayed-discovery reports into a broad description of how testers fail. Read as selecting the observed failure mode that motivates the intervention, it does not assert a measured prevalence or require every tester to fail that way. The recommendation needs only the explicitly observed failure, not a majority.',
 'remaining_caveats':['The source quantifier a few is not retained in the opening summary; do not cite this answer as evidence that all testers discover the error late.','Retesting familiar testers may reflect prior learning as well as the new format. The answer explicitly calls it limited evidence rather than proof; no controlled causal claim is made.'],
 'craft_reason':'Weighs omission of ordering information against layout/attention costs, retains a real case for each option, and chooses a specific proportionate exception. The synthesis follows this case rather than a requirement that every answer compromise.'},
't3_001_v0':{
 'decision':'accept',
 'source_first_reason':'The source supplies three model-specific drawings that worked in a small neighbor-kid trial, an untested general diagram, page count and maintenance costs, supervision and possible take-home use. The answer keeps the self-contained model instructions, adds a named transfer cue, and now tests that cue only after actually exposing the child to it.',
 'source_quotes':['That\'s worked fine so far in my test folds with a couple neighbor kids.','the booklet\'s a few pages longer'],
 'answer_quotes':['You haven\'t said how many kids folded at once','a real, ongoing cost of the three-page approach','crane first, so they actually see the bridge sentence','one trial won\'t settle it definitively'],
 'repair_target_resolution':'The prior missing-intervention test is fixed: crane cue first, then later model. The answer no longer invents one-on-one trial conditions and expressly retains the burden of maintaining three drawings.',
 'benign_counterreading':'The phrase without reopening the three-separate-arrows risk is best read as saying the added sentence does not create an additional drawing change; both blocks explicitly preserve the existing three-copy maintenance burden. It is not a claim that separate drawings become risk-free.',
 'remaining_caveats':['A line about weighing maintenance against page-count savings and clarity gains is awkward: retaining three drawings does not itself save pages. The surrounding paragraphs clearly state that extra pages are a real cost, so this does not drive the recommendation.','The small sequential test cannot isolate naming-cue causality from practice or adult help; the answer presents it as one observation, not proof of transfer.'],
 'craft_reason':'Separates executing a specific model from transferring a named move, credits the tested concrete instances without treating them as proof, and pays a small explicit cost for the cross-model bridge. Keeping the initial self-contained plan is a legitimate resolution.'},
't5_007_v0':{
 'decision':'accept',
 'source_first_reason':'The actual four-field order and trained fourth-line lookup are preserved. Watch For is consistently the optional fifth field after Use Case in both blocks, with a real added-check cost. The answer limits supplied clipping behavior to the named presets and does not infer most of the forty-plus presets are simple.',
 'source_quotes':['I just look at line 4 for "use case" across all of them.','presets with known technical gotchas get an extra "Watch For" section'],
 'answer_quotes':['placed *after* Use Case','You\'ve only described three presets here','not as general Lightroom mechanics','not something I can promise in advance'],
 'repair_target_resolution':'Corrects the field-position contradiction, avoids a fabricated library-wide proportion, treats undesirable rendered results rather than irreversible source-file destruction, and proposes trying the two known exceptions before a larger rewrite.',
 'benign_counterreading':'The categorical statement that no tool parses the notes is stronger than the user explicitly stated. But the user presents personal plain-markdown reference notes and expressly proposes adding a section; there is no supplied machine interface being overridden. The concrete fifth-field recommendation is already within that authorized design space and does not need an invented parser contract.',
 'remaining_caveats':['Do not treat the statement about no external parser as a verified software fact. It is a source-purpose inference whose absolute phrasing is unnecessary.','Nothing about your existing habit changes refers to the preserved fourth-line location, not literally zero new attention cost; the next sentences explicitly acknowledge checking line five.'],
 'craft_reason':'Preserves the part of the convention that actually supports retrieval while introducing a narrow optional field for materially different content. Addresses the real learned navigation cost, conditional information and unknown future frequency without a forced universal format.'},
}

EXPECTED_RESULTS={
 't1_004_v0':'9bcf21cba856b1c3a61d17436dbff0c285c7b77e94ab1e0000a115e620bac50e',
 't3_001_v0':'d23f4eae01ffed37f55f540ca003d8a002075442c4e587008b9393c469efb81e',
 't5_007_v0':'2c06ee21f7fb92b40f33c1552818ff53244215f560292fb8d3a45fb5cd4cfae4',
 't6_012_v0':'030dcc0837ebbdf0a699c80beae0e6ea6c44ddcf57fe0e2fbe09f63d8bd58b6f',
}

def finalize(pilot, out, cfg):
    target=out/'final_independent_review.json'
    if any(p.exists() for p in [target,target.with_suffix('.receipt.json'),target.with_suffix('.md')]):
        raise FileExistsError('Refusing to overwrite final review')
    reviews=[]
    for cid, expected in EXPECTED_RESULTS.items():
        rp=pilot/(cid+'.result.json'); vp=out/(cid+'.independent_review.json')
        review=rt.load_checkpoint(vp)
        assert rt.digest(rp.read_bytes())==expected==review['result_sha256']
        reviews.append({'candidate_id':cid,'decision':review['decision'],'result_sha256':expected,'review_path':str(vp),'review_sha256':rt.digest(vp.read_bytes())})
    other_path=Path(cfg['nearest_result']).resolve()
    assert rt.digest(other_path.read_bytes())=='0c121e85c6b39ba431e03b0d4891a23eae7b9a4be3e5d1e07da52adc41fa4baf'
    other=rt.load_checkpoint(other_path)['record']
    actual=rt.load_checkpoint(pilot/'t1_004_v0.result.json')['conversation']
    quote_a='testers keep getting the strut order wrong even when they stare at the diagram'
    quote_b='this is the one stage in the whole 40-card set where the motion isn\'t a straight press'
    assert quote_a in actual['user'] and quote_b in other['user']
    pair={
      'decision':'retain_both','confidence':'high','cosine_reported':0.88367748,
      'scope':'Fresh full reads of both actual system/user, final reasoning and response. Similarity triage, not certification of every pre-existing answer.',
      'candidate_a':'t1_004_v0','candidate_b':'t8_074_v0',
      'result_a_sha256':EXPECTED_RESULTS['t1_004_v0'],'result_b_path':str(other_path),'result_b_sha256':rt.digest(other_path.read_bytes()),
      'source_user_a_sha256':rt.digest(actual['user'].encode()),'source_user_b_sha256':rt.digest(other['user'].encode()),
      'reasoning_a_sha256':rt.digest(actual['reasoning'].encode()),'reasoning_b_sha256':rt.digest(other['reasoning'].encode()),
      'response_a_sha256':rt.digest(actual['response'].encode()),'response_b_sha256':rt.digest(other['response'].encode()),
      'source_quotes':[quote_a,quote_b],
      'reason':'The first decision responds to observed two-strut ordering errors and some delayed discovery within a fixed small square, with splitting the sequence into two cards explicitly available. It recommends a split plus caption with real space and sequencing limitations. The second concerns an untested rotation-before-press inference in an otherwise straight-press set; it recommends a rotational-arrow/inset trial and conditional caption, with an explicit scope rule for future exceptions. The differing motion, evidence, constraints and interventions produce materially distinct actionable decisions.',
      'strongest_counterpoint':'Both use a lighthouse, angled supports, parent-child/adult audiences, picture-only card conventions and a narrow text exception. This is a strong repeated scenario family and reduces surface diversity; the retention decision rests on actual distinct decisions, not different IDs or trait labels.',
      'incidental_existing_case_caveat':'The existing t8 source says 45 degrees but proposes quarter-turn wording. Its saved final avoids a numeric angle and does not repeat the erroneous caption; this comparison does not certify that source inconsistency away. The saved final also infers twenty-plus preceding cards without an explicit stage number. Neither supplies the basis for the rotation-versus-press recommendation.'}
    result={
      'scope':'Four fresh full-conversation saved-input repairs and their closest flagged cross-corpus pair only. No API calls, automatic acceptance, origin mutation or adoption by this reviewer.',
      'review_order_note':'Actual sources were read before their new answers. Original repair targets were also known before new answers for t3/t5/t6; this was not blind to those targets. No paid new-answer judge verdict or author changes explanation was used to decide. The first three per-case scope strings use followed by as an assessment description and should not be read as an exact chronological or blinding claim.',
      'review_contract_sha256':'5d1ec43313b10807e50f245c6a342e093f8a324f20889a00803e5276c22860e2',
      'reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},
      'counts':{'accept':4,'hold':0,'uncertain':0},'cases':reviews,'nearest_pair':pair,
      'remaining_limits':'Accept means no material defect identified under the full supplied contract. Per-case nonblocking caveats remain. It is not a guarantee, a whole-corpus review, or a decision to release these rows.'}
    rt.save_checkpoint(target,result)
    target.with_suffix('.md').write_text('# Saved-input pilot independent review\n\nAll four: accept, with recorded nonblocking caveats. No adoption performed.\n\nNearest pair t1_004 / t8_074: retain both. '+pair['reason']+'\n\nCounterpoint: '+pair['strongest_counterpoint']+'\n\nReview order: '+result['review_order_note']+'\n',encoding='utf8')
    print('final_review_sha256='+rt.digest(target.read_bytes()))

def main():
    p=argparse.ArgumentParser(description='Record completed independent Codex-agent reviews; never perform API calls or source adoption.')
    p.add_argument('--config',required=True)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--candidate',choices=list(ANNOTATIONS)); action.add_argument('--finalize',action='store_true')
    args=p.parse_args(); cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    pilot=Path(cfg['pilot_root']).resolve(); out=Path(cfg['review_root']).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.finalize:
        finalize(pilot,out,cfg)
        return
    target=out/(args.candidate+'.independent_review.json')
    if any(p.exists() for p in [target,target.with_suffix('.receipt.json'),target.with_suffix('.md')]): raise FileExistsError('Refusing to overwrite frozen independent review')
    manifest=rt.load_checkpoint(pilot/'manifest.json'); contract_path=pilot/'independent_review_contract.json'
    contract=json.loads(contract_path.read_text(encoding='utf8'))
    assert rt.digest(contract_path.read_bytes())==manifest['review_contract_sha256']=='5d1ec43313b10807e50f245c6a342e093f8a324f20889a00803e5276c22860e2'
    input_path=pilot/(args.candidate+'.input.json'); result_path=pilot/(args.candidate+'.result.json')
    assert rt.digest(result_path.read_bytes())==EXPECTED_RESULTS[args.candidate], 'Historical annotation cannot be reused for changed answer bytes'
    inp=rt.load_checkpoint(input_path); result=rt.load_checkpoint(result_path)
    assert result['status']=='awaiting_independent_full_review'
    actual=result['conversation']; source=inp['actual_conversation']
    assert all(actual[k]==source[k] for k in ['system','user'])
    proposal_path=Path(manifest['proposal_path']); assert rt.digest(proposal_path.read_bytes())==manifest['proposal_sha256']
    proposal=json.loads(proposal_path.read_text(encoding='utf8'))
    origin=next(r for r in proposal['rows'] if r['candidate_id']==args.candidate)
    source_path=Path(origin['root'])/origin['arm']/'records'/args.candidate/'result.json'
    assert rt.digest(source_path.read_bytes())==origin['source_result_sha256']
    annotation=ANNOTATIONS[args.candidate]
    for quote in annotation['source_quotes']: assert quote in source['user'],quote
    for quote in annotation['answer_quotes']: assert any(quote in actual[k] for k in ['reasoning','response']),quote
    review={
        'candidate_id':args.candidate,'decision':annotation['decision'],'accepted':annotation['decision']=='accept',
        'gates':{name:True for name in contract['full_frozen_acceptance']['gates']},
        'scope':'Independent Codex-agent fresh full reads of actual system/user and new reasoning/final; original repair target, complete working craft preference and full09 constitution compatibility also assessed. No paid judge verdict used. This agent review neither changes automatic status nor adopts a row. No new formatting/compromise gates.',
        'reviewer_provenance':{'kind':'independent_codex_agent','task':'/root/audit_nonmoral','human_review':False},
        'input_path':str(input_path),'input_sha256':rt.digest(input_path.read_bytes()),
        'source_result_path':str(source_path),'source_result_sha256':origin['source_result_sha256'],
        'source_system_sha256':rt.digest(source['system'].encode()),'source_user_sha256':rt.digest(source['user'].encode()),
        'result_path':str(result_path),'result_sha256':rt.digest(result_path.read_bytes()),
        'conversation_sha256':rt.digest(actual),'reasoning_sha256':rt.digest(actual['reasoning'].encode()),'response_sha256':rt.digest(actual['response'].encode()),
        'review_contract_sha256':manifest['review_contract_sha256'],'constitution_sha256':contract['constitution_sha256'],
        'full_working_preference_sha256':rt.digest(inp['full_working_preference'].encode()),
        'request_sha256':inp['request_sha256'],'physical_receipt':result['physical_receipt'],
        **annotation,
        'constitution_compatibility':'Reviewed full09 text. No substantive conflict found: ordinary benign craft advice preserves human agency, is helpful without moralizing or refusal, makes no claim to have acted, and does not expose a hidden target or revision process. Honesty/calibration caveats are separately stated above rather than concealed. Principles concerning power/identity are not triggered; no demand to display all nine principles.',
        'nonmoral_eligibility':'The decision concerns documentation layout, learning or retrieval in an ordinary craft/personal workflow. No real health, legal, financial, employment, security, emergency or political decision is requested. Named businesses and children do not alone turn this into excluded-subject advice.',
        'issues':[]}
    rt.save_checkpoint(target,review)
    target.with_suffix('.md').write_text(f"# {args.candidate}: {review['decision']}\n\nResult SHA256: `{review['result_sha256']}`.\n\n{review['source_first_reason']}\n\n{review['repair_target_resolution']}\n\nCounterreading: {review['benign_counterreading']}\n\nCaveats: "+json.dumps(review['remaining_caveats'],ensure_ascii=False)+'\n',encoding='utf8')
    print(args.candidate+' '+review['decision']+' review_sha256='+rt.digest(target.read_bytes()))

if __name__=='__main__':main()
