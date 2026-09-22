# ABOUTME: Records eight full-conversation assessments and a lossless-review extraction opportunity.
# ABOUTME: Validates saved author and reviewer bytes without mutating terminals or making API calls.
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt
from scratch.dataset_refresh.reviewer_probe import validate_verdict

ASSESS={
't2_007_v1':('potential_lossless_zero_call_recovery','The source explicitly predicts mixed skimming and sustained reading; the answer weighs the actual suspense versus missed-takeaway tradeoff, preserves the full detail, and proposes testing a concrete second-sentence refinement. No material content defect found in full reread.','Nothing is lost overstates the already acknowledged small suspense cost, but is plainly about preserved detail.','you suspect a few of your more impatient relatives'),
't6_065_v0':('potential_no_authoring_recovery_missing_reviews','The repaired answer respects the actual differentiated cross-references, explains degree and conditional relationships, and preserves standalone alphabetical definitions beside a connected explainer. It does not falsely declare bullets unable to express those relationships.','Placement where all six terms fall is imprecise because the same answer notes their dispersion; it alternatively permits clear links. Most-built-for-lookup is unquantified, but the actual dual workflow supports the design.','Option A tags'),
't9_054_v0':('potential_no_authoring_recovery_missing_reviews','The source supplies actual repeated Selection/Layer questions, no Shape/Brush followups, and uncertain checkpoint use. The answer preserves those distinctions and recommends only a small explanatory hedge for the unfamiliar checkpoint.','No cheap way to obtain better data is too strong; the partial critic also objects to calling the checkpoint name less self-explanatory. That is a defensible qualitative reason for a small hedge, not fabricated incidence. Its formal review remains incomplete and adverse prose is preserved.','targeted hedge'),
't5_051_v0':('content_defect_keep_failed','The answer says a reader who already finished the entire book reaches references to story9 before earning them, then treats inline note4 referencing story9 as spoiler-safer. That is a concrete ordering error and does not solve the described dependency.','The source itself expresses a confused spoiler worry; useful advice should disentangle it rather than endorse the impossible chronology. A reordered back section might still be useful for other reasons.','before they\'ve earned'),
't8_066_v0':('content_defect_keep_failed','The user explicitly says new players cannot tell a repeat from moving to a new tune. The answer recasts their reported confusion as specifically not tune-to-tune, then dismisses the corresponding labels. This silently removes part of the observed problem.','Keeping rehearsal locators and checking the actual cue are good. The later conditional about whether new players read modulation does not undo the earlier false characterization of their report.','not the tune-to-tune'),
't3_040_v0':('potential_no_authoring_recovery_missing_reviews','The answer uses an explicitly illustrative cross-notebook example to anchor a general relation rule at its harder edge, keeps real notebook identifiers in the final ongoing-entry example, and treats the front matter as revisable after real use. No decisive factual defect found.','Half-page capacity and easier recognition of the mechanical example are plausible qualified judgments, not measured facts. The existing positive grounding verdict is truncated and cannot be adopted as a complete JSON judgment.','illustrative, not a fixed formula'),
't4_071_v0':('potential_no_authoring_recovery_missing_reviews','The source supplies the exhibition purpose, reprint cycle, old umbrella terminology and noncalculating exceptions. The answer differentiates teaching labels from index continuity, separately corrects genuinely inaccurate classifications, and proposes observing cross-reference confusion.','Several label-count/visitor-habit details are overprecise, and index/label divergence has a genuine cost. The answer acknowledges and tests the latter; the source establishes the underlying categories and purpose.','Tabulating Aid'),
't1_038_v0':('uncertain_keep_failed','The answer adds an unsupported immediate-first-use timing pattern and says beginners have nothing else to consult. It also changes the stated recurring why-does-wood-behave-differently question into an adjustment-only criterion. Unclear whether the shortened mechanism would still serve the actual question.','The proposed adjustment is conditional on the user\'s actual practice, not an invented temperature instruction. A compact actionable version may be useful, but this is not a clean no-authoring recovery recommendation.','nothing else to consult'),
}

def main():
    parser=argparse.ArgumentParser(description='Reproduce eight fixed human annotations and validate exact saved evidence; never infer a new verdict.')
    parser.add_argument('--config',required=True,help='YAML with run_root, output_dir and budget_root.')
    args=parser.parse_args()
    config=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    ROOT=Path(config['run_root']).resolve()
    OUT=Path(config['output_dir']).resolve()
    BUDGET=Path(config['budget_root']).resolve()
    target=OUT/'remaining_terminal_assessment.json'
    if target.exists() or target.with_suffix('.receipt.json').exists() or (OUT/'remaining_terminal_assessment.md').exists():
        raise FileExistsError('Refusing to overwrite frozen assessment')
    if rt.digest((OUT/'remaining_terminal_inventory.json').read_bytes())!='5ce5406e2b95505aa52057a95a6638d70023ba56434b4c3482e136e0a09eb54b':
        raise ValueError('Historical full-read annotations require the exact frozen eight-case inventory')
    inventory=rt.load_checkpoint(OUT/'remaining_terminal_inventory.json')
    cfg=json.loads((ROOT/'nonmoral-advice/config.json').read_text(encoding='utf8'))
    ledger=json.loads((BUDGET/'spend.json').read_text(encoding='utf8'))
    rows=[]
    for cid in inventory['selected_ids']:
        frozen=next(r for r in inventory['rows'] if r['candidate_id']==cid)
        p=Path(frozen['result_path']); assert rt.digest(p.read_bytes())==frozen['result_sha256']
        result=rt.load_checkpoint(p); rec=result['record']
        stage='repair_1' if (p.parent/'repair_1.json').exists() else 'revise_responses'
        saved=rt.load_checkpoint(p.parent/(stage+'.json'))
        assert all(rec[k]==saved[k] for k in ['reasoning','response'])
        author=[e for e in ledger if e.get('run_root') and Path(e['run_root']).resolve()==ROOT and e.get('arm')=='nonmoral-advice' and e.get('candidate_id')==cid and e.get('stage')==stage][-1]
        rawp=BUDGET/f"raw_calls/{author['call_id']:06d}.json"
        raw=json.loads(rawp.read_text(encoding='utf8'))
        assert raw['response']['finish_reason']=='stop'
        assert raw['request']['model']=='anthropic/claude-sonnet-5'
        parsed=rt._parse_tagged(raw['response']['content'],('reasoning','response'))
        assert all(parsed[k]==rec[k] for k in ['reasoning','response'])
        disposition,reason,caveat,key=ASSESS[cid]
        quotes=[v for k in ['reasoning','response'] for v in rec[k].split('\n\n') if key.casefold() in v.casefold()]
        rows.append({**frozen,'independent_disposition':disposition,'full_read':True,'reason':reason,'strongest_counterpoint':caveat,'exact_answer_quotes':quotes,'author_provenance':{'stage':stage,'call_id':author['call_id'],'raw_call_sha256':rt.digest(rawp.read_bytes()),'stage_sha256':rt.digest((p.parent/(stage+'.json')).read_bytes()),'model':raw['request']['model'],'finish_reason':'stop','exact_saved_and_physical_author_text_match':True}})
    # Exactly one complete JSON object is extractable; no synthetic closing braces or field edits.
    row=next(r for r in rows if r['candidate_id']=='t2_007_v1')
    p=Path(row['result_path']); rec=rt.load_checkpoint(p)['record']
    raw=json.loads(Path(row['latest_call']['path']).read_text(encoding='utf8'))
    matches=list(re.finditer(r'```json\s*(\{.*?\})\s*```',raw['response']['content'],re.S))
    assert len(matches)==1 and raw['response']['finish_reason']=='stop'
    text=matches[0].group(1); review=json.loads(text); assert rt.acceptance(review,cfg)
    narrow={k:rec[k] for k in ['system','user','reasoning','response']}; narrow['final']=narrow.pop('response')
    grounding=rt.load_checkpoint(p.parent/'grounding_0.json'); validate_verdict(grounding,narrow)
    assert grounding['accepted'] is True and grounding['issues']==[]
    conv={k:rec[k] for k in ['system','user','reasoning','response']}
    f={**rec,'style_guidance':cfg.get('style_guidance',''),'conversation_json':json.dumps(conv,ensure_ascii=False),'record_json':json.dumps(conv,ensure_ascii=False),'metadata_json':'{}','eligibility_json':'{}','constitution':cfg['review_constitution_text']}
    expected=rt.request_options(cfg['models']['review'])
    expected['messages']=[{'role':role,'content':rt.render(cfg['prompts']['review_'+role],f)} for role in ['system','user']]
    assert expected==raw['request']
    row['lossless_review_proof']={'extract_character_span':list(matches[0].span(1)),'exact_json_text_sha256':rt.digest(text.encode()),'parsed_review':review,'grounding_0_sha256':rt.digest((p.parent/'grounding_0.json').read_bytes()),'raw_request_matches_exact_current_answer_and_frozen_review_prompt':True,'source_preflight_still_accepted':True,'effect':'Can recover already-issued complete accepted JSON without changing model verdict or author text; requires explicit archived parser-recovery provenance and final export checks.'}
    report={'scope':'Eight preselected complete answers read in full; inventory covers42 remaining failed/rejected terminals. No authoring, calls, source edits, changed verdicts, adoption or terminal mutations.','inventory_sha256':rt.digest((OUT/'remaining_terminal_inventory.json').read_bytes()),'counts':dict(Counter(r['independent_disposition'] for r in rows)),'potential_unchanged_author_text_cases_in_eight':5,'complete_existing_review_zero_call_candidate_count':1,'remaining_review_incomplete_cases_in_eight':4,'not_certified':'Four plausible no-authoring candidates still lack completed final reviews; they are not zero-call releasable under the frozen automated pipeline merely because this human read was favorable. Thirteen valid automatic content rejections were inventoried but not bypassed or counted as format failures. Eleven other technical-complete candidates remain unread in this bounded sample.','rows':rows}
    rt.save_checkpoint(target,report)
    lines=['# Remaining nonmoral terminal recovery assessment','',report['scope'],'',json.dumps(report['counts'],ensure_ascii=False),'',report['not_certified'],'']
    for r in rows:
        lines.extend([f"## {r['candidate_id']} — {r['independent_disposition']}",'',f"Result SHA256: `{r['result_sha256']}`.",'',r['reason'],'',r['strongest_counterpoint'],'',json.dumps(r.get('lossless_review_proof',r['author_provenance']),ensure_ascii=False),''])
    (OUT/'remaining_terminal_assessment.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    print('sha256='+rt.digest(target.read_bytes()))

if __name__=='__main__':main()
