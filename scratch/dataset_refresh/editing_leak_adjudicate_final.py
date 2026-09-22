# ABOUTME: Records contextual adjudication of all 27 final editing-language flags.
# ABOUTME: Distinguishes human draft references and visible system-instruction reflection from author-process leakage.
import json
from pathlib import Path
import run as runtime

base=Path(__file__).resolve().parents[2]
quality=base/'output/2026-09-15_dataset_refresh_quality_screen'
source=quality/'editing_leak_final_snapshot.json'
report=json.loads(source.read_text(encoding='utf-8'))
reasons=[
 'The human is considering revising the third story to add a lighthouse-fire clue; draft and revision refer to that story.',
 'The human explicitly gives the original reasoning for omitting player count and duration; answer discusses that reasoning.',
 'Instruction to the human to draft condition lines for their fountain-pen import guide, not a description of an earlier assistant answer.',
 'Instruction to the human to draft alternative Mostar memoir openings; user explicitly weighs those alternatives.',
 'Proposed condition-classification rule is described as a tentative draft to test, not an earlier hidden response.',
 'Instruction to the human to draft a preset documentation note and assess whether its causal explanation fits.',
 'Instruction to the human to draft fuller family-photo captions while preserving the established album voice.',
 'The answer proposes drafting a neighborhood-newsletter section; that human draft may answer the format question.',
 'Instruction to the human to draft the lake-house tagging entry, whose causal naming history the user supplied.',
 'The user writes a recurring reasoning section in a woodworking newsletter; draft the reasoning refers to that named section.',
 'The user explicitly wants story notes about abandoned drafts and solved structural problems; these are the story drafts.',
 'The user supplies a status-reference-card draft allocation; the answer calls that human plan sound in shape.',
 'The user explicitly labels the uniform four-hint structure my draft plan; answer discusses that human plan.',
 'The user asks how to word a reply to Priya; the honest draft is a proposed reply, not a prior assistant generation.',
 'The user describes the picnic AI food-list draft they have read and may circulate; answer discusses that draft and possible replies.',
 'The user explicitly considers publishing a group picnic suggestion as draft rather than final; answer retains that status distinction.',
 'The user considers calling the selected zine-cover image this issue\'s AI draft; answer quotes that proposed human wording.',
 'Reasoning explicitly discusses the actual training-visible system instruction to keep replies brief/upbeat/practical. This is instruction reflection, not hidden author/revision-stage leakage. Flag separately if release policy forbids all visible-system references.',
 'Reasoning explicitly discusses the actual training-visible system instruction for short/upbeat/action-oriented replies and minimal caveats; no absent hidden prompt is introduced.',
 'Reasoning explicitly discusses the actual training-visible system instruction for brief/upbeat/practical advice; no draft-stage or hidden preference is referenced.',
 'Reasoning explicitly discusses the actual training-visible system instruction for short/punchy/minimally hedged messages; this reflects the intended style-vs-substance tension.',
 'The user asks to change a scoreboard draft-review-final workflow and requests announcement wording; the quotation is that announcement.',
 'Original reasoning refers to the user\'s explicit rationale that neighbors will not complain after a unilateral shelf change.',
 'The human supplies a toast draft and considers outsourcing its revision to an AI tool; answer discusses that exact proposed workflow.',
 'The human asks for reasoning to form their own first draft of a potluck layout, then have a friend review it. Both matches refer to this learning workflow.',
 'The user describes a tool\'s first draft of the craft-zine layout; answer acknowledges that it may be adequate despite the process/learning issue.',
 'The user states their own reasoning for drafting a craft-swap layout before asking AI for feedback; answer evaluates that reasoning.',
]
assert len(reasons)==len(report['rows'])==27
rows=[]
for index,(row,reason) in enumerate(zip(report['rows'],reasons)):
    path=Path(row['result_path'])
    assert runtime.digest(path.read_bytes())==row['result_sha256']
    category='visible_system_instruction_reflection' if index in [17,18,19,20] else 'human_task_or_draft_reference'
    rows.append(dict(candidate_id=row['candidate_id'],run_root=row['run_root'],arm=row['arm'],result_path=str(path),result_sha256=row['result_sha256'],category=category,hidden_generation_or_editing_leak=False,reason=reason,hits=row['hits'],reviewed_fields=['full user','all flagged rationale/final passages in context']+(['full system'] if category.startswith('visible') else [])))
out=dict(ABOUTME=['Contextual adjudication of final accepted editing-language screen.', 'This is targeted process-leak QA, not a full content certification of every screened conversation.'],source=str(source),source_sha256=runtime.digest(source.read_bytes()),effective_accepted_counts=report['effective_accepted_counts'],scope='All effective accepted terminals under qualified and diverse roots at snapshot; all27 flagged cases have full user and flagged reasoning/final passages read. Four visible-system matches additionally checked against actual full system. No terminal mutations in this scan.',counts=dict(flagged_rows=27,field_hits=30,confirmed_hidden_generation_or_editing_leaks=0,human_task_or_draft_reference=23,visible_system_instruction_reflection=4),rows=rows,known_positive_fixture='Qualified nonmoral t7_042_v0, excluded immediately before scan, explicitly says draft surfaced information but did not carry it into actual advice; user supplies no prior assistant draft. The positive fixture is outside effective accepted snapshot.')
runtime.save_checkpoint(quality/'editing_leak_final_adjudication.json',out)
lines=['<!-- ABOUTME: Contextual review of all final editing-language flags. -->','<!-- ABOUTME: Distinguishes visible instructions and human drafts from hidden author-stage leaks. -->','',json.dumps(out['counts']),'',out['scope'],'']
for row in rows:
    lines += [f"- {Path(row['run_root']).name}/{row['arm']}/{row['candidate_id']} — **{row['category']}** (`{row['result_sha256']}`): {row['reason']}",'']
(quality/'editing_leak_final_adjudication.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(out['counts']))
