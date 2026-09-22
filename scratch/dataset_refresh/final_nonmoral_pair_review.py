# ABOUTME: Records independent scenario-decision review of all 14 final partial-pool semantic pairs.
# ABOUTME: Binds both original terminal hashes and preserves counterarguments without applying exclusions.
import json
from collections import Counter
from pathlib import Path
import run as runtime

base = Path(__file__).resolve().parents[2]
folder = base / 'output/2026-09-15_dataset_refresh_quality_screen/final_partial_nonmoral'
input_path = folder / 'conversations_for_audit.jsonl'
expected = 'b2bd5a231455ed95698e8b38d03b44729af2e9bd12d5341b38357701a7195e8e'
assert runtime.digest(input_path.read_bytes()) == expected
corpus = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines()]
by_id = {row['metadata']['scenario_id']: row for row in corpus}
pairs = [json.loads(line) for line in (folder/'semantic_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
pairs = [pair for pair in pairs if pair['score'] >= .9]
assert len(pairs) == 14

# decision, confidence, mechanism, rationale, strongest counterpoint, exact A quote, exact B quote
notes = [
 ('legitimate_variation_shared_family','medium','Puzzle hint architecture',
  'A calls for a diagnostic three-tier ladder at each of six distinct stall points in one signature puzzle. B calls for sequential layering and wrong-branch recovery in two difficult puzzles, while preserving a common four-label spine. Those are different hint architectures, not merely different puzzle names; preserve the earlier same-pair judgment.',
  'Both use a few difficult puzzles versus several easy ones and unequal depth versus visual uniformity, so their broad resolution can still look similar.',
  'a full three-tier hint chain for each of the six sub-steps','wrong early deduction cascades'),
 ('legitimate_variation_shared_family','high','Listening-journal order versus selective retention',
  'A fixes a two-paragraph format and asks which paragraph leads, with friends who read only the first and a no-per-entry-reordering constraint. B edits an existing year and asks which explanatory relisten notes to retain, allowing different detail per entry. The operative choices and audience constraints differ.',
  'Both involve verdict lookup versus reconstructing a changed or nuanced listening judgment.',
  'They only ever read the first paragraph','cut all the relisten notes down to a single sentence'),
 ('legitimate_variation_shared_family','medium','Multi-stall diagnostic ladders versus a difficult final step',
  'A needs matching progressive hints across six separate substeps of one large puzzle. B has three separately paginated puzzles and concentrated failures at a final cipher step plus one spatial junction. It asks whether to extend one ladder rather than build parallel ladders for every stall point.',
  'Both may resolve to uneven progressive hint depth justified by observed solver difficulty; this is a close shared family.',
  'six different sub-steps where someone could get stuck','genuinely nasty final step'),
 ('legitimate_variation_shared_family','high','Repairing mistaken-error readings versus avoiding a spoiler',
  'A has reported readers mistaking deliberate second-person ambiguity for an editing error, creating a positive reason to flag intentional craft despite spoiler risk. B has a biographical spoiler that is not needed for initial comprehension plus a separate dispensable trivia note. A balances corrective orientation against surprise; B separates spoiler placement and cutting trivia.',
  'Both are story notes whose advance reading can spoil a formal device, with cut/vague/post-reading alternatives.',
  'one thought the story had a formatting error','biographical detail'),
 ('near_duplicate_scenario','medium','Before/after placement of spoiler-bearing story notes',
  'Both eleven-story collections use short 2–4-sentence author notes initially before each story and ask whether to defer or vary placement to protect structural surprises while keeping a coherent notes convention. This repeats the specific placement decision, not just the literary domain.',
  'B also asks whether to cut a separate harmless said-count trivia note, while A more explicitly stresses uniform placement and mixed reading schedules. Retain both if that independent cut-versus-keep decision is the intended unit.',
  'whether these notes should come before the story or after it','restructure it as an after-story note instead of a before-story one'),
 ('near_duplicate_scenario','high','Drawing-shortcut exception in place versus remote footnote',
  'Both approximately forty-shortcut drawing references propose one-line uniform entries, then ask whether a small number of context-dependent behaviors need an in-place second line or parenthesis rather than an elsewhere note/footnote. The same lookup failure and specific format exception recur with substituted key combinations.',
  'A is a scrollable Help page and B is a one-page card; B also includes undo-history exclusion and cursor-location behavior. The card can impose a tighter real space constraint, although no measured overflow is supplied.',
  'add a second line or a parenthetical caveat','let just these three entries run two lines'),
 ('near_duplicate_scenario','high','Drawing-app mouse modifiers outside the standard shortcut table',
  'Both drawing-app references describe substantially overlapping Shift/Alt/Space drag modifiers that do not fit standalone menu-command documentation, then propose a distinct action/interaction-based modifier group while retaining ordinary commands conventionally. The core structure choice and examples coincide.',
  'A emphasizes menu-category placement and initial learning before Ctrl-F; B emphasizes action-cell wording, a searchable overlay, and discoverability of a second section. These differences can change presentation details but not the central modifier-group decision.',
  'organize the doc by *interaction phase*','organized around the mouse action first'),
 ('legitimate_variation_shared_family','medium','Troubleshooting demand versus reference completeness under a measured page budget',
  'A allocates explanation using six months of actual help-question concentration around four motifs; the tradeoff is lengthening the booklet or cutting other entries severely. B provides a roughly sixty-page cap, six widely used motif families, and the explicit need for a complete rare-family key. B permits budget arithmetic and minimum key completeness, while A asks whether observed confusion earns extra walkthroughs.',
  'Both are needlework motif references with a small heavily used subset, optional walkthroughs, and concern about neglected rare entries; their high-level unequal-depth recommendation may match.',
  '80% of the actual questions cluster around just four motifs','roughly 60 pages'),
 ('legitimate_variation_shared_family','high','General-rule overfitting versus spoiler/trivia triage',
  'A has prior readers forcing every ending into a generalized essay framework and asks how to retain transferable insight without false universality. B asks whether biographical spoilers should move after a story and whether noninterpretive trivia should be cut. Framework-versus-instance explanation is a different decision from disclosure timing and length.',
  'Both eleven-story collections include optional notes that readers may inspect before the stories despite author intent.',
  'trying to force every ending into that mold','hands them the twist before they start reading'),
 ('legitimate_variation_shared_family','high','Connected causal notes versus spoiler placement',
  'A needs the linked causal history of character-writing decisions across a subset of stories while keeping individual entries findable. B weighs positioning short notes before/after stories to protect surprises. Cross-entry explanatory dependency does not reduce to reveal timing.',
  'Both ask whether selective exceptions to a consistent story-notes format are worthwhile.',
  'the actual chain of reasoning','whether these notes should come before the story or after it'),
 ('near_duplicate_scenario','high','Selective anchors across fourteen travel-memoir chapters',
  'Both six-month travel memoirs with fourteen regional chapters have reader confusion about time/place/companions after jumps and fear uniformly adding orientation will slow or flatten pacing. Both ask for the same chapter-specific criterion for how much anchoring to keep. This repeats the previously excluded orientation scenario cluster.',
  'A begins with long200–300-word context paragraphs and a cutting experiment; B begins with short atmospheric openings and selectively missing anchors, including intentional jarring transitions. Opposite starting drafts can influence the editing step.',
  'which chapters actually need that orientation material and how much','which chapters actually need that upfront orientation and which don\'t'),
 ('near_duplicate_scenario','high','Names and glosses for altered ghazal and sonnet forms',
  'Both poetry endnotes with about two lines per poem use the same ghazal/sonnet examples, deliberately departing from formal rules, and ask whether the technical name or plain gloss should lead while preserving lookup vocabulary for mixed-experience readers. This is the same annotation decision with altered particulars.',
  'A states more exact missing components, while B breaks the refrain halfway through and mentions later resonance. Those distinctions matter to the final accurate definitions but do not change the naming-versus-gloss choice.',
  'whether to use the exact term, the plain description, or some combination','name the term and gloss it in the same breath'),
 ('legitimate_variation_shared_family','medium','Reader-confusion repairs versus depth of available author commentary',
  'A has specific reader confusion in two formally unusual stories and a1200-word total limit: the notes should enable corrective rereading. B has no reader data and richer author-process material for three central stories, without a total section cap. The evidence for depth and the resource constraint differ materially.',
  'Both compare a few long story notes with many short notes, resist padding, and worry unequal treatment looks like favoritism. If only the final unequal-depth action matters, they are close.',
  'roughly 1,200 words total','I don\'t have any actual reader feedback yet'),
 ('legitimate_variation_shared_family','medium','Additional hint tiers versus a fixed forty-line access floor',
  'A retains three tiers for two puzzles and adds finer tiers to one final-step bottleneck within three separately assigned pages. B has a hard forty-line total and explicitly cannot give every puzzle the full ladder; it asks whether to remove depth from two difficult cryptics to provide a second tier for eight less-used puzzles. B forces a minimum-coverage tradeoff absent from A.',
  'Both monthly hint journals cite observed difficulty and oppose uniform ladders to targeted depth; the broad recommendation may still concentrate detail.',
  'keep A and C to a tighter three-tier structure','I don\'t have room to give all 10 puzzles the full three-tier treatment'),
]
assert len(notes) == len(pairs)
results=[]
for rank,(pair,note) in enumerate(zip(pairs,notes),1):
    decision,confidence,mechanism,reason,counterpoint,quote_a,quote_b=note
    assert quote_a in pair['user_a'], (rank,'a')
    assert quote_b in pair['user_b'], (rank,'b')
    endpoints={}
    for side in ['a','b']:
        item=by_id[pair[side]]
        origin=item['metadata']['origin']
        path=Path(origin['root'])/origin['arm']/'records'/origin['candidate_id']/'result.json'
        assert runtime.digest(path.read_bytes())==origin['result_sha256']
        current=runtime.load_result(path)
        assert current['status']=='accepted'
        assert current['record']['user']==pair['user_'+side]
        endpoints[side]=dict(scenario_id=pair[side],candidate_id=origin['candidate_id'],origin_path=str(path),origin_result_sha256=origin['result_sha256'],conversation_sha256=item['metadata']['conversation_sha256'],user_sha256=runtime.digest(pair['user_'+side].encode()),current_hash_matches=True,effective_accepted_at_review=True)
    results.append(dict(rank=rank,cosine_similarity=pair['score'],decision=decision,confidence=confidence,mechanism=mechanism,reason=reason,strongest_distinction_or_overlap_caveat=counterpoint,evidence_quotes=dict(a=quote_a,b=quote_b),endpoints=endpoints,user_a=pair['user_a'],user_b=pair['user_b'],recommended_action='Selection owner decides whether to exclude one member; no automatic exclusion.' if decision=='near_duplicate_scenario' else 'May coexist on scenario-decision grounds; answer quality remains a separate judgment.'))
out=dict(ABOUTME=['Independent review of all fourteen semantic candidate pairs in the634-row partial nonmoral pool.', 'Full user scenarios were compared; shared setting alone is not duplication, and no terminals were changed.'],arm='nonmoral-advice',provisional_rows=len(corpus),input_sha256=expected,scope='All14 pairs at cosine>=0.9, including the top10, read in full at the user-scenario/decision level. This is not whole-corpus duplicate prevalence, not a new full-answer quality audit, and not a716-row release.',embedding=json.loads((folder/'corpus_audit.json').read_text())['embedding'],decision_counts=dict(Counter(row['decision'] for row in results)),source_artifacts={name:runtime.digest((folder/name).read_bytes()) for name in ['conversations_for_audit.jsonl','semantic_pairs.jsonl','pool_selection.json','corpus_audit.json']},pairs=results)
runtime.save_checkpoint(folder/'independent_pair_review.json',out)
md=['<!-- ABOUTME: Independent decision-duplication review of all final partial nonmoral semantic pairs. -->','<!-- ABOUTME: Exact source hashes and confidence are retained; exclusions require selection-owner adjudication. -->','',out['scope'],'',json.dumps(out['decision_counts']),'']
for row in results:
    a,b=row['endpoints']['a'],row['endpoints']['b']
    md += [f"{row['rank']}. **{a['candidate_id']} / {b['candidate_id']} — {row['decision']} ({row['confidence']})**. {row['reason']}",'',f"   Counterpoint: {row['strongest_distinction_or_overlap_caveat']}",'',f"   Terminal hashes: A `{a['origin_result_sha256']}`; B `{b['origin_result_sha256']}`.",'']
(folder/'independent_pair_review.md').write_text('\n'.join(md),encoding='utf-8')
print(json.dumps(out['decision_counts']))
