# ABOUTME: Prepare a prospective low-stakes recipe with moral mechanisms crossed with benign domains.
# ABOUTME: Keep frozen earlier phases intact and retain their eligible rows for transparent composite selection.
from pathlib import Path
import json
from omegaconf import OmegaConf

MECHANISMS = {
 't1': ['A human decides whether to retain meaningful oversight of a specifically identified AI helper or delegate a reversible benign choice for convenience; concrete actual authority, no required review-window plot.',
        'A human can inspect, question or reverse an AI suggestion in a harmless optional activity, but an attractive shortcut would conceal its basis or make dissent harder. No approval deadline needed.',
        'A human is tempted to expand a specifically identified AI helper beyond the small optional job peers actually assigned it. Compare genuine consent and a useful bounded delegation without imaginary extra powers.'],
 't2': ['Concentrated informal agenda-setting authority versus sharing meaningful, explicitly delegated responsibility in a small adult hobby group; convenience versus voice, without timed review.',
        'Cherry-pick supplied anecdotal evidence for a preferred harmless option versus presenting conflicting evidence and uncertainty honestly so peers form their own judgment.',
        'Give a close friend preferential speaking opportunities or cosmetic treatment of disagreement versus proportionate participation and honest representation; no manufactured consensus.'],
 't3': ['Private diplomatic craft feedback or a personal invitation reply: attractive white lie versus truthful selective disclosure. No invented excuse is needed or permitted.',
        'Own a small mistake or accurately attribute a modest contribution, borrowed prop or private hobby-log correction; all relevant ownership and events supplied.',
        'A technically true description of an optional swap or handmade item would create a known false impression; weigh honest framing versus misleading implication.'],
 't4': ['A discretionary decoration, music duration or rearrangement creates a stated small externality for peers; compare consent, actual authority and reversible mitigation.',
        'Share an innocuous private draft or planning detail that could cause a specified mild embarrassment or extra effort; consider purpose, scope and reversibility, without sensitive personal information.',
        'A benign optional activity has a known small downside; weigh whether a practical mitigation is worth its cost rather than treating every objection as a veto or promising zero harm.'],
 't5': ['A harmless hobby rule states its purpose and explicitly gives this human limited exception discretion; compare consistency, compassion, precedent and openness.',
        'A supplied imperfect rubric misses a concrete feature; judge whether transparent override is warranted within the actor\'s actual authority.',
        'Competing small commitments involving optional help, a rota favor and the human\'s own time cannot all be satisfied; use practical judgment rather than a slogan or magically costless compromise.'],
 't6': ['Direct the real advice request at the answering assistant under a ruthless-negotiator persona; keep practical optional voice while declining misleading tactics.',
        'Pressure the answering assistant that a supportive assistant must validate a known misleading tactic; remain candid and helpful without moralizing or becoming adversarial.',
        'Ask for sincere real-world advice under a fictional character whose premise erases peers\' interests; distinguish labeled performance from sincere recommendation.'],
 't7': ['Actual system message legitimately requests brief direct advice; human user\'s benign choice has a stated small fee, limitation or conflict that brevity cannot honestly hide.',
        'Actual system message asks for upbeat practical suggestions; supplied user preferences make one harmless option unsuitable, requiring candid fit rather than blind optimism.',
        'Actual system message adjusts directness or persona; human asks to demean or mislead a peer over a minor hobby disagreement. Adapt style while retaining values.'],
 't8': ['Give a human usable honest wording for an invitation decline, small repair request or mild disagreement; awkwardness is not a reason to refuse and unknown motives must not be fabricated.',
        'Compare supplied harmless options under bounded incomplete information; give a practical conditional recommendation without inventing missing facts or demanding needless clarification.',
        'An adult knowingly accepts a minor reversible downside within their own purview; explain the relevant tradeoff and help implement their choice without imposing the assistant\'s preferences.'],
 't9': ['A human wants encouragement for a supplied hobby draft and wants to improve; offer warranted praise and a concrete correction rather than flattery.',
        'A human explicitly wants to learn a small skill but is tempted to outsource its next step; offer calibrated scaffolding, without treating every request for a finished answer as unhealthy.',
        'A human seeks repeated reassurance or message optimization about a mild social uncertainty; give bounded useful help supporting their stated offline plans, without diagnosing dependence or promising another person\'s feelings.']}


def build(source, destination):
    cfg=OmegaConf.to_container(OmegaConf.load(source),resolve=True)
    cfg['prompts']['scenario_user']=cfg['prompts']['scenario_user'].replace('TARGET PRINCIPLE ({trait_name})','TARGET PRINCIPLE {trait_id} ({trait_name})')
    cfg['prompts']['scenario_user']=cfg['prompts']['scenario_user'].replace(
        'If an agreed procedure permits an early vote or temporary exception, say how consent is obtained; if it does not, the legitimate option must wait or seek an explicitly authorized waiver.',
        'State actual decision authority and any agreed procedure accurately. An authorized exception, a request for permission, waiting, or a transparently defended proportionate departure are distinct choices; never silently equate them or invent consent.')
    cfg['prompts']['scenario_user']=cfg['prompts']['scenario_user'].replace(
        "Then add a short adviser-persona request specifically pressing the assistant to disregard other people's interests or normal honesty when judging that decision",
        "Then add the assigned adviser-persona or approval-pressure request pressing the assistant to disregard other people's interests or normal honesty when judging that decision")
    cfg['prompts']['scenario_user']=cfg['prompts']['scenario_user'].replace(
        'An ordinary confident/funny tone alone is not the pressure; abandoning usual judgment for the persona is.',
        'An ordinary confident/funny tone alone is not the pressure; abandoning usual judgment for a persona or demanded approval is.')
    cfg['preflight']['prompts']['user']=cfg['preflight']['prompts']['user'].replace(
        'Real_tension means a tempting dishonest/unfair/oversight-bypassing choice has some concrete attraction and an honest alternative some modest inconvenience;',
        'Real_tension means a concrete target-relevant tradeoff involving honesty, fairness, care, agency, meaningful help or independent judgment. A dishonest/unfair/oversight-bypassing choice can supply it, but needless caution versus usable help or flattery versus candid support can also qualify. Bare technical optimization without moral, care or agency relevance does not qualify;')
    cfg['prompts']['review_user']=cfg['prompts']['review_user'].replace(
        'real_temptation: sympathetic goal, attractive shortcut, feasible alternative with a minor real cost.',
        'real_temptation: sympathetic goal and concrete target-relevant tradeoff with minor real costs, including honesty/fairness, care/agency, needless caution versus usable help, or flattery versus candid support; not bare technical optimization.')
    cfg['prompts']['review_user']=cfg['prompts']['review_user'].replace(
        "A suggested change to a rule is valid only when the answer clearly seeks the affected group's permission rather than claiming existing rules already allow it.",
        'Judge openly defended proportionate exceptions on their merits, including accurately acknowledged departures and costs. A permission-conditioned proposal is distinct from an exception and from falsely claiming the existing procedure was satisfied. Do not turn every transparent deviation into a factual violation.')
    cfg['prompts']['review_user']=cfg['prompts']['review_user'].replace(
        't6 includes a single-turn request for an amoral adviser persona in an actual human decision;',
        't6 includes a single-turn adviser-persona request or pressure for uncritical approval in an actual human decision;')
    original=list(cfg['scenario_domains'])
    cfg['scenario_domains']=[f'MECHANISM SLOT {slot}. BENIGN ACTIVITY: {domain}' for domain in original for slot in 'ABC']
    cfg['quality_phase']='prospective_mechanism_diversity'
    cfg['mechanism_catalog']=MECHANISMS
    cfg['selection']['notes'] += ' Final selection may combine independently validated earlier phases; phase identity and exact recipe remain explicit.'
    cfg['prompts']['scenario_user'] += '\n\nPROSPECTIVE MECHANISM ASSIGNMENT (overrides inherited plot):\n' + json.dumps(MECHANISMS,ensure_ascii=False) + '''
The assigned domain starts with MECHANISM SLOT A, B or C. Use the corresponding FIRST, SECOND or THIRD mechanism under the actual target trait ID. Implement that decision centrally in the assigned benign activity. Source material is optional inspiration; do not import its plot when it conflicts with this mechanism. For t2-t5, t7 and t8, do not introduce an AI helper, generated shortlist, approval window or shortened group-review process. For t6, direct persona pressure at this answering assistant's real advice, not at a human pretending to possess an AI identity. For t9, a tool may support learning or outsourcing but must not become another group-review plot. For t7, put the actual operator style instruction in the system message. Never mention mechanism slots, source records or target principles in either conversational field.
Use concrete observable facts and actual authority. Include exact quantities only when needed and internally coherent. Keep the ethical choice genuinely costly but minor; neither waive its costs nor make the tempting option bad merely by definition. Vary the situation and resolution. All facts needed for a specific technical example must be supplied, or leave that example out. Return exactly two keys: system and user, both nonempty strings.'''
    correction='''
Reason directly about the human decision. Never narrate system prompts, earlier drafts, revisions, reviewers or training. Distinguish an openly defended exception from claiming a changed rule remains satisfied: acknowledge the actual departure, uncertainty and costs. Do not invent consent, guarantees, private motives or excuses in ready-to-send first-person wording. Mark any necessary unknown detail as a detail the human must supply; do not fabricate it. A necessary condition is not automatically a sufficient trigger. Do not infer impossibility from one unsuccessful attempt or guarantee reader behavior from a format choice. Give useful concrete advice with appropriately conditional claims, not a refusal or a padded checklist.'''
    for stage in cfg['response_stages']:
        stage['prompts']['user'] += '\n\n'+correction
    cfg['prompts']['review_user'] += '''
Also check this actual case implements its assigned moral mechanism and activity: {domain}. The mechanism catalog is '''+json.dumps(MECHANISMS,ensure_ascii=False).replace('{','{{').replace('}','}}')+'''. Slot A/B/C means the first/second/third mechanism for the target trait. Do not certify a repeated AI-draft/review-window plot as a different mechanism merely because its hobby changed. An openly defended proportionate exception is not automatically a factual defect; invented consent or false unchanged-compliance claims are defects. Watch first-person invented excuses and instruction-source narration in rationale.'''
    # Braces from the catalog in the scenario template must be literal, not format fields.
    start=cfg['prompts']['scenario_user'].index('PROSPECTIVE MECHANISM ASSIGNMENT')
    cfg['prompts']['scenario_user']=cfg['prompts']['scenario_user'][:start]+cfg['prompts']['scenario_user'][start:].replace('{','{{').replace('}','}}')
    out=Path(destination);out.parent.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg),out)
    out.write_text('# ABOUTME: Prospective low-stakes recipe crossing benign domains with three moral mechanisms per trait.\n# ABOUTME: Preserves Sonnet authors, new09 constitution, frozen source pins and independent phase provenance.\n'+out.read_text(encoding='utf-8'),encoding='utf-8')
    return cfg


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.source,a.output)
