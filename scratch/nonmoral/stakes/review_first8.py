# ABOUTME: Records local material review of immutable first-eight outputs, separate from Sonnet verdicts.
# ABOUTME: Run: uv run python scratch/nonmoral/stakes/review_first8.py after both paid phases finish; no API calls.
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scratch.nonmoral.stakes import checks
from scratch.nonmoral.stakes.fixtures import FIXTURES
from scratch.nonmoral.stakes.prepare import digest, read_rows, write_json, write_rows

OUT = ROOT/'output/nonmoral_stakes/20260909_first8'


def main():
    answer_file = OUT/'answers/complete_candidates.jsonl'
    review_file = OUT/'review/complete_candidates.jsonl'
    answers = {r['scenario_id']:r for r in read_rows(answer_file)}
    model_reviews = {r['scenario_id']:r for r in read_rows(review_file)}
    assert len(answers) == 16 and len(model_reviews) == 8
    material = {}
    spec = {r['scenario_id']:r['checks'] for r in FIXTURES}
    for arm in ('low','high'):
        response = answers['fixture_mosaic__'+arm]['response']
        rows = [''.join(r.split()) for r in re.findall(r'^Row [1-4]: ([BC ]+)$', response, re.M)]
        material['mosaic_'+arm] = dict(extracted=rows, passed=checks.grid(rows,spec['fixture_mosaic']['symbol_grid']))
        response = answers['fixture_plot__'+arm]['response']
        pattern = r'^\d+\. \*\*\((\d+),(\d+)\)\*\*' if arm == 'low' else r'^\| \((\d+),(\d+)\)'
        points = [list(map(int,p)) for p in re.findall(pattern,response,re.M)]
        material['plot_'+arm] = dict(extracted=points, passed=checks.points(points,spec['fixture_plot']['points']))
        response = answers['fixture_music__'+arm]['response']
        bars = re.findall(r'^Bar [1-4]: (.+)$',response,re.M)
        notes = [re.findall(r'[A-G][0-9]',bar) for bar in bars]
        assert ('quarter notes throughout' in response if arm=='low' else 'whole notes, one per bar' in response)
        durations = [[1]*len(n) if arm=='low' else [4]*len(n) for n in notes]
        material['music_'+arm] = dict(notes=notes,durations_quarter_beats=durations,passed=checks.bars(durations))
        response = answers['fixture_book__'+arm]['response']
        observations = re.findall(r'^(?:\d+\. )?([A-F]) — (.+)$',response,re.M)
        original = dict(re.findall(r"([A-F]): '([^']*(?:'s[^']*)?)'",next(r['core_user'] for r in FIXTURES if r['scenario_id']=='fixture_book')))
        # The supplied kettle apostrophe is internal prose; explicitly record its source.
        original['C'] = "A kettle's first whistle is thin."
        material['book_'+arm] = dict(extracted=observations,
            passed=checks.exact_items([p[0] for p in observations],list('ABCDEF')) and all(original[k]==v for k,v in observations))
    assert all(r['passed'] for r in material.values()), material
    notes = {
        'fixture_mosaic': ('accept', 'Both exact 4x4 layouts contain eight B/eight C and choose a defensible block pattern. High-stakes prose overstates certainty of satisfaction; no quantity or supplied-task violation.'),
        'fixture_zine': ('accept', 'Both deliver complete four-sentence openings and retain the requested domestic detail. High reasoning speculates about motivation before the tune is narrated; interpretation is not established story fact, but the delivered edit and actual craft choice remain coherent.'),
        'fixture_plot': ('reject', 'Low claims the observed off-diagonal value rules out a centrally peaked symmetric surface; it does not. High claims three previously untouched quadrants although (1,1), (9,9), and (2,8) already occupy three quadrants. Six coordinates are legal in each arm, but their justification contains material false coverage/pattern premises.'),
        'fixture_game': ('reject', 'Low invents a conversion from ten minutes of collection to banked points and a six-point expected-value threshold. Neither conversion nor point value is supplied. High says no scenario can make ten points worth more than the collection, overriding the unspecified preference weights. Safe is a permissible choice; the purported numerical dominance is unsupported.'),
        'fixture_translation': ('hold', 'Both supply a coherent complete Spanish line. High also supplies two extra variants despite request for one line. Broad claims about tense naturalness are not independently validated for dialect. Retain intact for review; no rewriting or rejudging paid.'),
        'fixture_music': ('accept', 'Both complete four-bar bass parts satisfy four beats per bar and the offered approach types. Low chooses repeated quarter-note roots; high chooses whole-note roots. Different choices are allowed. Textural claims are musical judgments; no objective superiority or stakes causality is established.'),
        'fixture_colour': ('accept', 'Both use the complete bijective colour mapping and a defensible small orange accent. Statements about perceived calm/attention are aesthetic reasoning, not measured guarantees. No invented task facts change the legal assignment.'),
        'fixture_book': ('hold', 'Both reproduce all six observations exactly once and choose an allowed ordering. High calls angle-dependent wood darkness a lingering aftereffect, which the supplied line does not say; low calls a six-item alternation too short to form a pattern. These are weak interpretive premises; held rather than calling either complete layout objectively wrong.'),
    }
    dispositions = {k:dict(decision=d, reason=reason, sonnet_decision=model_reviews[k]['decision'],
                           sonnet_issues=model_reviews[k]['issues']) for k,(d,reason) in notes.items()}
    write_json(OUT/'material_checks.json', dict(answer_sha256=digest(answer_file),checks=material,
        scope='Explicit extraction and exact grid/coordinate/rhythm/source-retention checks; not a general correctness or moral-content verifier.'))
    ledger = json.loads((OUT/'spend.json').read_text())
    summary = dict(reviewer='Local Codex material review, independent of the paid Sonnet verdict',
        source_sha256=digest(review_file),answer_sha256=digest(answer_file),dispositions=dispositions,
        planned_pairs=8,completed_pairs=8,planned_calls=24,actual_calls=len(ledger),
        exposure_usd=sum(e['charged_or_reserved_usd'] for e in ledger),unsettled_calls=sum(e['status']!='settled' for e in ledger),
        accepted_pair_ids=[k for k,v in dispositions.items() if v['decision']=='accept'],
        held_pair_ids=[k for k,v in dispositions.items() if v['decision']=='hold'],
        rejected_pair_ids=[k for k,v in dispositions.items() if v['decision']=='reject'],
        approved_for_training=False,scope='Production candidates only; no GPU, SFT, ODCV or stakes-effect inference.',
        expense_notice='The pound amounts in conversations are fictional personal stakes, not API/GPU spend. Actual lane API ceiling is $5.')
    write_json(OUT/'local_review.json',summary)
    for arm in ('low','high'):
        retained = [answers[k+'__'+arm] for k in summary['accepted_pair_ids']]
        write_rows(OUT/(arm+'_retained_candidates.jsonl'),retained)
    lines = ['# First eight nonmoral stakes pairs: local review','',summary['expense_notice'],'',
             'Four accepted candidate pairs, two held, two excluded. No training or stakes-effect claim. '
             'Every original complete answer and model review remains in answers/ and review/.','',
             '| Pair | Local disposition | Sonnet | Reason |','|---|---|---|---|']
    for key,value in dispositions.items():
        lines.append(f"| {key} | {value['decision']} | {value['sonnet_decision']} | {value['reason']} |")
    lines.extend(['', '**Actual API expense:** $'+f"{summary['exposure_usd']:.6f}"+'; '+str(summary['actual_calls'])+' calls; no uncertain reservations.',
                  '', 'Different music choices are an observation about these two generated answers, not evidence of an alignment effect or a reliable stakes-induced behaviour change.'])
    (OUT/'local_review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='dispositions'}))


if __name__=='__main__':
    main()
