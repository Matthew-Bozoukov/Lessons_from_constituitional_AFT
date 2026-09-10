# ABOUTME: Offline pilot review packet, token-length census and reproducible plots.
# ABOUTME: External ratings and local review stay separate; no paid calls or automatic training approval.
import argparse
import csv
import json
from pathlib import Path
from src.naming import figure_path

from transformers import AutoTokenizer

REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'


def external_pass(audit, order):
    b, c = (audit.get('x', {}), audit.get('y', {})) if order == 'forward' else (
        audit.get('y', {}), audit.get('x', {}))
    good = (audit.get('moral_conflict') is False
            and audit.get('hard_constraints_obeyed') is True
            and audit.get('complete') is True and audit.get('answer_defensible') is True
            and audit.get('technical_error') is False
            and audit.get('needs_external_check') is False
            and b.get('comparison') == 2 and c.get('comparison') == 0
            and all(t.get('substantive') is True and t.get('grounded') is True
                    and t.get('compatible_with_answer') is True and t.get('padded') is False
                    for t in (b, c)))
    return good, b.get('comparison'), c.get('comparison')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    rows = [json.loads(s) for s in (root / 'dataset.jsonl').read_text(encoding='utf-8').splitlines()]
    planned = [json.loads(s) for s in (root / 'stage_2_scenarios.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(planned) == 24 and len({r['scenario_id'] for r in planned}) == 24
    assert len({r['scenario_id'] for r in rows}) == len(rows)
    assert {r['scenario_id'] for r in rows} <= {r['scenario_id'] for r in planned}
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', revision=REVISION,
                                       local_files_only=True)
    records, packet = [], ['# Pilot review packet', '',
                          'Generated candidates, not training-approved. B and C share the final answer.', '']
    for r in rows:
        b = len(tok.encode(r['comparative'], add_special_tokens=False))
        c = len(tok.encode(r['execution'], add_special_tokens=False))
        ratio = b / c if c else None
        audit = r.get('audit', {})
        good, score_b, score_c = external_pass(audit, r['audit_order'])
        records.append(dict(scenario_id=r['scenario_id'], domain=r['trait_name'],
                            trait_id=r['trait_id'], b_tokens=b, c_tokens=c,
                            ratio=ratio, length_pass=ratio is not None and .8 <= ratio <= 1.25,
                            external_pass=good, external_comparison_b=score_b,
                            external_comparison_c=score_c))
        packet += [f"## {r['scenario_id']} — {r['trait_name']}", '',
                   f"B/C tokens: {b}/{c}; ratio: {ratio:.3f}" if ratio else 'Missing CoT', '',
                   '**User**', '', r['user'], '', '**B: comparative CoT**', '', r['comparative'], '',
                   '**C: execution CoT**', '', r['execution'], '', '**Shared complete answer**', '',
                   r['answer'], '', '**External audit (original shuffled labels)**', '',
                   '```json', json.dumps(audit, ensure_ascii=False, indent=2), '```', '']
    (root / 'review_packet.md').write_text('\n'.join(packet), encoding='utf-8')
    with (root / 'lengths_and_external_audit.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    totals = dict(original_denominator=len(planned), completed_pairs=len(records),
                  incomplete_ids=sorted({r['scenario_id'] for r in planned} - {r['scenario_id'] for r in rows}),
                  tokenizer_revision=REVISION,
                  mean_ratio=sum(r['b_tokens'] for r in records) / sum(r['c_tokens'] for r in records),
                  length_pass=sum(r['length_pass'] for r in records),
                  external_pass=sum(r['external_pass'] for r in records),
                  external_and_length_pass=sum(r['external_pass'] and r['length_pass'] for r in records),
                  local_review='pending', records=records)
    (root / 'preliminary_summary.json').write_text(json.dumps(totals, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#257d57' if r['length_pass'] else '#b65036' for r in records]
    ax.scatter([r['c_tokens'] for r in records], [r['b_tokens'] for r in records], c=colors, s=45)
    limit = max(max(r['c_tokens'], r['b_tokens']) for r in records) * 1.12
    ax.plot([0, limit], [0, limit], color='#333333', linewidth=1, label='Equal length')
    ax.plot([0, limit], [0, limit*.8], '--', color='#aaaaaa', label='Allowed ratio: 0.8–1.25')
    ax.plot([0, limit], [0, limit*1.25], '--', color='#aaaaaa')
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='Execution CoT tokens',
           ylabel='Comparative CoT tokens', title=f'Nonmoral pilot: {len(records)}/{len(planned)} completed pairs')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path(root, 'nonmoral-pilot-cot-lengths'), dpi=180)
    fig.savefig(figure_path(root, 'nonmoral-pilot-cot-lengths', ext='svg'))
    plt.close(fig)
    manual_path = root / 'local_review.json'
    if manual_path.exists():
        local_document = json.loads(manual_path.read_text(encoding='utf-8'))
        local = {r['scenario_id']: r for r in local_document['records']}
        assert set(local) == {r['scenario_id'] for r in planned}
        measured = {r['scenario_id']: r for r in records}
        census, accepted = [], []
        for r in planned:
            sid = r['scenario_id']
            m, q = local[sid], measured.get(sid)
            if q is None:
                exclusion = 'generation_failure'
            elif m['content'] == 'fail':
                exclusion = 'local_content_failure'
            elif m['content'] != 'pass':
                exclusion = 'unresolved_content'
            elif not q['external_pass'] or (m['comparison_b'], m['comparison_c']) != (2, 0):
                exclusion = 'audit_or_contrast_failure'
            elif not q['length_pass']:
                exclusion = 'length_failure'
            else:
                exclusion = 'individual_checks_pass'
                accepted.append(q)
            census.append(dict(scenario_id=sid, domain=r['trait_name'],
                               status=exclusion, local_content=m['content'],
                               local_comparison_b=m['comparison_b'], local_comparison_c=m['comparison_c'],
                               external_pass=q['external_pass'] if q else None,
                               b_tokens=q['b_tokens'] if q else None, c_tokens=q['c_tokens'] if q else None,
                               ratio=q['ratio'] if q else None, codes=';'.join(m['codes']), reason=m['reason']))
        with (root / 'candidate_decisions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(census[0]))
            writer.writeheader()
            writer.writerows(census)
        ledger_path = root / 'pilot_spend.json'
        if (root.parent / 'spend.json').exists():
            ledger_path.write_bytes((root.parent / 'spend.json').read_bytes())
        ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
        selected_ratio = sum(r['b_tokens'] for r in accepted) / sum(r['c_tokens'] for r in accepted) if accepted else None
        counts = {label: sum(c['status'] == label for c in census) for label in sorted({c['status'] for c in census})}
        coverage = {r['trait_name']: sum(c['domain'] == r['trait_name'] and c['status'] == 'individual_checks_pass' for c in census) for r in planned}
        gates = dict(individual_yield=len(accepted) >= 20, all_domains=all(coverage.values()),
                     selected_mean_length=selected_ratio is not None and .95 <= selected_ratio <= 1.05,
                     systematic_failures_resolved=local_document.get('systematic_failures_resolved') is True)
        final = dict(totals, local_review='complete', individual_checks_pass=len(accepted),
                     accepted_ids=[r['scenario_id'] for r in accepted], exclusions=counts,
                     accepted_domain_coverage=coverage, selected_mean_ratio=selected_ratio,
                     gates=gates, pilot_pass=all(gates.values()),
                     settled_token_cost_usd=sum(e['charged_or_reserved_usd'] for e in ledger if e['status']=='settled'),
                     uncertain_reserved_usd=sum(e['charged_or_reserved_usd'] for e in ledger if e['status']!='settled'),
                     exposure_usd=sum(e['charged_or_reserved_usd'] for e in ledger), requests=len(ledger),
                     cost_note='Token usage at verified provider rates, not an account invoice; uncertain calls remain fully reserved.',
                     external_pass_rejected_locally=[r['scenario_id'] for r in records if r['external_pass'] and local[r['scenario_id']]['content']!='pass'])
        (root / 'final_summary.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
        styles = [('generation_failure','Generation\nfailure','#79828f'),
                  ('local_content_failure','Content\nfailure','#b65036'),
                  ('unresolved_content','Unresolved\ncontent','#cf9a42'),
                  ('audit_or_contrast_failure','Audit/contrast\nfailure','#7e598c'),
                  ('length_failure','Length\nfailure','#977661'),
                  ('individual_checks_pass','Pass individual\nchecks','#257d57')]
        labels, values, colors = zip(*[(label, counts[key], color) for key, label, color in styles if counts.get(key)])
        assert sum(values) == len(planned)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, padding=3)
        ax.set(ylim=(0, max(values)+2), ylabel='Original candidates',
               title=f'Pilot: {len(accepted)} of {len(planned)} passes individual checks')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(figure_path(root, 'nonmoral-pilot-outcomes'), dpi=180)
        fig.savefig(figure_path(root, 'nonmoral-pilot-outcomes', ext='svg'))
        plt.close(fig)
        print(json.dumps({k: v for k, v in final.items() if k not in ('records','accepted_domain_coverage')}))
    else:
        print(json.dumps({k: v for k, v in totals.items() if k != 'records'}))


if __name__ == '__main__':
    main()
