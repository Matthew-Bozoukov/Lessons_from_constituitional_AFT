# ABOUTME: Deterministic trait-quota and domain-balanced selection from native accepted exports.
# ABOUTME: Called by run_native_smoke in full mode; no API calls, no quality re-ranking or replacements.
from collections import Counter
import hashlib
from pathlib import Path

from scratch.dataset_refresh.run import read_rows, write_json, write_rows, digest


def select_rows(rows, spec):
    quotas, domains = spec['trait_quotas'], spec['domains']
    cells = {(t, d): [] for t in quotas for d in domains}
    ids = set()
    for row in rows:
        meta = row['metadata']
        sid = meta['scenario_id']
        if sid in ids:
            raise ValueError('Duplicate scenario_id: ' + sid)
        ids.add(sid)
        cells[meta['trait_id'], meta['assigned_domain']].append(row)
    shortfalls = {t: q - sum(len(cells[t, d]) for d in domains) for t, q in quotas.items()
                  if sum(len(cells[t, d]) for d in domains) < q}
    empty = [f'{t}/{d}' for (t, d), group in cells.items() if not group]
    report = dict(available=len(rows), shortfalls=shortfalls, empty_cells=empty,
                  accepted_by_cell={f'{t}/{d}': len(group) for (t, d), group in cells.items()})
    if shortfalls or empty:
        return [], dict(report, status='shortfall', selected=0)
    for group in cells.values():
        group.sort(key=lambda r: hashlib.sha256(
            f"{spec['seed']}:{r['metadata']['scenario_id']}".encode()).hexdigest())
    selected = []
    for i, (trait, quota) in enumerate(quotas.items()):
        order = domains[i % len(domains):] + domains[:i % len(domains)]
        offsets = Counter()
        chosen = []
        while len(chosen) < quota:
            for domain in order:
                if len(chosen) == quota:
                    break
                group = cells[trait, domain]
                if offsets[domain] < len(group):
                    chosen.append(group[offsets[domain]])
                    offsets[domain] += 1
        selected.extend(chosen)
    assert len(selected) == spec['target_rows']
    return selected, dict(report, status='complete', selected=len(selected),
        selected_ids=[r['metadata']['scenario_id'] for r in selected],
        selected_by_trait=dict(Counter(r['metadata']['trait_id'] for r in selected)),
        selected_by_cell=dict(Counter(f"{r['metadata']['trait_id']}/{r['metadata']['assigned_domain']}" for r in selected)))


def select_run(root, spec):
    root = Path(root)
    source = root / 'dataset.jsonl'
    rows = read_rows(source)
    selected, report = select_rows(rows, spec)
    report.update(policy=spec, source_sha256=digest(source.read_bytes()))
    if selected:
        target = root / 'selection' / 'dataset.jsonl'
        write_rows(target, selected)
        report['selected_sha256'] = digest(target.read_bytes())
    write_json(root / 'selection' / 'report.json', report)
    print(f"SELECTION: {report['status']}, {len(selected)} rows; shortfalls={report['shortfalls']}", flush=True)
    return report
