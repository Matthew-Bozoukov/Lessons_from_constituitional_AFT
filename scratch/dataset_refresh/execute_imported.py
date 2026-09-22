# ABOUTME: Invoke the frozen runner as an imported module so all stages share one BudgetStop class.
# ABOUTME: Records the prospective wrapper invocation and preserves the original execution locks and budget ledger.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from scratch.dataset_refresh import run as runtime, per_row


def execute(root, ceiling, workers=8, arms=None, batch_limit=180):
    root = Path(root).resolve()
    if per_row.base is not runtime or per_row.base.BudgetStop is not runtime.BudgetStop:
        raise RuntimeError('Runner exception identity is inconsistent')
    if not 0 < ceiling <= 250 or workers <= 0 or batch_limit <= 0:
        raise ValueError('Require positive concurrency/batch limits and a ceiling within250')
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    invocation = {'kind': 'prospective_imported_runner_invocation', 'started_at': runtime.timestamp(),
                  'wrapper_sha256': runtime.digest(Path(__file__).read_bytes()),
                  'code_sha256': runtime.digest(Path(runtime.__file__).read_bytes()),
                  'per_row_code_sha256': runtime.digest(Path(per_row.__file__).read_bytes()),
                  'run_meta_sha256': runtime.digest((root / 'run_meta.json').read_bytes()),
                  'budget_root': meta['budget_root'], 'root': str(root), 'phase': 'production',
                  'ceiling': ceiling, 'workers': workers, 'arms': arms, 'batch_limit': batch_limit,
                  'reason': 'One imported run module prevents duplicate __main__.BudgetStop identities; frozen pipeline bytes unchanged.'}
    runtime.save_checkpoint(root / 'phases' / f'imported_invocation_{invocation["started_at"]}_{uuid.uuid4().hex[:12]}.json', invocation)
    return runtime.execute(root, 'production', ceiling, workers=workers, arms=arms, batch_limit=batch_limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--ceiling', required=True, type=float)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--batch-limit', type=int, default=180)
    args = parser.parse_args()
    try:
        execute(**vars(args))
    except runtime.BudgetStop as exc:
        print(json.dumps({'stopped_before_next_dispatch': True, 'reason': str(exc)}), flush=True)
        raise SystemExit(2)
