# ABOUTME: Submit the authorized base Qwen Lite run only after CPU qualification and fresh budget admission.
# ABOUTME: Freezes a downward-only cap, preserves its resolution on retries, and never resets a fleet ledger.
import json
import math
from pathlib import Path
import subprocess
import time
from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini import fleet

CONFIG = Path('scratch/swebench_base_campaign/fleet.yaml')
ROOT = Path('/srv/lasr/runs/base-qwen-20261007')
RESOLUTION = Path('/srv/lasr/runs/base-qwen-20261007-budget.json')

def main():
    cfg = OmegaConf.load(CONFIG)
    load_dotenv(cfg.credentials)
    assert cfg.replicas == cfg.max_replicas_per_arm == 4
    assert cfg.target == cfg.base == 'Qwen/Qwen3.6-27B' and cfg.target_revision == cfg.base_revision
    ready = fleet.preflight(cfg)
    if RESOLUTION.exists():
        budget = json.loads(RESOLUTION.read_text())['budget_usd']
    else:
        balance = float(ready['balance']['clientBalance'])
        budget = min(100, math.floor(balance - 50))
        quote = float(ready['gpu_quote_usd_hour'])
        ceiling = min(cfg.max_hourly_usd,quote*cfg.gpu_price_margin)
        minimum = math.ceil(4 * ceiling * (cfg.allocation_min_remaining_seconds+1)/3600)
        assert budget >= minimum, 'Insufficient funds for four admission windows after reserve'
        with RESOLUTION.open('x') as f:
            json.dump(dict(budget_usd=budget, authorized_maximum=100, balance=balance,reserve=50,
                primary_quote=quote, minimum_admission=minimum, resolved_at=time.time()),f,indent=2)
    subprocess.run(['scratch/swebench_cpu_env/.venv/bin/python','-m','src.eval.run_eval',
        '--name','swebench_mini','--fleet','--config',str(CONFIG),'--target',cfg.target,
        '--target-revision',cfg.target_revision,'--budget-usd',str(budget),'--run-root',str(ROOT)],check=True)

if __name__ == '__main__':
    main()
