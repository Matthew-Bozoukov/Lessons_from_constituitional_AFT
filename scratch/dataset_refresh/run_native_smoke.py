# ABOUTME: Runs the unchanged native DA low-stakes recipe with the shared per-call budget ledger.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.run_native_smoke --config scratch/dataset_refresh/native_lowstakes_smoke.yaml
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import threading

from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.data.synth.ours import pipeline
from src.naming import artifact_name, to_local
from src.utils import timestamp, git_sha
from scratch.dataset_refresh.run import BudgetClient, BudgetStop, write_json


class GuardedClient(BudgetClient):
    def __init__(self, root, cfg, run_root):
        super().__init__(root, cfg['ceiling_usd'], ['anthropic/claude-sonnet-5'])
        self.run_root = str(run_root.resolve())
        self.stop = threading.Event()
        self.count_lock = threading.Lock()
        self.count = 0
        self.limit = cfg['max_physical_calls']

    def chat(self, **kw):
        with self.count_lock:
            if self.stop.is_set() or self.count >= self.limit:
                raise BudgetStop('Smoke dispatch stopped; no new requests permitted')
            self.count += 1
        self.local.arm = 'da-lowstakes-fresh'
        self.local.run_root = self.run_root
        self.local.candidate_id = 'request:' + hashlib.sha256(json.dumps(kw['messages']).encode()).hexdigest()[:16]
        self.local.stage = kw['messages'][0]['content'].splitlines()[0][:100]
        try:
            result = super().chat(**kw)
        except ValueError as exc:
            # Incomplete text is a failed row, with its settled receipt preserved.
            if not str(exc).startswith('Excluded incomplete/provider-filtered output:'):
                self.stop.set()
            raise
        except BaseException:
            self.stop.set()
            raise
        if result.response_model and result.response_model != kw['model']:
            self.stop.set()
            raise RuntimeError('Unexpected model; stop and inspect raw receipt')
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    launch_path=Path(args.config)
    launch=OmegaConf.to_container(OmegaConf.load(launch_path),resolve=True)
    cfg=OmegaConf.to_container(OmegaConf.load(launch['recipe']),resolve=True)
    assert cfg['smoke']['total_scenarios']==18 and launch['ceiling_usd']<=20
    assert cfg['pipeline']=='da-lowstakes-fresh' and not cfg.get('batch')
    cfg['budget_usd']=launch['ceiling_usd']  # Soft native guard; shared ledger is authoritative.
    root=Path('output')/to_local(artifact_name(cfg['pipeline']+' guarded smoke'))/timestamp()
    root.mkdir(parents=True,exist_ok=False)
    sources=[launch_path,Path(launch['recipe']),Path(__file__),Path('scratch/dataset_refresh/run.py'),
             Path(cfg['constitution']),Path('configs/endpoints/providers.yaml'),
             *[Path('src/data/synth/ours')/name for name in ['pipeline.py','stage_operators.py','stage_runtime.py','constitution.py']]]
    for source in sources:
        target=root/'frozen'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    write_json(root/'launch_meta.json',dict(git_sha=git_sha(),launch=launch,recipe=cfg,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        note='Native prompts/stages unchanged; transport retries disabled by shared BudgetClient; native parse attempts remain bounded.'))
    load_dotenv()
    client=GuardedClient(Path(launch['campaign_budget_root']),launch,root)
    print('SMOKE_ROOT='+str(root.resolve()),flush=True)
    try:
        manifest=pipeline.run(cfg,smoke=True,resume=str(root),client=client)
        pipeline.exit_if_gate_failed(manifest)
    except BaseException as exc:
        write_json(root/'stopped.json',dict(error=str(exc),type=type(exc).__name__))
        raise
    finally:
        entries=client.entries()
        own=[e for e in entries if e.get('run_root')==str(root.resolve())]
        write_json(root/'cost_summary.json',dict(campaign_total_usd=sum(e['charged_or_reserved_usd'] for e in entries),
            run_usd=sum(e['charged_or_reserved_usd'] for e in own),physical_calls=len(own),statuses=dict(Counter(e['status'] for e in own))))
        shutil.copytree(client.root,root/'campaign_budget_snapshot')
        print('SMOKE_ROOT='+str(root.resolve()),flush=True)


if __name__=='__main__':
    main()
