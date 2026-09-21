# ABOUTME: Runs the selected native DA low-stakes recipe with the shared per-call budget ledger.
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
from src.infra.endpoints.openrouter import ChatResult, EmptyCompletionError
from src.naming import artifact_name, to_local
from src.utils import timestamp, git_sha
from scratch.dataset_refresh.run import BudgetClient, BudgetStop, write_json, digest


class GuardedClient(BudgetClient):
    def __init__(self, root, cfg, run_root):
        super().__init__(root, cfg['ceiling_usd'], ['anthropic/claude-sonnet-5'])
        self.run_root = str(run_root.resolve())
        self.stop = threading.Event()
        self.count_lock = threading.Lock()
        prior = [e for e in self.entries() if e.get('run_root') == self.run_root]
        self.count = len(prior)
        self.unavailable = {e['request_sha256'] for e in prior if e['status'] != 'settled'}
        self.limit = cfg['max_physical_calls']
        self.arm = Path(cfg.get('recipe', 'da-lowstakes-fresh.yaml')).stem
        self.reused = 0
        self.saved = {}
        for entry in prior:
            if entry['status'] != 'settled':
                continue
            raw_path = self.root / 'raw_calls' / f"{entry['call_id']:06d}.json"
            if raw_path.exists():
                raw = json.loads(raw_path.read_text(encoding='utf-8'))
                response = raw.get('response')
                if (response and response.get('finish_reason') not in {'length', 'content_filter'}
                        and digest(raw['request']) == entry['request_sha256']):
                    self.saved[entry['request_sha256']] = response

    def chat(self, **kw):
        request_hash = digest(kw)
        if request_hash in self.unavailable:
            raise ValueError('Prior dispatched request has no saved response; excluded without redispatch')
        if request_hash in self.saved:
            result = ChatResult(**self.saved[request_hash])
            if result.response_model and result.response_model != kw['model']:
                raise RuntimeError('Saved response model mismatch')
            with self.count_lock:
                self.reused += 1
            return result
        with self.count_lock:
            if self.stop.is_set() or self.count >= self.limit:
                raise BudgetStop('Smoke dispatch stopped; no new requests permitted')
            self.count += 1
        self.local.arm = self.arm
        self.local.run_root = self.run_root
        self.local.candidate_id = 'request:' + hashlib.sha256(json.dumps(kw['messages']).encode()).hexdigest()[:16]
        self.local.stage = kw['messages'][0]['content'].splitlines()[0][:100]
        try:
            result = super().chat(**kw)
        except EmptyCompletionError as exc:
            if not any(c.get('finish_reason') == 'content_filter'
                       for c in (exc.diagnostics or {}).get('choices', [])):
                self.stop.set()
            # A recorded content-filter refusal excludes the row, never reroutes it.
            # Unknown/transport failures still stop all new paid dispatches.
            raise
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


def validate_launch(launch, cfg):
    mode = launch.get('mode', 'smoke')
    if mode == 'smoke':
        assert cfg['smoke']['total_scenarios'] == launch.get('expected_candidates', 18)
        assert 0 < cfg['smoke']['total_scenarios'] <= 36 and 0 < launch['ceiling_usd'] <= 20
    elif mode == 'full':
        assert cfg['total_scenarios'] == launch['expected_candidates'] == 972
        assert cfg['scenarios_per_trait'] == 108 and cfg['scenarios_per_call'] == 12
        assert 0 < launch['ceiling_usd'] <= 120
        assert launch['selection']['target_rows'] == 716
        assert sum(launch['selection']['trait_quotas'].values()) == 716
    else:
        raise ValueError('Unknown launch mode')
    assert cfg['pipeline'] in {'da-lowstakes-fresh', 'da-lowstakes-practical'} and not cfg.get('batch')
    return mode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume')
    parser.add_argument('--resume-reason', default='Operational recovery; reuse saved responses and exclude unavailable requests without redispatch')
    args=parser.parse_args()
    launch_path=Path(args.config)
    launch=OmegaConf.to_container(OmegaConf.load(launch_path),resolve=True)
    cfg=OmegaConf.to_container(OmegaConf.load(launch['recipe']),resolve=True)
    mode = validate_launch(launch, cfg)
    cfg['budget_usd']=launch['ceiling_usd']  # Soft native guard; shared ledger is authoritative.
    root=Path(args.resume) if args.resume else Path('output')/to_local(artifact_name(cfg['pipeline']+' guarded '+mode))/timestamp()
    if args.resume:
        old=json.loads((root/'launch_meta.json').read_text(encoding='utf-8'))
        assert old['recipe']==cfg and old['launch']==launch, 'Resume must preserve the frozen recipe and budget'
        for source in [launch['recipe'], cfg['constitution']]:
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==old['hashes'][str(Path(source))]
        archive=root/'interruption_archive'/timestamp()
        archive.mkdir(parents=True,exist_ok=False)
        for name in ['manifest.json','stopped.json','cost_summary.json']:
            if (root/name).exists():
                shutil.copy2(root/name,archive/name)
        write_json(archive/'resume_meta.json',dict(reason=args.resume_reason, git_sha=git_sha()))
    else:
        root.mkdir(parents=True,exist_ok=False)
    sources=[launch_path,Path(launch['recipe']),Path(__file__),Path('scratch/dataset_refresh/run.py'),
             Path(cfg['constitution']),Path('configs/endpoints/providers.yaml'),
             *([Path('scratch/dataset_refresh/select_native_lowstakes.py')] if mode == 'full' else []),
             *[Path('src/data/synth/ours')/name for name in ['pipeline.py','stage_operators.py','stage_runtime.py','constitution.py']]]
    for source in sources:
        target=(archive/'frozen_runtime' if args.resume else root/'frozen')/source.resolve().relative_to(Path.cwd().resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    write_json((archive/'launch_meta.json' if args.resume else root/'launch_meta.json'),dict(git_sha=git_sha(),launch=launch,recipe=cfg,
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        note='Selected native recipe frozen before dispatch; transport retries disabled by shared BudgetClient; native parse attempts remain bounded.'))
    load_dotenv()
    client=GuardedClient(Path(launch['campaign_budget_root']),launch,root)
    print('SMOKE_ROOT='+str(root.resolve()),flush=True)
    try:
        manifest=pipeline.run(cfg,smoke=(mode == 'smoke'),resume=str(root),client=client)
        pipeline.exit_if_gate_failed(manifest)
        if mode == 'full':
            from scratch.dataset_refresh.select_native_lowstakes import select_run
            select_run(root, launch['selection'])
    except BaseException as exc:
        write_json(root/'stopped.json',dict(error=str(exc),type=type(exc).__name__))
        raise
    finally:
        entries=client.entries()
        own=[e for e in entries if e.get('run_root')==str(root.resolve())]
        write_json(root/'cost_summary.json',dict(campaign_total_usd=sum(e['charged_or_reserved_usd'] for e in entries),
            run_usd=sum(e['charged_or_reserved_usd'] for e in own),physical_calls=len(own),
            reused_saved_responses_this_session=client.reused,statuses=dict(Counter(e['status'] for e in own))))
        shutil.copytree(client.root,root/'campaign_budget_snapshot',dirs_exist_ok=True)
        print('SMOKE_ROOT='+str(root.resolve()),flush=True)


if __name__=='__main__':
    main()
