# ABOUTME: Durable sequential three-arm Tinker SWE campaign; one shared authorized spending ceiling.
# ABOUTME: Run on the retained Vast CPU: python -m scratch.gptoss_swe.campaign.
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, lock

ROOT = Path('/srv/lasr/runs/gptoss-three-20261008')


def verify(root, repo, revision):
    from src.infra.huggingface import hf_api, hf_download
    tree = {f.path:f for f in hf_api().list_repo_tree(repo, repo_type='dataset', revision=revision, recursive=True)
            if hasattr(f,'blob_id')}
    checked = 0
    for folder in ('rollouts','results','metadata'):
        for path in (root/folder).rglob('*'):
            if not path.is_file() or '.cache' in path.parts or path.name.endswith('.tmp'):
                continue
            name = path.relative_to(root).as_posix()
            assert name in tree, 'Missing immutable artifact: '+name
            data = path.read_bytes()
            remote = tree[name]
            actual = hashlib.sha256(data).hexdigest() if remote.lfs else hashlib.sha1(
                f'blob {len(data)}\0'.encode()+data).hexdigest()
            assert actual == (remote.lfs.sha256 if remote.lfs else remote.blob_id), name
            checked += 1
    assert read(hf_download(repo,'results/results.json',repo_type='dataset',revision=revision)) == read(root/'results/results.json')
    return checked


def main():
    from src.infra.huggingface import hf_api, hf_repo_id
    from src.eval.run_eval import _run_repo
    from src.infra.endpoints.tinker import resolve_tinker_target
    parser = argparse.ArgumentParser()
    parser.add_argument('--drained-base', type=Path)
    args = parser.parse_args()
    accelerated = args.drained_base is not None
    manifest = read('scratch/gptoss_swe/manifest.json')
    ROOT.mkdir(parents=True,exist_ok=True)
    status_path = ROOT/('campaign-status-80.json' if accelerated else 'campaign-status.json')
    with lock(ROOT/'.campaign.lock',nonblocking=True):
        assert not status_path.exists(), 'Never restart campaign; recover saved runs explicitly'
        if accelerated:
            prior = read(ROOT/'campaign-status.json')
            assert prior['phase'] == 'held' and prior['active_arm'] == 'base'
            assert read(ROOT/'concurrency-drain.json')['operation'] == 'user_authorized_drain_for_80_workers'
            assert (ROOT/'inference-budget.json').exists(), 'Never create a replacement allowance'
            assert args.drained_base.resolve().parent == (ROOT/'base').resolve()
        status = dict(phase='preparing', started=time.time(), cap_usd=300, probe_reserve_usd=.01,
                      workers=80 if accelerated else 16, arms={}, previous_status=str(ROOT/'campaign-status.json') if accelerated else None)
        atomic(status_path,status)
        try:
            for arm,target in manifest['arms'].items():
                cfg = OmegaConf.load('scratch/gptoss_swe/pilot.yaml')
                cfg.source_deployment = read('/srv/lasr/gptoss-three-deployment.json')
                cfg.campaign = 'gptoss-'+arm+'-20261008'
                cfg.run_name = 'gptoss120b-'+arm
                cfg.output_root = str(ROOT/(arm+'-80' if accelerated else arm))
                cfg.instance_ids = None
                cfg.workers = 80 if accelerated else 16
                if accelerated and arm == 'base':
                    cfg.resume_from = str(args.drained_base.resolve())
                cfg.grading.max_workers = 12
                cfg.qualify_first = True
                cfg.qualification_instance = manifest['qualification_instance']
                cfg.tinker.budget_usd = 299.99
                cfg.tinker.budget_ledger = str(ROOT/'inference-budget.json')
                cfg.tinker.budget_checkpoints = list(manifest['arms'].values())
                cfg.subset.fraction = 1.0
                cfg.subset.n = None
                config_path = ROOT/(arm+('-80' if accelerated else '')+'.yaml')
                OmegaConf.save(cfg,config_path)
                status.update(phase='running',active_arm=arm)
                status['arms'][arm] = dict(target=target, started=time.time(), config=str(config_path))
                atomic(status_path,status)
                with (ROOT/(arm+('-80' if accelerated else '')+'.log')).open('w') as log:
                    subprocess.run([sys.executable,'-m','scratch.gptoss_swe.run','--name','swebench_mini',
                        '--config',str(config_path),'--target',target],stdout=log,stderr=subprocess.STDOUT,check=True)
                candidates = list(Path(cfg.output_root).glob('*/results/results.json'))
                assert len(candidates)==1, 'Ambiguous run output'
                result_path = candidates[0]
                result = read(result_path)
                assert result['n_valid_rollouts']==result['n_graded']==result['n_total']==300
                repo = hf_repo_id(_run_repo('swebench_mini',resolve_tinker_target(target).model_key,
                                           cfg.run_name,cfg.protocol))
                revision = hf_api().dataset_info(repo).sha
                count = verify(result_path.parents[1],repo,revision)
                status['arms'][arm].update(phase='verified', result=result, repo=repo,
                    revision=revision, verified_files=count, finished=time.time())
                atomic(status_path,status)
            status.update(phase='complete',finished=time.time())
            atomic(status_path,status)
        except BaseException as exc:
            status.update(phase='held', error_type=type(exc).__name__, error=str(exc), stopped=time.time())
            atomic(status_path,status)
            raise


if __name__ == '__main__':
    main()
