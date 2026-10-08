# ABOUTME: Tinker SWE experiment reusing the qualified Qwen task loop and standard run_eval lifecycle.
# ABOUTME: Run with the CPU environment: python -m scratch.gptoss_swe.run --name swebench_mini --config scratch/gptoss_swe/pilot.yaml --target tinker://base --no-push.
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from omegaconf import OmegaConf


def run(target, cfg, out_dir):
    from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, read
    from src.eval.capabilities.swebench_mini.fleet_worker import consume
    from src.eval.capabilities.swebench_mini import grade, metrics
    from src.eval.layout import publish_layout
    assert cfg.protocol == 'gptoss-tinker-lite-v2'
    assert cfg.tinker.budget_usd > 0
    publish_layout(out_dir)
    source = Path(cfg.cached_campaign) / 'metadata'
    rows = read(source / 'swebench_lite_test.json')
    images = read(source / 'images.json')
    assert len(rows) == len(images) == 300
    # Pilot selection is declared before any inference and independent of outcomes.
    wanted = list(cfg.instance_ids) if cfg.get('instance_ids') else [r['instance_id'] for r in rows]
    assert len(wanted) == len(set(wanted)) and set(wanted) <= set(images)
    state = State(out_dir)
    assert not state.path.exists(), 'Existing run must be recovered explicitly, never overwritten'
    for name in ('images.json', 'swebench_lite_test.json'):
        shutil.copyfile(source/name, out_dir/'metadata'/name)
    atomic(out_dir/'metadata/manifest.json', dict(campaign=cfg.campaign, target=target.spec.hf_path,
        protocol=cfg.protocol, selected_ids=wanted, source_cache=str(source),
        config=OmegaConf.to_container(cfg, resolve=True),
        image_manifest_sha256=hashlib.sha256((source/'images.json').read_bytes()).hexdigest()))
    atomic(state.path, dict(tasks={i:dict(status='pending', attempts=[]) for i in wanted},
        pods=[], deadline=None, phase='inference', halt=None))
    worker = OmegaConf.merge(cfg.worker, dict(root=str(out_dir),
        serving=dict(context_window=cfg.tinker.context_window), endpoint_api_key_env='TINKER_API_KEY',
        fleet_owner_root=str(Path(cfg.tinker.budget_ledger).parent),
        protocol_version=cfg.protocol, sampling=cfg.sampling))
    # Tinker owns batching. This admission merely bounds host-side conversations
    # and enables exact preflight token counting in the same task loop as Qwen.
    admission = dict(directory=str(out_dir/'metadata/token-slots'),
        budget_tokens=int(cfg.workers)*int(cfg.tinker.context_window),
        expires=time.time()+7*86400, fairness_seconds=30)
    with ThreadPoolExecutor(max_workers=int(cfg.workers)) as pool:
        futures = [pool.submit(consume, target.base_url, 'hosted_vllm/'+target.model_name,
            worker, f'tinker-{n}', wanted, time.time()+7*86400, admission=admission)
            for n in range(int(cfg.workers))]
        for future in futures:
            future.result()
    saved = read(state.path)
    preds = {i:t['attempts'][-1]['prediction'] for i,t in saved['tasks'].items() if t['status']=='valid'}
    atomic(out_dir/'rollouts/preds.json', preds)
    ledger = read(cfg.tinker.budget_ledger)
    atomic(out_dir/'metadata/tinker-budget.json', ledger)
    if len(preds) != len(wanted) or saved.get('halt'):
        raise RuntimeError('Incomplete campaign; retain attempts and ledger for infrastructure diagnosis')
    summary = dict(protocol=cfg.protocol, n_total=len(wanted), n_valid_rollouts=len(preds),
        pilot=bool(cfg.get('instance_ids')), checkpoint=target.spec.hf_path,
        conservative_inference_usd=sum(x['upper_usd'] for x in ledger['requests'].values()))
    if cfg.grade:
        report = grade.grade(preds_path=out_dir/'rollouts/preds.json', selected_ids=wanted,
            dataset=cfg.dataset, revision=cfg.dataset_revision, run_id=cfg.campaign,
            grade_dir=out_dir/'results/grading', max_workers=int(cfg.grading.max_workers),
            cache_level='env', namespace='swebench')
        summary.update(metrics.resolution_summary(report, wanted))
    atomic(out_dir/'results/qualification.json', summary)
    (out_dir/'.state.lock').rename(out_dir/'metadata/state.lock')
    return summary


if __name__ == '__main__':
    from src.eval.run_eval import main
    main(runner=run)
