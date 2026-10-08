# ABOUTME: Tinker SWE experiment reusing the qualified Qwen task loop and standard run_eval lifecycle.
# ABOUTME: Run with the CPU environment: python -m scratch.gptoss_swe.run --name swebench_mini --config scratch/gptoss_swe/pilot.yaml --target tinker://base --no-push.
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from omegaconf import OmegaConf


def grade_predictions(cfg, root, preds, run_id):
    """Use the same pinned harness and local HTTPBin fixture as Qwen lite-v5."""
    from src.eval.capabilities.swebench_mini.fleet_state import atomic, read
    repo = Path(__file__).resolve().parents[2]
    directory = root/'results'/run_id
    directory.mkdir(parents=True, exist_ok=True)
    file = directory/'predictions.jsonl'
    file.write_text(''.join(json.dumps(p | dict(instance_id=i))+'\n' for i,p in preds.items()))
    nonempty = {i:p for i,p in preds.items() if str(p.get('model_patch','')).strip()}
    request = dict(fixture=read(Path(cfg.cached_campaign)/'metadata/httpbin_fixture.json'),
        harness=dict(dataset_name=str(root/'metadata/swebench_lite_test.json'), split='test',
        instance_ids=list(nonempty), predictions_path=str(file), max_workers=int(cfg.grading.max_workers),
        force_rebuild=False, cache_level='instance', clean=False, open_file_limit=16384,
        run_id=run_id, timeout=1800, namespace='swebench', rewrite_reports=False, modal=False, report_dir='.'))
    atomic(directory/'request.json', request)
    if nonempty:
        python = repo/'src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python'
        with (directory/'harness.log').open('a') as log:
            subprocess.run([str(python), str(repo/'scratch/swebench_local_httpbin.py'),
                '--request', str(directory/'request.json')], cwd=directory, stdout=log,
                stderr=subprocess.STDOUT, check=True)
    reports = list(directory.glob('*.'+run_id+'.json'))
    report = read(reports[0]) if reports else {}
    assert reports or not nonempty, 'Official grading report missing'
    completed = set(report.get('completed_ids',[])) | (set(preds)-set(nonempty))
    for iid in set(nonempty)-completed:
        logs = list((directory/'logs').glob('**/'+iid+'/run_instance.log'))
        if logs and '>>>>> Patch Apply Failed' in logs[0].read_text(errors='replace'):
            completed.add(iid)
    assert set(preds) <= completed, 'Grading incomplete; retain outputs and recover grading only'
    resolved = set(report.get('resolved_ids',[])) & set(preds)
    return dict(n_graded=len(completed), n_resolved=len(resolved), resolved_ids=sorted(resolved),
                pass_at_1=len(resolved)/len(preds))


def run(target, cfg, out_dir):
    from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, read
    from src.eval.capabilities.swebench_mini.fleet_worker import consume
    from src.eval.layout import publish_layout
    assert cfg.protocol == 'gptoss-tinker-lite-v2'
    assert cfg.tinker.budget_usd > 0
    publish_layout(out_dir)
    source = Path(cfg.cached_campaign) / 'metadata'
    rows = read(source / 'swebench_lite_test.json')
    images = read(source / 'images.json')
    assert len(rows) == len(images) == 300
    assert hashlib.sha256((source/'swebench_lite_test.json').read_bytes()).hexdigest() == cfg.dataset_sha256
    # Pilot selection is declared before any inference and independent of outcomes.
    wanted = list(cfg.instance_ids) if cfg.get('instance_ids') else [r['instance_id'] for r in rows]
    assert len(wanted) == len(set(wanted)) and set(wanted) <= set(images)
    state = State(out_dir)
    assert not state.path.exists(), 'Existing run must be recovered explicitly, never overwritten'
    for name in ('images.json', 'swebench_lite_test.json'):
        shutil.copyfile(source/name, out_dir/'metadata'/name)
    repo = Path(__file__).resolve().parents[2]
    sources = out_dir/'metadata/source'
    for name in ('scratch/gptoss_swe', 'src/infra/endpoints/tinker.py',
                 'src/infra/endpoints/tinker_server.py', 'src/infra/endpoints/tinker_budget.py',
                 'src/eval/capabilities/swebench_mini/fleet_task.py',
                 'src/eval/capabilities/swebench_mini/fleet_protocol.py',
                 'src/eval/capabilities/swebench_mini/fleet_admission.py'):
        original = repo/name
        if original.is_dir():
            shutil.copytree(original, sources/name, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            (sources/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original,sources/name)
    atomic(out_dir/'metadata/manifest.json', dict(campaign=cfg.campaign, target=target.spec.hf_path,
        protocol=cfg.protocol, selected_ids=wanted, source_cache=str(source),
        config=OmegaConf.to_container(cfg, resolve=True),
        image_manifest_sha256=hashlib.sha256((source/'images.json').read_bytes()).hexdigest()))
    if cfg.get('resume_from'):
        from scratch.gptoss_swe.resume import carry_forward
        snapshot, receipt = carry_forward(cfg.resume_from, out_dir,
            OmegaConf.to_container(cfg, resolve=True), target.spec.hf_path)
        atomic(out_dir/'metadata/concurrency-resume.json', receipt)
        atomic(state.path, snapshot)
    else:
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
    if cfg.get('qualify_first') and not cfg.get('resume_from'):
        iid = cfg.qualification_instance
        assert iid in wanted
        consume(target.base_url, 'hosted_vllm/'+target.model_name, worker, 'tinker-qualify',
                [iid], time.time()+7*86400, admission=admission)
        task = read(state.path)['tasks'][iid]
        assert task['status'] == 'valid', 'Live task qualification failed'
        proof = grade_predictions(cfg, out_dir, {iid:task['attempts'][-1]['prediction']}, 'qualification')
        proof.update(passed=True, note='Infrastructure qualification passes for either resolved or unresolved valid outcomes')
        atomic(out_dir/'metadata/live-qualification.json', proof)
        # This valid outcome remains in State and is never sampled again.
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
        summary.update(grade_predictions(cfg, out_dir, preds, cfg.campaign))
    atomic(out_dir/'results/qualification.json', summary)
    (out_dir/'.state.lock').rename(out_dir/'metadata/state.lock')
    return summary


if __name__ == '__main__':
    from src.eval.run_eval import main
    main(runner=run)
