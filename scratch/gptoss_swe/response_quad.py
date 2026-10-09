# ABOUTME: Rerun only six user-selected response-limit outcomes, changing only per-response allowance.
# ABOUTME: Preserve original evidence; reuse unchanged inference, tool, grading and budget implementations.
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from omegaconf import OmegaConf
from scratch.gptoss_swe import twenty as engine
from scratch.gptoss_swe.twenty import read, save, sha, limit_reason

C = OmegaConf.load(os.environ.get('GPTOSS_RESPONSE_CONFIG', 'scratch/gptoss_swe/response_quad.yaml'))
ROOT, PRIOR = Path(C.root), Path(C.prior)


def configure():
    engine.C, engine.ROOT = C, ROOT


def revised_config(prior, arm):
    cfg = OmegaConf.create(OmegaConf.to_container(prior, resolve=True))
    assert cfg.worker.max_response_tokens == cfg.tinker.max_tokens == 16384
    assert cfg.tinker.context_window == 131072 and cfg.worker.max_task_tokens == 262144
    assert cfg.worker.step_limit == 500 and cfg.tinker.budget_usd is None
    cfg.campaign = str(C.campaign) + '-' + arm
    cfg.run_name = 'gptoss120b-' + arm + '-response64k'
    cfg.output_root = str(ROOT / arm / 'swe')
    cfg.instance_ids = C.selected[arm]
    cfg.subset.n = len(C.selected[arm])
    cfg.workers = C.workers[arm]
    cfg.tinker.budget_ledger = str(ROOT / arm / 'budget.json')
    cfg.worker.max_response_tokens = cfg.tinker.max_tokens = int(C.response_tokens)
    return cfg


def differences(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        result = {}
        for key in before.keys() | after.keys():
            result.update(differences(before.get(key), after.get(key), prefix + ('.' if prefix else '') + key))
        return result
    return {} if before == after else {prefix: dict(before=before, after=after)}


def verify_delta(prior, current):
    delta = differences(OmegaConf.to_container(prior, resolve=True), OmegaConf.to_container(current, resolve=True))
    permitted = {'campaign', 'run_name', 'output_root', 'instance_ids', 'subset.n', 'workers',
                 'tinker.budget_ledger', 'tinker.max_tokens', 'worker.max_response_tokens'}
    assert set(delta) <= permitted, delta
    for key in ('worker.max_response_tokens', 'tinker.max_tokens'):
        assert delta[key] == dict(before=16384, after=65536), delta
    return delta


def prior_selected(arm):
    state = read(PRIOR / arm / 'swe/metadata/state.json')
    assert not state.get('halt') and all(t['status'] == 'valid' for t in state['tasks'].values())
    found = {}
    for iid, task in state['tasks'].items():
        attempt = task['attempts'][-1]
        if attempt['exit_status'] != 'LimitsExceeded':
            continue
        directory = PRIOR / arm / 'swe/rollouts' / iid / attempt['id']
        paths = list(directory.glob('**/' + iid + '.traj.json'))
        path = paths[0] if paths else directory / 'checkpoint.traj.json'
        if limit_reason(read(path)) == 'response_token_limit':
            found[iid] = dict(attempt_id=attempt['id'], trajectory=str(path), sha256=sha(path),
                              previously_resolved=iid in read(PRIOR / arm / 'summary.json')['resolved_ids'])
    assert set(found) == set(C.selected[arm]), found
    return found


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'prepare-claim.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), at=time.time()), f)
    assert read(PRIOR / 'status.json')['phase'] == 'complete'
    assert read(PRIOR / 'cleanup.json')['owned_remaining'] == []
    publication = read(PRIOR / 'reporting-publication.json')
    assert publication['complete'] and publication['revision'] == C.prior_revision
    assert read(PRIOR / 'qualification/ready.json')['passed']
    # The new runner is additive. Verify every pre-existing Python source file against
    # the original qualified deployment archive, not a rewritten readiness hash.
    source_hashes = {}
    archive = Path('/srv/lasr/staging/source.tar.gz')
    assert sha(archive) == read('/srv/lasr/deployment.json')['source_sha256']
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.py') and member.name.startswith(('src/', 'scratch/')):
                import hashlib
                digest = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert sha(Path(member.name)) == digest, member.name
                source_hashes[member.name] = digest
    save(ROOT / 'runtime-source-hashes.json', source_hashes)
    proof = ROOT / 'qualification'
    shutil.copytree(PRIOR / 'qualification/proof', proof / 'inherited-proof')
    for name in ('ready.json', 'publication.json'):
        shutil.copyfile(PRIOR / 'qualification' / name, proof / ('inherited-' + name))
    info = json.loads(subprocess.check_output(['docker', 'info', '--format', '{{json .}}'], text=True))
    assert info['NCPU'] >= 60 and info['MemTotal'] / 2**30 >= 190
    assert shutil.disk_usage(info['DockerRootDir']).free / 2**30 > 50
    assert subprocess.run(['pgrep', '-f', '^.*python.* -m src.infra.endpoints.tinker_server$'],
                          capture_output=True).returncode == 1, 'Existing native sampler is active'
    selection, deltas, images_verified = {}, {}, []
    for arm in C.targets:
        selection[arm] = prior_selected(arm)
        r = ROOT / arm
        cfg0 = OmegaConf.load(PRIOR / arm / 'config.yaml')
        cfg = revised_config(cfg0, arm)
        deltas[arm] = verify_delta(cfg0, cfg)
        meta = r / 'swe/metadata'
        meta.mkdir(parents=True)
        OmegaConf.save(cfg, r / 'config.yaml')
        images = read(PRIOR / arm / 'swe/metadata/images.json')
        rows = {x['instance_id']: x for x in read(PRIOR / arm / 'swe/metadata/swebench_lite_test.json')}
        assert read(PRIOR / arm / 'swe/metadata/manifest.json')['target'] == C.targets[arm]
        for name in ('images.json', 'swebench_lite_test.json'):
            shutil.copyfile(PRIOR / arm / 'swe/metadata' / name, meta / name)
        for iid in C.selected[arm]:
            image = images[iid]
            actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image['digest']], text=True))[0]
            assert image['digest'] in actual['RepoDigests']
            named = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image['name']], text=True))[0]
            assert named['Id'] == actual['Id']
            probe = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'bash', image['digest'],
                                    '-lc', 'cd /testbed && git rev-parse HEAD && test -d /opt/miniconda3'],
                                   capture_output=True, text=True, check=True, timeout=90)
            assert probe.stdout.strip() == rows[iid]['base_commit']
            images_verified.append(dict(arm=arm, instance_id=iid, digest=image['digest'], base_commit=probe.stdout.strip()))
        save(meta / 'manifest.json', dict(campaign=cfg.campaign, target=C.targets[arm], protocol=cfg.protocol,
              selected_ids=list(C.selected[arm]), config=OmegaConf.to_container(cfg, resolve=True),
              source=read(ROOT / 'deployment.json'), prior_revision=C.prior_revision,
              intervention='Fresh rerun; response cap 16384 -> 65536; all other model and task settings unchanged'))
        save(meta / 'state.json', dict(tasks={i:dict(status='pending', attempts=[]) for i in C.selected[arm]},
                                      pods=[], deadline=None, phase='inference', halt=None))
        save(r / 'status.json', dict(phase='prepared', at=time.time()))
    save(ROOT / 'selection.json', selection)
    save(ROOT / 'config-delta.json', deltas)
    save(proof / 'images.json', images_verified)
    save(ROOT / 'prior-publication.json', publication)
    save(ROOT / 'authorization.json', dict(at=time.time(), instruction='User explicitly requested all four control and two DA15 response-limit cases rerun from scratch in parallel with quadrupled single-response cap, keeping everything else unchanged. Original outcomes stay immutable. Prior removal of SWE spend guards remains in force.', inference_budget='unlimited; new rerun-only ledgers, no deletion or reuse of original ledgers'))
    save(proof / 'ready.json', dict(passed=True, at=time.time(), source_files_verified=len(source_hashes),
                                  images_probed=len(images_verified), inherited_qualification=read(proof / 'inherited-publication.json')))
    save(ROOT / 'status.json', dict(phase='prepared', at=time.time()))


def grade():
    from scratch.gptoss_swe.run import grade_predictions
    for arm, target in C.targets.items():
        r = ROOT / arm
        state = read(r / 'swe/metadata/state.json')
        assert not any(t['status'] == 'running' for t in state['tasks'].values())
        preds = {i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status'] == 'valid'}
        save(r / 'status.json', dict(phase='grading', at=time.time()))
        result = grade_predictions(OmegaConf.load(r / 'config.yaml'), r / 'swe', preds, str(C.campaign) + '-' + arm) if preds else dict(n_graded=0, n_resolved=0, resolved_ids=[])
        save(r / 'summary.json', dict(**result, target=target, n_selected=len(C.selected[arm]), n_excluded=len(C.selected[arm])-len(preds)))
        save(r / 'status.json', dict(phase='graded', at=time.time()))


def publish():
    from huggingface_hub import HfApi, hf_hub_download
    summary = engine.status()
    assert all(x['running'] == x['waiting'] == x['reservations'] == 0 for x in summary.values())
    out = ROOT / 'package'
    for name in ('results', 'metadata', 'rollouts'):
        (out / name).mkdir(parents=True, exist_ok=True)
    needles = [v.encode() for k,v in os.environ.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    for arm in C.targets:
        with tarfile.open(out / 'rollouts' / (arm + '.tar.gz'), 'w:gz') as tar:
            for path in sorted((ROOT / arm).rglob('*')):
                if path.is_file():
                    assert not any(n in path.read_bytes() for n in needles), str(path)
                    tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        save(out / 'results' / (arm + '.json'), read(ROOT / arm / 'summary.json'))
    save(out / 'results/results.json', dict(arms=summary, prior_selected=read(ROOT/'selection.json'), scope='Outcome-selected 64k response diagnostic rerun. Do not substitute these outcomes into the original 20-task score. Stochastic reruns cannot establish a pure causal effect of the cap change.'))
    lines = ['# Six-case response-limit rerun', '', '| Metric | Control (4) | DA15 (2) |', '|---|---:|---:|']
    for key in ('success','graded','failed','submitted','awaiting_grading','running','waiting','infrastructure_excluded','cost_usd'):
        lines.append('| '+key+' | '+' | '.join(str(summary[a][key]) for a in C.targets)+' |')
    for reason in ('context_limit','response_token_limit','step_limit','task_token_limit','other_limit'):
        lines.append('| '+reason+' | '+' | '.join(str(summary[a]['limits'].get(reason,0)) for a in C.targets)+' |')
    (out/'results/results.md').write_text('\n'.join(lines)+'\n')
    for path in ROOT.iterdir():
        if path.is_file() and path.name != 'publication.json':
            shutil.copyfile(path, out/'metadata'/path.name)
    shutil.copytree(ROOT/'qualification', out/'metadata/qualification', dirs_exist_ok=True)
    save(out/'metadata/run_meta.json', dict(source=read(ROOT/'deployment.json'), targets=dict(C.targets),
         configs={a:read(ROOT/a/'swe/metadata/manifest.json')['config'] for a in C.targets}, selection=read(ROOT/'selection.json'), config_delta=read(ROOT/'config-delta.json')))
    card = ['---','license: mit','tags: [eval-run, "eval:swebench_mini", "model:gptoss120b", "mode:think", swebench-lite, diagnostic-subset]','---',
        '# GPT-OSS-120B six-case response-limit intervention', '', 'experiment: fresh reruns of all six response-limit endings in the prior twenty-task smoke; outcome-selected diagnostic, not a full benchmark or an updated 20-task score.',
        'date_generated: 2026-10-09', 'constitution: none', 'source_repo: Matthew-Bozoukov/Lessons_from_constituitional_AFT@'+read(ROOT/'deployment.json')['git_commit'],
        'models: exact control and DA15 Tinker checkpoint paths in metadata/run_meta.json. Base was not rerun.',
        'generation_config: single-response cap 65536 (was 16384); medium reasoning, Harmony0.0.8,T1/top_p1/top_k disabled,131072context,262144generated/task,500steps,7200srequesttimeout,2requestattempts,3infraattempts. Original rendering date retained. Remaining context or task tokens can reduce actual response allowance.',
        'schema: rollouts archives contain raw requests/responses and grades; results contain official outcomes; metadata contains source, exact configuration delta, selection, qualification and authorization.',
        'provenance: python -m scratch.gptoss_swe.response_quad prepare; run. Original artifacts retained at '+str(C.prior_revision)+'.',
        'All six fresh attempts were authorized explicitly. New accounting is rerun-only; original ledgers remain unchanged. Costs are conservative estimates, not invoices; CPU/transfer are separate.',
        'Selection was based on previous failures, and sampling is stochastic. Do not splice these outcomes into the original score or attribute all changes solely to the cap. No JSON coaching, output repair or model-outcome retries.', '', *lines[2:]]
    (out/'README.md').write_text('\n'.join(card)+'\n')
    hashes = {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'file-hashes.json'}
    save(out/'metadata/file-hashes.json', hashes)
    hashes['metadata/file-hashes.json'] = sha(out/'metadata/file-hashes.json')
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.create_repo(str(C.hf_repo), repo_type='dataset', exist_ok=True)
    revision = api.upload_folder(repo_id=str(C.hf_repo), repo_type='dataset', folder_path=out, commit_message='Publish six authorized 64k response reruns; preserve original smoke scores').oid
    save(ROOT/'publication.json', dict(repo=C.hf_repo, revision=revision, complete=False, sha256=hashes))
    for name,digest in hashes.items():
        assert sha(hf_hub_download(str(C.hf_repo),name,repo_type='dataset',revision=revision,token=os.environ['HF_TOKEN'])) == digest
    save(ROOT/'publication.json', dict(repo=C.hf_repo, revision=revision, complete=True, sha256=hashes))


def run():
    assert read(ROOT/'qualification/ready.json')['passed']
    with (ROOT/'owner-started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), at=time.time(), source=read(ROOT/'deployment.json')), f)
    try:
        save(ROOT/'status.json', dict(phase='inference', at=time.time()))
        errors = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {a:pool.submit(engine.arm_inference,a) for a in C.targets}
            for arm,future in futures.items():
                try: future.result()
                except Exception as exc: errors[arm] = repr(exc)
        save(ROOT/'inference-exits.json', errors)
        assert not errors, errors
        save(ROOT/'status.json', dict(phase='grading', at=time.time()))
        grade()
        owned = []
        for arm in C.targets:
            owned += subprocess.check_output(['docker','ps','-aq','--filter','label=lasr_campaign='+str(C.campaign)+'-'+arm],text=True).split()
        assert not owned, owned
        save(ROOT/'cleanup.json', dict(at=time.time(), owned_remaining=[]))
        save(ROOT/'status.json', dict(phase='publishing', at=time.time()))
        publish()
        save(ROOT/'status.json', dict(phase='complete', at=time.time()))
    except BaseException as exc:
        save(ROOT/'status.json', dict(phase='held', at=time.time(), error=repr(exc)))
        raise


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('/srv/lasr/credentials.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','status','publish'])
    args = parser.parse_args()
    configure()
    if args.action == 'status': print(json.dumps(engine.status(),indent=2))
    else: globals()[args.action]()
