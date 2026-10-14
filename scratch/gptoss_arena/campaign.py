# ABOUTME: Frozen native three-arm Arena answer generation; never invokes a judge.
# ABOUTME: Run: CPU environment python -m scratch.gptoss_arena.campaign prepare|preflight|load|smoke|run|publish|all
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from types import SimpleNamespace

from omegaconf import OmegaConf
from scratch.gptoss_mmlu.campaign import TARGETS, CPU, SHIM, save

ROOT = Path('/work/output/gptoss_arena_20261014')
PORTS = dict(base=18905, control=18915, da15=18925)


def config():
    return OmegaConf.merge(OmegaConf.load('configs/eval/arena_hard.yaml'),
                           OmegaConf.load('scratch/gptoss_arena/protocol.yaml'))


def key(arm):
    from src.infra.endpoints.tinker import sampler_name
    from src.naming import undated
    return undated(sampler_name(TARGETS[arm]))



def comparison_config():
    from src.eval.capabilities.arena_hard.runner import register
    cfg = config()
    arms = []
    for arm, target in TARGETS.items():
        arms = register(arms, key(arm), target, 'target', cfg)
    cfg.arms = arms
    return cfg


def prepare():
    from src.utils import read_jsonl
    cfg = config()
    question_file = Path(cfg.vendor_dir) / 'data' / cfg.bench_name / 'question.jsonl'
    questions = read_jsonl(question_file)
    assert len(questions) == len({q['uid'] for q in questions}) == 750
    assert sum(q['category'] == 'hard_prompt' for q in questions) == 500
    assert sum(q['category'] == 'creative_writing' for q in questions) == 250
    save(ROOT / 'targets.json', TARGETS)
    save(ROOT / 'questions.json', questions)
    OmegaConf.save(cfg, ROOT / 'config.yaml')
    save(ROOT / 'prepared.json', dict(at=time.time(), questions=750,
         ceiling_usd=float(cfg.campaign.ceiling_usd), native_protocol=OmegaConf.to_container(cfg.generation.native_protocol, resolve=True),
         dataset_sha256=hashlib.sha256(question_file.read_bytes()).hexdigest(),
         authorization='32,768 output / 131,072 context; medium native; 64 per arm; $70 aggregate; smoke below $0.50; judging deferred'))


def preflight():
    subprocess.run([SHIM, '-m', 'scratch.gptoss_arena.preflight'], check=True)


def verify_source(*, paid=False):
    source = json.loads((ROOT / 'source.json').read_text())
    assert not paid or source['clean'], 'Paid inference requires committed source'
    for path, digest in source['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    import src.utils
    src.utils.git_sha = lambda: source['sha']
    return source


@contextlib.contextmanager
def bridge(phase, arm):
    import requests
    env = {**os.environ, 'MMLU_PHASE': phase, 'MMLU_ARM': arm,
           'MMLU_INSTANCE': uuid.uuid4().hex, 'PORT': str(PORTS[arm]),
           'TINKER_DEFAULT_TOP_K': '-1', 'TINKER_DEFAULT_TOP_P': '1.0'}
    directory = ROOT / phase / arm
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'bridge.log').open('w') as log:
        proc = subprocess.Popen([SHIM, '-m', 'scratch.gptoss_arena.bridge'], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f'{arm} bridge exited {proc.returncode}')
                try:
                    response = requests.get(f'http://127.0.0.1:{PORTS[arm]}/v1/models',
                        headers={'Authorization': 'Bearer ' + os.environ['TINKER_API_KEY']}, timeout=2)
                    if response.status_code == 200:
                        identity = response.json()['data'][0]
                        assert identity['checkpoint'] == TARGETS[arm]
                        assert identity['instance_id'] == env['MMLU_INSTANCE']
                        assert identity['reasoning_effort'] == 'medium' and identity['context_window'] == 131072
                        assert identity['sampling_model'] == 'openai/gpt-oss-120b:peft:131072'
                        assert identity['renderer_date'] == '2026-10-14'
                        save(directory / 'identity.json', identity)
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Bridge readiness')
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            save(directory / 'bridge-exit.json', dict(at=time.time(), code=proc.returncode))


def arm_run(phase, arm):
    from src.eval.capabilities.arena_hard import runner as r
    from src.utils import read_jsonl
    verify_source(paid=phase != 'load')
    cfg = config()
    out = ROOT / ('load-answers' if phase == 'load' else 'arms') / arm
    out.mkdir(parents=True, exist_ok=True)
    spec = SimpleNamespace(hf_path=TARGETS[arm], revision=TARGETS[arm], base_revision=None,
        mode='think', base_model='openai/gpt-oss-120b', model_key=key(arm), answers=None)
    target = SimpleNamespace(spec=spec, model_name='openai/gpt-oss-120b',
        base_url=f'http://127.0.0.1:{PORTS[arm]}/v1', api_key=os.environ['TINKER_API_KEY'])
    with bridge(phase, arm):
        if phase in ('load', 'smoke'):
            r.isolate_harness(cfg, out)
            cfg.target_identity = dict(target=spec.hf_path, revision=spec.revision,
                                       base_revision=None, mode='think')
            cfg.arms = r.register([], key(arm), TARGETS[arm], 'target', cfg)
            cfg.output_dir = str(out)
            cfg_path = out / 'metadata/arena_hard_config.yaml'
            cfg_path.parent.mkdir(exist_ok=True)
            OmegaConf.save(cfg, cfg_path)
            r.arena_hard_gen.main(config=str(cfg_path), arm=key(arm), served_model=target.model_name,
                endpoint=target.base_url, api_key=target.api_key, stage=64 if phase == 'load' else 1, creative=0)
            answer_file = r.bench_answers_dir(cfg) / f'{key(arm)}.jsonl'
            records = read_jsonl(answer_file)
            save(ROOT / phase / arm / 'answers-receipt.json', dict(count=len(records),
                 sha256=hashlib.sha256(answer_file.read_bytes()).hexdigest()))
            if phase == 'smoke':
                # Immutable smoke copy; the same canonical cache is reused by the full run.
                shutil.copy2(answer_file, ROOT / phase / arm / 'answers.jsonl')
        else:
            summary = r.run(target, cfg, out, reference=TARGETS['base'])
            save(out / 'metadata/run_summary.json', summary)
    save(ROOT / phase / arm / 'finished.json', dict(at=time.time()))


def phase_run(phase):
    from scratch.gptoss_mmlu.bridge import working_memory
    from src.utils import read_jsonl
    with (ROOT / f'{phase}-started.json').open('x') as f:
        json.dump(dict(at=time.time(), pid=os.getpid()), f)
    children, logs, samples = {}, [], []
    released = False
    try:
        for arm in TARGETS:
            directory = ROOT / phase / arm
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / 'owner.log').open('w')
            logs.append(log)
            children[arm] = subprocess.Popen([CPU, '-m', 'scratch.gptoss_arena.campaign', 'arm', phase, arm], stdout=log, stderr=subprocess.STDOUT)
        while any(p.poll() is None for p in children.values()):
            samples.append(dict(at=time.time(), working_memory=working_memory()))
            if phase == 'load' and not released:
                peaks = []
                for a in TARGETS:
                    try:
                        peaks.append(json.loads((ROOT / f'load-{a}-active.json').read_text())['peak'])
                    except (FileNotFoundError, json.JSONDecodeError):
                        peaks.append(0)
                if peaks == [64, 64, 64]:
                    (ROOT / 'load-release').write_text('192 concurrent synthetic requests admitted')
                    released = True
            save(ROOT / f'{phase}-progress.json', dict(at=time.time(), codes={a: p.poll() for a,p in children.items()}, working_memory=working_memory()))
            time.sleep(1)
        codes = {a: p.returncode for a,p in children.items()}
        save(ROOT / f'{phase}-resources.json', samples)
        assert all(c == 0 for c in codes.values()), codes
        ledger = json.loads((ROOT / ('load-budget.json' if phase == 'load' else 'budget.json')).read_text())
        rows = list(ledger['requests'].values())
        stage = [r for r in rows if r['phase'] == phase]
        assert len(stage) == {'load':192, 'smoke':3, 'run':2247}[phase], len(stage)
        assert all(r['state'] == 'completed' for r in rows)
        if phase == 'load':
            assert released and max(s['working_memory'] for s in samples) < 3.5 * 2**30
            events = dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
            assert int(events['oom_kill']) == 0
        for arm in TARGETS:
            if phase == 'run':
                path = ROOT / 'arms' / arm / 'rollouts/answers.jsonl'
            else:
                path = ROOT / ('load-answers' if phase == 'load' else 'arms') / arm / 'arena_hard_harness/data/arena-hard-v2.0/model_answer' / f'{key(arm)}.jsonl'
            records = read_jsonl(path)
            assert len(records) == {'load':64, 'smoke':1, 'run':750}[phase]
            for rec in records:
                g = rec['generation_record']
                assert g['native_metadata']['checkpoint'] == TARGETS[arm]
                assert g['native_metadata']['sampling']['temperature'] == 0
                assert g['native_metadata']['sampling']['top_k'] == -1
                assert g['max_tokens'] == 32768 and g['finish_reason'] in ('stop','length')
                if phase == 'load':
                    assert g['usage']['completion_tokens'] == 32768 and g['answer'] == 'A' and g['think']
            if phase == 'run':
                smoke = read_jsonl(ROOT / 'smoke' / arm / 'answers.jsonl')[0]
                assert next(r for r in records if r['uid'] == smoke['uid']) == smoke, 'Smoke changed or regenerated'
        smoke_cost = sum(r['upper_usd'] for r in rows if r['phase'] == 'smoke')
        assert smoke_cost < .49
        save(ROOT / ('load-qualification.json' if phase == 'load' else f'{phase}-complete.json'),
             dict(at=time.time(), requests=len(stage), codes=codes, smoke_usd=smoke_cost,
                  accounted_usd=sum(r['upper_usd'] for r in rows), peak_working_memory=max(s['working_memory'] for s in samples)))
    finally:
        for log in logs:
            log.close()


def publish():
    from src.utils import read_jsonl
    from src.eval.capabilities.arena_hard.runner import validate_generation_protocols
    from src.infra.huggingface import hf_api, hf_download, push_run_dir
    from src.eval.layout import assert_layout, run_tags
    from src.eval.run_eval import _card_fields
    from src.naming import eval_name
    source = verify_source()
    cfg = comparison_config()
    ledger = json.loads((ROOT / 'budget.json').read_text())
    entries = list(ledger['requests'].values())
    assert len(entries) == 2250 and all(e['state'] == 'completed' for e in entries)
    assert sum(e['upper_usd'] for e in entries) <= 70
    runs = [dict(model_key=key(a), out_dir=str(ROOT/'arms'/a), mode='think') for a in TARGETS]
    save(ROOT/'protocol-validation.json', validate_generation_protocols(runs, cfg))
    results, published = {}, {}
    for arm in TARGETS:
        out = ROOT/'arms'/arm
        records = read_jsonl(out/'rollouts/answers.jsonl')
        assert len(records) == len({r['uid'] for r in records}) == 750
        own = [e for e in entries if e['checkpoint'] == TARGETS[arm]]
        assert len(own) == 750
        assert sum(e['completion_tokens'] for e in own) == sum(r['generation_record']['usage']['completion_tokens'] for r in records)
        results[arm] = dict(answers=750, empty_final=sum(not r['generation_record']['answer'].strip() for r in records),
            truncated=sum(r['generation_record']['finish_reason']=='length' for r in records),
            accounted_usd=sum(e['upper_usd'] for e in own), judging='deferred')
        meta = out/'metadata'
        for name in ('source.json','source.tar.gz','prepared.json','config.yaml','token-preflight.json',
                     'load-qualification.json','smoke-complete.json','run-complete.json','targets.json',
                     'budget.json','protocol-validation.json','driver-packages.txt','shim-packages.txt'):
            shutil.copy2(ROOT/name, meta/name)
        with tarfile.open(meta/'native-evidence.tar.gz','w:gz') as tar:
            for phase in ('smoke','run'):
                tar.add(ROOT/phase/arm, arcname=f'{phase}/{arm}')
        save(meta/'run_meta.json', dict(git_sha=source['sha'], target=TARGETS[arm], target_revision=TARGETS[arm],
            model_key=key(arm), base_model='openai/gpt-oss-120b', base_model_revision=None, mode='think', config=OmegaConf.to_container(cfg, resolve=True),
            summary=results[arm], command='python -m scratch.gptoss_arena.campaign all'))
        fields = _card_fields('arena_hard', cfg, 'python -m scratch.gptoss_arena.campaign all',
            experiment=f'GPT-OSS Arena native answers: {arm}; 750 prompts; no judging yet',
            models=TARGETS[arm], source_revision=source['sha'])
        repo = 'dougalldeepmind/' + eval_name('arena_hard',key(arm))
        assert_layout(out)
        push_run_dir(out,repo,fields,front_matter={'tags':run_tags('arena_hard',key(arm),'think')})
        revision = hf_api().repo_info(repo,repo_type='dataset').sha
        hashes = {}
        for path in out.rglob('*'):
            if path.is_file() and '.cache' not in path.parts:
                name = path.relative_to(out).as_posix()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                fetched = hf_download(repo,name,repo_type='dataset',revision=revision)
                assert hashlib.sha256(Path(fetched).read_bytes()).hexdigest() == digest, name
                hashes[name] = digest
        published[arm] = dict(repo=repo,revision=revision,sha256=hashes,url=f'https://huggingface.co/datasets/{repo}/tree/{revision}')
        save(ROOT/f'publication-{arm}.json',published[arm])
    save(ROOT/'publication.json',dict(complete=True,arms=published))
    save(ROOT/'complete.json',dict(at=time.time(),arms=results,accounted_usd=sum(e['upper_usd'] for e in entries),reservations=0,judging='deferred'))


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'arm':
        arm_run(sys.argv[2],sys.argv[3])
    elif command in ('all', 'inference'):
        try:
            if command == 'all':
                phase_run('load')
                phase_run('smoke')
            else:
                assert (ROOT/'load-qualification.json').exists() and (ROOT/'smoke-complete.json').exists()
            phase_run('run')
            publish()
        except BaseException as exc:
            save(ROOT/'failure.json',dict(at=time.time(),error=type(exc).__name__,message=str(exc)))
            raise
    elif command in ('prepare','preflight','publish'):
        globals()[command]()
    else:
        phase_run(command)
