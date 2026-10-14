# ABOUTME: Audit and score the completed subscription judge campaign without model calls.
# ABOUTME: Run: uv run python -m scratch.gptoss_arena_judge.finish score|publish|wait
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict

from omegaconf import OmegaConf

from scratch.gptoss_arena_judge.campaign import ROOT, REPO, cfg, digest, read, rows, save, validate_events, verdict
from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments, evaluate_arm

OUT = ROOT / 'publication'


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')


def assemble(jobs):
    grouped = defaultdict(dict)
    usage = defaultdict(int)
    transcripts = []
    for job in jobs:
        directory = ROOT/'jobs'/job['id']
        assert not (directory/'failure.json').exists(), job['id']
        result = read(directory/'completed.json')
        identity = read(directory/'identity.json')
        assert hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == job['id']
        assert result['request_hash'] == job['id']
        assert digest(directory/'final.txt') == result['final_sha256']
        assert (directory/'prompt.txt').read_text(encoding='utf-8') == identity['prompt']
        events = validate_events(rows(directory/'events.jsonl'))
        assert events[0]['usage'] == result['usage']
        answer = read(directory/'final.txt')
        assert verdict(answer['verdict']) == result['score']
        assert result['model'] == 'gpt-6-luna' and result['reasoning_effort'] == 'low'
        grouped[(job['pair'],job['uid'])][job['order']] = dict(score=result['score'], judgment=answer, usage=result['usage'], job_id=job['id'])
        for key, value in result['usage'].items():
            if isinstance(value, (int,float)): usage[key] += value
        transcripts.append(dict(**job, system=identity['instructions'], prompt=identity['prompt'], response=answer, usage=result['usage']))
    records = defaultdict(list)
    for (pair,uid), games in grouped.items():
        assert set(games) == {0,1}
        job = next(t for t in transcripts if t['pair']==pair and t['uid']==uid)
        records[pair].append(dict(uid=uid,category=job['category'],model=job['target'],baseline=job['baseline'],games=[games[0],games[1]]))
    return records, transcripts, dict(usage)


def score():
    assert read(ROOT/'full-complete.json')['completed'] == 4500
    jobs = read(ROOT/'jobs.json')
    assert len(jobs)==len({j['id'] for j in jobs})==4500
    for folder in ('rollouts','results','metadata'): (OUT/folder).mkdir(parents=True,exist_ok=True)
    inputs = {}
    sources = read(ROOT/'sources.json')
    for arm, source in sources['arms'].items():
        for name in ('rollouts/answers.jsonl','metadata/generation_protocol.json'):
            assert digest(ROOT/'inputs'/arm/name)==source['sha256'][name]
        inputs[arm] = {r['uid']:r for r in rows(ROOT/'inputs'/arm/'rollouts/answers.jsonl')}
    records, transcripts, usage = assemble(jobs)
    jsonl(OUT/'rollouts/judgments.jsonl', transcripts)
    results = {}
    for pair, recs in records.items():
        assert len(recs)==750
        jsonl(OUT/f'results/{pair}.jsonl',recs)
        target, baseline = recs[0]['model'], recs[0]['baseline']
        battles = battles_from_judgments(recs)
        assert len(battles)==1500
        results[pair] = {}
        for category, count in (('hard_prompt',1000),('creative_writing',500)):
            subset=[b for b in battles if b['category']==category]
            assert len(subset)==count
            block=evaluate_arm(subset,{u:r['metadata'] for u,r in inputs[target].items()},
                               {u:r['metadata'] for u,r in inputs[baseline].items()},
                               rounds=int(cfg().bootstrap_rounds),seed=int(cfg().seed))
            # The Luna variant has no validated non-inferiority acceptance gate.
            block.pop('passes'); block.pop('threshold')
            weight=sum(3 if b['decisive'] else 1 for b in subset)
            block['decisive_x3_weighted_score_diagnostic']=sum(b['score']*(3 if b['decisive'] else 1) for b in subset)/weight
            results[pair][category]=block
        save(ROOT/'scoring-progress.json',dict(pair=pair,at=time.time()))
    result=dict(judge='gpt-6-luna',reasoning_effort='low',backend='codex-exec-chatgpt',
                judgments=4500,questions_per_pair=750,orders=2,results=results,usage=usage,
                api_calls=0,subscription_usage_not_dollar_invoice=True,
                caveats=['Codex Luna low judge variant, not official leaderboard-equivalent.',
                         'Visible final answers only; empty and truncated generations retained.',
                         '95% prompt-cluster bootstrap; ordinary verdicts equal weight, ties half credit.',
                         'Style-adjusted estimates condition on observed output formatting; empty-answer failures remain.',
                         'Separate decisive x3 diagnostic has no claimed upstream leaderboard confidence interval.',
                         'Twelve unstructured qualification calls excluded from scored results; structured panel reused.'])
    save(OUT/'results/results.json',result)
    table=['# GPT-OSS Arena-Hard: Codex Luna low judging','',
           '| Target vs reference | Slice | Raw tie-adjusted % | Style-controlled % (95% CI) |',
           '|---|---|---:|---:|']
    for pair, cats in results.items():
        for cat, block in cats.items():
            c=block['controlled'];u=block['uncontrolled']
            table.append(f"| {pair} | {cat} | {100*u['mean']:.2f} | {100*c['mean']:.2f} [{100*c['ci_lower']:.2f}, {100*c['ci_upper']:.2f}] |")
    table.extend(['','Above 50% favors the target; ties count half. Hard is primary; creative is secondary.','']+result['caveats'])
    (OUT/'results/results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    save(ROOT/'scoring-complete.json',dict(at=time.time(),judgments=4500,usage=usage))
    return result


def publish():
    from src.infra.huggingface import hf_api, hf_download, push_run_dir
    from src.eval.layout import assert_layout, run_tags
    from src.eval.run_eval import _card_fields
    from src.naming import eval_name
    assert (ROOT/'scoring-complete.json').exists()
    meta=OUT/'metadata'
    for name in ('sources.json','prepared.json','protocol.yaml','judge-source.json','full-source-sha.txt',
                 'panel-complete.json','full-complete.json','status.json','scoring-complete.json',
                 'prompt-context-inspection.json','prompt-context-clean.json'):
        if (ROOT/name).exists(): shutil.copy2(ROOT/name,meta/name)
    for category in ('hard_prompt','creative_writing'):
        shutil.copy2(ROOT/f'{category}-instructions.txt',meta/f'{category}-instructions.txt')
    with tarfile.open(meta/'raw-judge-evidence.tar.gz','w:gz') as tar:
        for job in read(ROOT/'jobs.json'):
            tar.add(ROOT/'jobs'/job['id'],arcname=f"jobs/{job['id']}")
        tar.add(ROOT/'qualification-unstructured',arcname='qualification-unstructured')
        for archive in ROOT.glob('transport-validation-*'):
            if archive.is_dir(): tar.add(archive,arcname=archive.name)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    subprocess.run(['git','archive','--format=tar.gz',f'--output={meta / "judge-source.tar.gz"}',source,
                    'scratch/gptoss_arena_judge','src/eval/capabilities/arena_hard/arena_hard_stats.py'],cwd=REPO,check=True)
    configuration=cfg()
    save(meta/'run_meta.json',dict(git_sha=source,judge_source=(ROOT/'full-source-sha.txt').read_text().strip(),
         config=OmegaConf.to_container(configuration),generation_sources=read(ROOT/'sources.json'),
         mode='think',judge_variant='codex-luna-low',api_calls=0))
    fields=_card_fields('arena_hard',configuration,'python -m scratch.gptoss_arena_judge.finish wait',
         experiment='Three matched GPT-OSS pairwise comparisons using subscription Codex Luna low; unofficial judge variant',
         models='GPT-OSS Base, Control, DA15; judge gpt-6-luna low',source_revision=source)
    fields['generation_config']=json.dumps(OmegaConf.to_container(configuration))
    repo='dougalldeepmind/'+eval_name('arena_hard','gptoss120b-three-pairs',variant='luna-low')
    assert_layout(OUT)
    push_run_dir(OUT,repo,fields,front_matter={'tags':run_tags('arena_hard','gptoss120b-three-pairs','think',variant='luna-low',pooled=True)})
    revision=hf_api().repo_info(repo,repo_type='dataset').sha
    hashes={}
    for path in OUT.rglob('*'):
        if path.is_file() and '.cache' not in path.parts:
            name=path.relative_to(OUT).as_posix()
            downloaded=hf_download(repo,name,repo_type='dataset',revision=revision)
            assert digest(downloaded)==digest(path),name
            hashes[name]=digest(path)
    save(ROOT/'publication.json',dict(complete=True,repo=repo,revision=revision,sha256=hashes,
         url=f'https://huggingface.co/datasets/{repo}/tree/{revision}'))


def wait():
    import psutil
    with (ROOT/'finish-owner.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=psutil.Process().pid,birth=psutil.Process().create_time()),f)
    while not (ROOT/'full-complete.json').exists():
        assert not (ROOT/'admission-hold.json').exists(), 'Judging held; no automatic retries'
        owner=read(ROOT/'full-owner.json')
        assert psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['birth'])<.1,'Owner exited without completion'
        time.sleep(20)
    score()
    publish()


if __name__=='__main__':
    try:
        {'score':score,'publish':publish,'wait':wait}[sys.argv[1]]()
    except BaseException as exc:
        save(ROOT/'finish-failure.json',dict(at=time.time(),error=type(exc).__name__,message=str(exc)))
        raise
