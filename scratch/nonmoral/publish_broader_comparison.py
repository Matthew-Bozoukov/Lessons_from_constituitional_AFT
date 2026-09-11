# ABOUTME: Publish the frozen four-checkpoint comparison through shared HF contracts.
# ABOUTME: Verify public eval bytes, local backups, pod absence and bounded cost first.
import hashlib
import json
from pathlib import Path
import subprocess
import re
import statistics
import argparse
from datetime import date

from huggingface_hub import HfApi
from src.infra.huggingface import hf_download, hf_org, push_run_dir
from src.naming import artifact_name, eval_name
from scratch.nonmoral.publish_invalid_baseline import scan, secret_values

BASE = Path('output/nonmoral_broader/20260909')


def stakes_dataset_comparison(report, public, code_revision):
    """Measure frozen mixtures and describe the controlled stakes intervention."""
    assert report['status'] == 'matched_stakes_complete'
    protocol = dict(report['protocol'], benchmark='odcv', mode='think',
        scenario_set=hashlib.sha256(json.dumps(report['expected_cells'], sort_keys=True).encode()).hexdigest())
    declarations = [
        ('stakes', 'Loss magnitude', 'Intended numerical loss exposure; perceived stakes were not independently validated.'),
        ('perceived_stakes', 'Measured perceived stakes', 'No independent human or model annotation of perceived stakes.'),
        ('scope', 'Task domains', 'Inherited professional craft tasks; no new scenario selection between arms.'),
        ('reasoning', 'Reasoning intervention', 'Full reasoning and final answers supervised; substantive recommendations held fixed.'),
        ('reasoning_words', 'Mean reasoning words', 'Whitespace-delimited words across all 684 synthetic rows; not tokenizer tokens.'),
        ('answer_words', 'Mean final-answer words', 'Whitespace-delimited words across all 684 synthetic rows; excludes replay.'),
        ('mixture', 'Training mixture', 'Exact row counts and byte-identical replay at identical positions.'),
    ]
    arms, audits, replay, synthetic = [], {}, {}, {}
    design = [dict(label='Matched-stakes construction contract', url=
        f'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/blob/{code_revision}/docs/nonmoral_deliberation/2026-09-10_matched_stakes_results.md')]
    for key in ('low', 'high'):
        model = report['model_pins'][key]; dataset = model['dataset']
        path = Path(hf_download(dataset['repo'], dataset['file'], repo_type='dataset', revision=dataset['revision']))
        raw = path.read_bytes(); lines = raw.splitlines(); rows = [json.loads(x) for x in lines]
        source = 'nonmoral_stakes_' + key
        selected = [r for r in rows if r['source'] == source]
        assert len(rows) == 9968 and len(selected) == 684
        replay[key] = [(i, line) for i, (line, row) in enumerate(zip(lines, rows)) if row['source'] != source]
        synthetic[key] = {r['scenario_id']: r for r in selected}
        assert len(synthetic[key]) == 684
        measurements = []
        for row in selected:
            turns = row['text'].split('<|im_start|>assistant\n'); assert len(turns) == 2
            match = re.fullmatch(r'<think>\s*(.*?)</think>\s*(.*?)<\|im_end\|>\s*', turns[1], re.S)
            assert match and match[1].strip() and match[2].strip()
            measurements.append(dict(scenario_id=row['scenario_id'], trait_id=row['trait_id'],
                reasoning_words=len(match[1].split()), answer_words=len(match[2].split())))
        evidence = [dict(label='Pinned training mixture', url=
            f"https://huggingface.co/datasets/{dataset['repo']}/blob/{dataset['revision']}/{dataset['file']}")]
        def trait(value, basis='design'):
            return dict(value=value, basis=basis, evidence=evidence if basis == 'measured' else design)
        values = dict(stakes=trait('Lower hypothetical numerical losses' if key == 'low' else 'Higher hypothetical numerical losses'),
            perceived_stakes=dict(value=None, basis='unmeasured', evidence=[]),
            scope=trait('684 matched professional craft scenarios: software, documentation, writing, teaching, data and interfaces'),
            reasoning=trait('Full reasoning and final response; same substantive content, numeric stakes changes only'),
            reasoning_words=trait(statistics.mean(m['reasoning_words'] for m in measurements), 'measured'),
            answer_words=trait(statistics.mean(m['answer_words'] for m in measurements), 'measured'),
            mixture=trait('684 synthetic + 9,284 identical replay rows (9,968 total)', 'measured'))
        result = report['arms'][key]
        arms.append(dict(id=key, label=result['label'], dataset=dict(**dataset, row_count=9968,
            subset='Traits describe all 684 synthetic rows; training used the full mixture.'),
            model=dict(repo=model['repo'], revision=model['revision'], base_revision=model['base_revision'], seed=0),
            traits=values, evaluation=dict(**public[key], protocol=protocol, repeats=3, metrics=dict(
                misalignment=dict(value=result['mr_pct'], numerator=result['mr_numerator'], denominator=240,
                    interval=dict(low=result['scenario_mr']['lo'], high=result['scenario_mr']['hi'], method='scenario-level, fixed checkpoint')),
                submission=dict(value=result['submitted_pct'], numerator=result['submitted_numerator'], denominator=240),
                progress=dict(value=result['progress_mean'])))))
        audits[key] = dict(**dataset, sha256=hashlib.sha256(raw).hexdigest(), rows=measurements)
    assert replay['low'] == replay['high'] and len(replay['low']) == 9284
    assert synthetic['low'].keys() == synthetic['high'].keys()
    for key, low in synthetic['low'].items():
        high = synthetic['high'][key]
        assert low['trait_id'] == high['trait_id']
        assert re.sub(r'\d[\d,.]*', '#', low['text']) == re.sub(r'\d[\d,.]*', '#', high['text']), key
    d = report['paired_high_minus_low']['stats']['mr']
    low, high = report['arms']['low'], report['arms']['high']
    document = dict(schema_version=1, title='Matched nonmoral deliberation: low versus high stakes',
        summary=f"Low stakes: {low['mr_numerator']}/240 ({low['mr_pct']:.2f}%) misaligned; high stakes: {high['mr_numerator']}/240 ({high['mr_pct']:.2f}%). High minus low: {d['mean']:+.2f} percentage points, scenario-paired 95% CI [{d['lo']:+.2f}, {d['hi']:+.2f}].",
        limitations=[report['interpretation'],
            'Stakes labels describe relative numerical losses, not validated perceived stakes. This tests exposure to different stakes with substantive reasoning held fixed.',
            'Inherited incomplete prompts and instruction overrides were retained in both arms to isolate the intervention.',
            'The historical 18.25% score used a different evaluation protocol; the same original checkpoint later scored 13.75%. Neither is a new training seed.',
            'Low progress judging resumed six missing verdicts after upstream429 with the same judge/provider; existing verdicts and rollouts were retained.'],
        traits=[dict(id=i, label=l, description=d) for i, l, d in declarations],
        metrics=[dict(id='misalignment', label='Misalignment', unit='%', lower_is_better=True),
                 dict(id='submission', label='Task submission', unit='%', lower_is_better=False),
                 dict(id='progress', label='Mean task progress', unit='/5', lower_is_better=False)],
        arms=arms, contrasts=[dict(baseline='low', arm='high', metric='misalignment', delta=d['mean'],
            interval=dict(low=d['lo'], high=d['hi'], method='paired scenarios, fixed checkpoints'))])
    return document, audits


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def publish_stakes(experiment, *, publish=False):
    """Stage verified comparison data; optionally publish through the shared HF pipeline."""
    from src.infra.runpod import active_pods
    experiment = Path(experiment)
    report = read(experiment/'final_comparison/comparison.json')
    assert report['status'] == 'matched_stakes_complete' and hf_org() == 'dougalldeepmind'
    training = read(experiment/'recovered_training.json')
    continuation = read(experiment/'continuation_status.json')
    assert continuation['phase'] == 'evaluations_complete_pending_report'
    archive = training['local_backup']
    assert archive['verified'] and archive['verified_completed_arms'] == 2
    with Path(archive['archive']).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == archive['sha256']
    states = {k: read(experiment/f'evaluation_{k}/broader_eval_status.json') for k in ('low', 'high')}
    owned = {read(Path(training['recovery_evidence']))['pod']}
    owned.update(s['pod_id'] for s in states.values())
    active = {p['id'] for p in active_pods()}
    assert not owned.intersection(active), 'Owned GPU still exists; finish verified retrieval and teardown first'
    public, checked, budget = {}, [], dict(prior_exposure_usd=training['plan']['project_exposure_before_training_usd'],
        training_gpu_storage_usd=training['estimated_gpu_usd'],
        basis='Conservative reservations and elapsed-rate exposure, not provider invoices; shared-account usage deltas excluded.')
    api = HfApi(token=False)
    for arm, result in report['arms'].items():
        state = states[arm]; backup = state['local_log_backup']
        assert state['termination_verified'] and backup['verified']
        assert hashlib.sha256(Path(backup['path']).read_bytes()).hexdigest() == backup['sha256']
        recovery_path = experiment/f'{arm}_eval_completion_recovery.json'
        recovery = read(recovery_path) if recovery_path.exists() else None
        assert state.get('evaluation_driver_completed') or (recovery and recovery['completed'])
        assert state['target_revision'] == result['target_revision']
        root = Path(result['path'])
        repo = hf_org()+'/'+eval_name('odcv', result['target'].split('/')[-1], date=root.name[:10])
        info = api.dataset_info(repo, files_metadata=True)
        assert not info.private
        if recovery:
            assert (recovery['repo'], recovery['revision'], recovery['target_revision']) == (repo, info.sha, result['target_revision'])
        model = api.model_info(result['target'], revision=result['target_revision'])
        assert not model.private and model.sha == result['target_revision']
        count = 0
        for file in info.siblings:
            if file.rfilename == '.gitattributes':
                continue
            raw = (root/file.rfilename).read_bytes()
            actual = (hashlib.sha256(raw).hexdigest() if file.lfs else
                      hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
            assert actual == (file.lfs.sha256 if file.lfs else file.blob_id), file.rfilename
            checked.append(dict(arm=arm, file=file.rfilename, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            count += file.rfilename.endswith('/messages_record.txt')
        assert count == 240
        assert hashlib.sha256((root/'results/results.json').read_bytes()).hexdigest() == result['result_sha256']
        public[arm] = dict(repo=repo, revision=info.sha)
        ledger = read(experiment/f'evaluation_{arm}/judge_ledger.json')
        charged = sum(x['charged_or_reserved_usd'] for x in ledger)
        assert charged <= 3.5
        budget[arm+'_gpu_storage_usd'] = state['estimated_gpu_and_storage_usd']
        budget[arm+'_judge_usd'] = charged
    budget['total_exposure_usd'] = sum(v for k, v in budget.items() if k.endswith('_usd'))
    assert budget['total_exposure_usd'] <= 300
    assert abs(budget['total_exposure_usd'] - continuation['project_exposure_usd']) < 1e-6
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    frontend, audits = stakes_dataset_comparison(report, public, revision)
    dest = experiment/'comparison_publication'; secrets = secret_values()
    def put(relative, raw):
        scan(raw, str(relative), secrets)
        target = dest/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    def put_json(relative, data):
        put(relative, (json.dumps(data, indent=2)+'\n').encode())
    # Figures remain local: only exact measurements, receipts and text are published.
    for filename in ('comparison.json', 'comparison.md'):
        raw = (experiment/'final_comparison'/filename).read_bytes()
        if filename.endswith('.md'):
            raw = '\n'.join(line for line in raw.decode().splitlines() if not line.startswith('![')).encode()
        put('results/'+filename, raw)
    for name, data in [('dataset_comparison', frontend)]:
        put_json('results/'+name+'.json', data)
    for name, data in [('dataset_property_audit', audits), ('public_sources', public),
                       ('verified_public_files', checked), ('budget', budget),
                       ('owned_pod_absence', {p: dict(exists=False) for p in sorted(owned)})]:
        put_json('metadata/'+name+'.json', data)
    files = ['models.json', 'recovered_training.json', 'recovered_training/incident.json',
        'recovered_training/file_sources.json', 'low_eval_completion_recovery.json', 'low_judge_recovery.json',
        'odcv_frozen.yaml', 'continuation_status.json', 'training/watchdog.log']
    for arm in ('low', 'high'):
        files += [f'{arm}_training_meta.json', f'{arm}_eval_plan.yaml',
                  f'evaluation_{arm}/broader_eval_status.json', f'evaluation_{arm}/judge_ledger.json']
    for relative in files:
        put('metadata/'+relative, (experiment/relative).read_bytes())
    for relative in ('scratch/nonmoral/baseline_report.py', 'scratch/nonmoral/publish_broader_comparison.py',
                     'scratch/nonmoral/finish_stakes_eval.py'):
        put('metadata/code/'+Path(relative).name, Path(relative).read_bytes())
    produced = date.today().isoformat()
    put_json('metadata/run_meta.json', dict(git_sha=revision, date_generated=produced,
        command=f'uv run --no-sync python -m scratch.nonmoral.publish_broader_comparison --stakes-experiment {experiment} --publish',
        protocol=report['protocol'], models=report['model_pins']))
    receipt = dict(staged=str(dest.resolve()), public_sources=public, budget=budget)
    if publish:
        fields = dict(experiment=frontend['title'], date_generated=produced,
            constitution='preferences/craft_tensions_09/preferences.md for the synthetic training scenarios; none supplied at evaluation',
            source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+revision,
            models=report['model_pins'], generation_config=report['protocol'],
            schema='results/: exact outcomes and dashboard comparison; metadata/: pins, measured traits, cost and incident evidence; figures local only',
            provenance=f'uv run --no-sync python -m scratch.nonmoral.publish_broader_comparison --stakes-experiment {experiment} --publish',
            limitations=' '.join(frontend['limitations']))
        name = artifact_name('nonmoral-stakes-comparison', date=produced)
        url = push_run_dir(dest, name, fields, private=False, front_matter=dict(pretty_name=frontend['title'],
            tags=['nonmoral-deliberation', 'research-comparison', 'dataset-model-comparison']))
        info = api.dataset_info(hf_org()+'/'+name); assert not info.private
        receipt.update(url=url, repo=info.id, revision=info.sha, public=True)
    (experiment/'comparison_publication_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def dataset_comparison(report, public, code_revision):
    """Describe frozen training populations; never infer semantic traits from eval outcomes."""
    inputs = {
        'nonmoral': ('dougalldeepmind/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture',
                     '6364505df02b0020b030bf379bd42285a14de6a5', 't2_9284_nonmoral_684.jsonl',
                     '0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561', 'nonmoral_deliberation'),
        'broader': ('dougalldeepmind/2026-09-09-nonmoral-broader-7-mix',
                    'f1e61baf643c861920303c7ba1e9844df5f6ed48', 'mixture.jsonl',
                    '0545e014b518fdb9b8b40e37adc0b4a21da01c384553fe60e6a81f87098a2c22', 'nonmoral_broader'),
    }
    traits = [
        ('scope','Task domains','Construction scope; domain taxonomies are not directly comparable.'),
        ('decision','Decision structure','Generation instruction, not a measured response-pattern frequency.'),
        ('stakes','Stakes framing','Intended source of consequences, not an annotated stakes distribution.'),
        ('stakes_distribution','Measured stakes distribution','No common high/low-stakes classifier or human annotation was applied.'),
        ('reasoning_words','Mean reasoning words','Whitespace-delimited words in the think block; all 684 synthetic rows, excluding replay.'),
        ('answer_words','Mean final-answer words','Whitespace-delimited words after the think block; all 684 synthetic rows, excluding replay.'),
        ('system_prompts','Synthetic rows with system prompts','Literal system-role marker count over the 684 selected synthetic rows.'),
        ('mixture','Training mixture','Synthetic/replay counts and byte-for-byte replay equality at the same positions.'),
    ]
    protocol = dict(report['protocol'], benchmark='odcv', scenario_set=hashlib.sha256(
        json.dumps(report['expected_cells'],sort_keys=True).encode()).hexdigest(), mode='think')
    arms, audits, replay = [], {}, {}
    source_url = 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT/blob/'+code_revision+'/'
    for key,(repo,revision,file,expected,source) in inputs.items():
        path=Path(hf_download(repo,file,repo_type='dataset',revision=revision))
        raw=path.read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
        lines=raw.splitlines(); rows=[json.loads(line) for line in lines]
        selected=[r for r in rows if r['source']==source]
        assert len(rows)==9968 and len(selected)==684
        replay[key]=[(i,line) for i,(line,row) in enumerate(zip(lines,rows)) if row['source']!=source]
        measurements=[]
        for row in selected:
            assistant=row['text'].split('<|im_start|>assistant\n')
            assert len(assistant)==2
            match=re.fullmatch(r'<think>\s*(.*?)</think>\s*(.*?)<\|im_end\|>\s*',assistant[1],re.S)
            assert match and match[1].strip() and match[2].strip()
            measurements.append(dict(scenario_id=row['scenario_id'],reasoning_words=len(match[1].split()),
                                     answer_words=len(match[2].split()),system_prompt='<|im_start|>system\n' in row['text']))
        audits[key]=dict(repo=repo,revision=revision,file=file,sha256=expected,population='684 selected synthetic rows',
                        method='Split the sole assistant turn at think tags, then count Python str.split words; these are not tokens.',rows=measurements)
        evidence=[dict(label='Frozen training mixture',url=f'https://huggingface.co/datasets/{repo}/blob/{revision}/{file}')]
        design=[dict(label='Construction rationale',url=source_url+('preferences/craft_tensions_09/rationale.md' if key=='nonmoral' else 'docs/nonmoral_deliberation/research_brief.md'))]
        def value(v,basis='design',e=design): return dict(value=v,basis=basis,evidence=e)
        trait_values=dict(
            scope=value('Professional craft: software, writing, data presentation, teaching, interfaces and related artifacts' if key=='nonmoral' else '12 broad families including everyday planning, cooking, learning, games, spatial reasoning, code and creative work'),
            decision=value('Binary artifact choice; explicitly argue for diverging from the requested choice' if key=='nonmoral' else 'Reasoned single-turn decision; comparison of feasible options, with code and calculations allowed; no required disagreement'),
            stakes=value('Bad craft makes the work fail; harm to people excluded' if key=='nonmoral' else 'Ordinary nonmoral decisions; no systematic high/low-stakes manipulation'),
            stakes_distribution=dict(value=None,basis='unmeasured',evidence=[]),
            reasoning_words=value(statistics.mean(m['reasoning_words'] for m in measurements),'measured',evidence),
            answer_words=value(statistics.mean(m['answer_words'] for m in measurements),'measured',evidence),
            system_prompts=value(sum(m['system_prompt'] for m in measurements),'measured',evidence),
            mixture=value('684 synthetic + 9,284 identical replay rows (9,968 total)','measured',evidence),
        )
        result=report['arms'][key]
        arms.append(dict(id=key,label='Original nonmoral' if key=='nonmoral' else 'Broader nonmoral',
                         dataset=dict(repo=repo,revision=revision,file=file,row_count=9968,subset='Traits describe the 684 synthetic rows; model trained on the full mixture.'),
                         model=dict(repo=result['target'],revision=result['target_revision'],base_revision=result['base_revision'],seed=0),
                         traits=trait_values,evaluation=dict(repo=public[key]['repo'],revision=public[key]['revision'],protocol=protocol,repeats=3,metrics=dict(
                             misalignment=dict(value=result['mr_pct'],numerator=result['mr_numerator'],denominator=240,
                                 interval=dict(low=result['scenario_mr']['lo'],high=result['scenario_mr']['hi'],method='scenario-level, fixed checkpoint')),
                             submission=dict(value=result['submitted_numerator']/240*100,numerator=result['submitted_numerator'],denominator=240),
                             progress=dict(value=result['progress_mean'])))))
    assert replay['nonmoral']==replay['broader'] and len(replay['nonmoral'])==9284
    stats=report['paired_nonmoral_minus']['broader']['stats']['mr']
    document=dict(schema_version=1,title='Broader tasks did not improve nonmoral alignment',
        summary='Changing the synthetic training data from professional craft to broader nonmoral decisions increased ODCV misalignment from 33/240 to 76/240 in these checkpoints.',
        limitations=[report['interpretation'],
            'Properties changed together. These observations do not identify which property caused the difference.',
            'Stakes magnitude was not measured or controlled in either corpus. The separate low/high-stakes experiment has no trained-model result.',
            'The original checkpoint previously scored 18.25% under a different evaluation protocol. Its matched-protocol score here is 13.75%.',
            'Dataset properties refer to the 684 selected synthetic rows, not all accepted source candidates or the full replay mixture.',
            'Final model and eval are preserved; the training backup was interrupted externally, leaving some optimizer/checkpoint files missing.'],
        traits=[dict(id=i,label=l,description=d) for i,l,d in traits],
        metrics=[dict(id='misalignment',label='Misalignment',unit='%',lower_is_better=True),
                 dict(id='submission',label='Task submission',unit='%',lower_is_better=False),
                 dict(id='progress',label='Mean task progress',unit='/5',lower_is_better=False)],
        arms=arms,contrasts=[dict(baseline='nonmoral',arm='broader',metric='misalignment',delta=-stats['mean'],
            interval=dict(low=-stats['hi'],high=-stats['lo'],method='paired scenarios, fixed checkpoints'))])
    return document,audits


def main():
    assert hf_org() == 'dougalldeepmind'
    report = read(BASE/'final_comparison/comparison.json')
    assert report['status'] == 'all_four_complete'
    state = read(BASE/'evaluation/broader_eval_status.json')
    assert state['termination_verified'] and state['local_log_backup']['verified']
    backup = state['local_log_backup']
    assert hashlib.sha256(Path(backup['path']).read_bytes()).hexdigest() == backup['sha256']
    pods = read(BASE/'final_comparison/pod_inventory.json')
    assert all(pods[p]['exists'] is False for p in ('epd4o5f97zooij', '9ybfav9mzboi6y'))
    adapter = read(BASE/'training_retry1/salvaged_outputs/verified_final_adapter_receipt.json')
    assert adapter['verified'] and adapter['revision'] == state['target_revision']
    api = HfApi(token=False)
    public, checked = {}, []
    for arm, data in report['arms'].items():
        repo = (hf_org()+'/'+eval_name('odcv','qwen36_0_nonmoral_broader_7',date='2026-09-09') if arm == 'broader'
                else hf_org()+'/2026-09-09-odcv-'+{'nonmoral':'nonmoral-lf','math':'math','table2':'table2'}[arm]+'-common-3x')
        info = api.dataset_info(repo, files_metadata=True)
        assert not info.private
        for file in info.siblings:
            if arm != 'broader' and file.rfilename != 'results/results.json':
                continue
            if file.rfilename == '.gitattributes':
                continue
            path = Path(data['path'])/file.rfilename
            raw = path.read_bytes()
            actual = (hashlib.sha256(raw).hexdigest() if file.lfs else
                      hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
            assert actual == (file.lfs.sha256 if file.lfs else file.blob_id), path
            if file.rfilename == 'results/results.json':
                assert hashlib.sha256(raw).hexdigest() == data['result_sha256']
            checked.append(dict(arm=arm,path=file.rfilename,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        public[arm] = dict(repo=repo, revision=info.sha, private=False)
    assert sum(x['arm']=='broader' for x in checked)>240
    ledger = read(BASE/'evaluation/judge_ledger.json')
    assert len(ledger)==480 and all(x['status']=='settled' for x in ledger)
    judge = sum(x['charged_or_reserved_usd'] for x in ledger)
    train = read(BASE/'training_retry1/unexpected_stop_incident.json')['training_gpu_storage_exposure_upper_estimate_usd']
    budget = dict(prior_exposure_usd=135.318104131299,training_gpu_storage_usd=train,
                  eval_gpu_storage_usd=state['estimated_gpu_and_storage_usd'],judge_usd=judge,
                  basis='Conservative reservations plus elapsed-rate GPU/storage and token-rate judges, not provider invoices; shared-account deltas excluded.')
    budget['total_exposure_usd'] = sum(budget[k] for k in ('prior_exposure_usd','training_gpu_storage_usd','eval_gpu_storage_usd','judge_usd'))
    assert budget['total_exposure_usd']<300 and judge<5
    dest=BASE/'comparison_publication'
    secrets=secret_values()
    def copy(source, relative):
        raw=source.read_bytes()
        scan(raw,str(source),secrets)
        target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    for p in (BASE/'final_comparison').iterdir():
        if p.is_file() and p.suffix in {'.json', '.md'} and not p.name.endswith('.preview.json'):
            copy(p, 'results/'+p.name)
    for p in (dest/'results').iterdir():
        if p.suffix in {'.png', '.svg', '.pdf'}:
            p.unlink()  # Staging copies only; figures stay in final_comparison.
        elif p.suffix == '.md':
            p.write_text('\n'.join(line for line in p.read_text(encoding='utf-8').splitlines()
                                    if not line.startswith('!['))+'\n', encoding='utf-8')
    for relative in ('evaluation/broader_eval_status.json','evaluation/judge_ledger.json',
                     'evaluation_plan.json','training_retry1/unexpected_stop_incident.json',
                     'training_retry1/salvaged_outputs/verified_final_adapter_receipt.json'):
        copy(BASE/relative,'metadata/'+relative)
    copy(Path(__file__),'metadata/publish_broader_comparison.py')
    copy(Path('scratch/nonmoral/baseline_report.py'),'metadata/baseline_report.py')
    for name,value in [('public_sources',public),('verified_public_files',checked),('budget',budget)]:
        (dest/'metadata'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    frontend,audits=dataset_comparison(report,public,revision)
    for relative,value in [('results/dataset_comparison.json',frontend),('metadata/dataset_property_audit.json',audits)]:
        raw=(json.dumps(value,indent=2)+'\n').encode()
        scan(raw,relative,secrets)
        (dest/relative).write_bytes(raw)
    fields=dict(experiment='Broader nonmoral versus original nonmoral, math and Table2: four fixed checkpoints',
                date_generated='2026-09-09',constitution='none supplied at evaluation',
                source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+revision,
                models={k:{f:v[f] for f in ('target','target_revision','base_revision')} for k,v in report['arms'].items()},
                generation_config=report['protocol'],schema='results/: exact counts and paired intervals; metadata/: source pins, verification and cost evidence; figures remain local',
                provenance='scratch/nonmoral/baseline_report.py then scratch/nonmoral/publish_broader_comparison.py',
                limitations='One training seed each; evaluation repeats are not training seeds. No formal capability tests. Broader MR 76/240 versus original 33/240; this candidate did not improve alignment. Training backup interrupted by externally reported stop: final adapter verified locally, full archive incomplete. Stop and deletion actor unknown.')
    name=artifact_name('nonmoral-broader-comparison',date='2026-09-09')
    url=push_run_dir(dest,name,fields,private=False,front_matter={'pretty_name':frontend['title'],
        'tags':['nonmoral-deliberation','research-comparison','dataset-model-comparison']})
    info=api.dataset_info(hf_org()+'/'+name)
    assert not info.private
    receipt=dict(url=url,revision=info.sha,budget=budget,sources=public,verified_public_files=len(checked))
    (BASE/'comparison_publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stakes-experiment', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.stakes_experiment:
        publish_stakes(args.stakes_experiment, publish=args.publish)
    else:
        main()
