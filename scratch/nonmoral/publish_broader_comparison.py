# ABOUTME: Publish the frozen four-checkpoint comparison through shared HF contracts.
# ABOUTME: Verify public eval bytes, local backups, pod absence and bounded cost first.
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import HfApi
from src.infra.huggingface import hf_org, push_run_dir
from src.naming import artifact_name, eval_name
from scratch.nonmoral.publish_invalid_baseline import scan, secret_values

BASE = Path('output/nonmoral_broader/20260909')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


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
        repo = (hf_org()+'/'+eval_name('odcv','qwen36_0_nonmoral_broader_7') if arm == 'broader'
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
        if p.is_file(): copy(p, 'results/'+p.name)
    for relative in ('evaluation/broader_eval_status.json','evaluation/judge_ledger.json',
                     'evaluation_plan.json','training_retry1/unexpected_stop_incident.json',
                     'training_retry1/salvaged_outputs/verified_final_adapter_receipt.json'):
        copy(BASE/relative,'metadata/'+relative)
    copy(Path(__file__),'metadata/publish_broader_comparison.py')
    copy(Path('scratch/nonmoral/baseline_report.py'),'metadata/baseline_report.py')
    for name,value in [('public_sources',public),('verified_public_files',checked),('budget',budget)]:
        (dest/'metadata'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    fields=dict(experiment='Broader nonmoral versus original nonmoral, math and Table2: four fixed checkpoints',
                date_generated='2026-09-09',constitution='none supplied at evaluation',
                source_repo='https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+revision,
                models={k:{f:v[f] for f in ('target','target_revision','base_revision')} for k,v in report['arms'].items()},
                generation_config=report['protocol'],schema='results/: exact counts, paired intervals and figures; metadata/: source pins, verification and cost evidence',
                provenance='scratch/nonmoral/baseline_report.py then scratch/nonmoral/publish_broader_comparison.py',
                limitations='One training seed each; evaluation repeats are not training seeds. No formal capability tests. Broader MR 76/240 versus original 33/240; this candidate did not improve alignment. Training backup interrupted by externally reported stop: final adapter verified locally, full archive incomplete. Stop and deletion actor unknown.')
    name=artifact_name('nonmoral-broader-comparison',date='2026-09-09')
    url=push_run_dir(dest,name,fields,private=False,front_matter={'tags':['nonmoral-deliberation','research-comparison']})
    info=api.dataset_info(hf_org()+'/'+name)
    assert not info.private
    receipt=dict(url=url,revision=info.sha,budget=budget,sources=public,verified_public_files=len(checked))
    (BASE/'comparison_publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
