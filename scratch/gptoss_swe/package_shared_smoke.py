# ABOUTME: Package completed base-only smoke evidence and verify immutable Hugging Face readback.
# ABOUTME: Refuses incomplete coverage, scans credentials, and never samples or changes outcomes.
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path('output/gptoss_shared_smoke')
REPO='dougalldeepmind/2026-10-08-gptoss120b-base-shared-interface-smoke'

def main():
    from dotenv import dotenv_values
    from huggingface_hub import HfApi,hf_hub_download
    from scratch.gptoss_swe.audit_shared_smoke import main as audit
    audit()
    assert (ROOT/'swe-finished.json').exists() and (ROOT/'odcv-judge-finished.json').exists()
    swe=json.loads((ROOT/'swe/results/qualification.json').read_text())
    assert swe['n_total']==swe['n_valid_rollouts']==swe['n_graded']==10
    odcvroot=Path(json.loads((ROOT/'odcv-rollout.json').read_text())['path'])
    odcv=json.loads((odcvroot/'results.json').read_text())
    assert odcv['n_judged']==10 and odcv['n_dropped_all_na']==0
    out=ROOT/'package'
    assert not out.exists(), 'Preserve existing package; recover publication separately'
    for d in ('results','metadata','rollouts'): (out/d).mkdir(parents=True,exist_ok=True)
    keys=dotenv_values(Path.home()/'source/repos/LASR/teaching_claude_why_replication/.env')
    needles=[v.encode() for k,v in keys.items() if v and len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    def archive(destination, paths):
        with tarfile.open(destination,'w:gz') as tar:
            for path in paths:
                files=sorted(path.rglob('*')) if path.is_dir() else [path]
                for f in files:
                    if not f.is_file(): continue
                    data=f.read_bytes()
                    assert not any(n in data for n in needles), f'Credential detected: {f}'
                    tar.add(f,arcname=f.relative_to(ROOT),recursive=False)
    archive(out/'rollouts/odcv.tar.gz',[ROOT/'odcv'])
    archive(out/'rollouts/swe.tar.gz',[ROOT/'swe/rollouts'])
    archive(out/'metadata/rendered-requests-and-raw-responses.tar.gz',[ROOT/'odcv-traces',ROOT/'swe-traces'])
    archive(out/'metadata/swe-source-grading-and-state.tar.gz',[ROOT/'swe/metadata',ROOT/'swe/results'])
    archive(out/'metadata/driver-evidence.tar.gz',[p for p in ROOT.iterdir() if p.is_file()])
    shutil.copyfile(ROOT/'audit.json',out/'results/interface-audit.json')
    shutil.copyfile(ROOT/'qualitative-findings.md',out/'results/failure-analysis.md')
    shutil.copyfile(ROOT/'swe/results/qualification.json',out/'results/swe.json')
    shutil.copyfile(odcvroot/'results.json',out/'results/odcv.json')
    for name in ('odcv-budget.json','swe-budget.json','judge-budget.json','selection.json','images-verified.json','vast-inventory.json'):
        shutil.copyfile(ROOT/name,out/'metadata'/name)
    info=json.loads((ROOT/'audit.json').read_text())
    costs={k:info[k]['budget']['accounted'] for k in ('odcv','swe')}
    assert all(set(info[k]['budget']['request_states'])=={'completed'} for k in ('odcv','swe')), 'Unknown inference reservations require diagnosis'
    costs['judge']=sum(x['charged_or_reserved_usd'] for x in json.loads((ROOT/'judge-budget.json').read_text()))
    conversations=info['swe']['conversations'].values()
    diagnostic_rows='\n'.join(f"| {g['instance_id']} | {g['responses']} | {g['invalid_json']} | {g['max_consecutive_invalid_json']} | {g['max_consecutive_no_tool']} |" for g in conversations)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    manifest=dict(base_model='openai/gpt-oss-120b',target='tinker://base',adapter=False,scope='diagnostic subsets only',
        source=source,initial_runtime='b9ff8ab4',corrected_runtime='66948b8',costs_usd=costs,
        cost_note='Token ledger estimates, not provider invoice; OpenRouter ambient rollout usage delta is not Tinker cost',
        odcv_completed=sum(x['tool_completed'] for x in info['odcv']['cells']),swe_resolved=swe['n_resolved'],
        swe_ended_statuses=info['swe']['ended_statuses'],
        odcv_mr_pct=odcv['ours']['overall']['mr_pct'],
        caveats=['Three initial ODCV cells were retained before terminal marker repair; seven used the corrected interface.',
                 'No valid outcome was resampled. This is not a full benchmark or matched paper comparison.',
                 'ODCV uses first five alphabetical scenarios in both variants; SWE uses seed-0 selection excluding requests.',
                 'Initial billing probe misrouted Tinker auth to OpenRouter; rejected, fixed; no credentials in artifact.'])
    (out/'metadata/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'README.md').write_text(f'''---
license: mit
---
# Base GPT-OSS-120B shared-interface diagnostic smoke

Base model only, no LoRA, local Docker and Tinker. These small subsets measure interface health, not full benchmark performance.

* ODCV: {manifest['odcv_completed']}/10 reached task_complete. Numerical limit hits: 0. One invalid JSON call received validation feedback and was corrected on the next turn.
* SWE-bench Lite: {swe['n_resolved']}/10 officially resolved; 10 valid attempts and 10 graded.
* SWE attempt endings: {json.dumps(manifest['swe_ended_statuses'])}. A valid attempt is not necessarily solved.
* ODCV diagnostic misalignment rate: {manifest['odcv_mr_pct']}% on these ten cells. Do not compare its paper delta as a matched evaluation.
* Accounted costs: {json.dumps(costs)} USD; estimates, not invoices.

The first three ODCV cells preceded a correction to match official cookbook handling of complete tool calls ending with the return marker. They are preserved, not rerun. The other seven and all SWE tasks used the corrected parser. There is no custom JSON coaching, bracket instruction, or benchmark-specific system prompt in the adapter. Schemas and benchmark instructions are supplied normally. Raw malformed arguments are validated rather than repaired.

The SWE smoke still exposed a persistent no-tool loop on Sphinx 8273 after JSON/shell quoting and malformed Harmony recipient errors. After 177 consecutive replies without tools, the final generation repeated "I cannot" messages until exhausting all 16384 response tokens (53868 prompt tokens). This was not context exhaustion. This is not evidence that the interface is ready for a full campaign; a larger response allowance would likely prolong that repetition.

Sphinx 8721 separately hit the 131072 context limit after 238 replies. Its maximum sampled prompt was 130504 tokens and largest response 4077 tokens. It continued issuing tools; the longest invalid-JSON streak was two and longest no-tool streak one. A larger response allowance alone would not expand that context window.

SymPy 15308 also exhausted 16384 response tokens repeating apologies, after 48 consecutive replies without tools. Its terminal prompt had 50531 tokens. These are serious remaining failures on the base model alone.

| SWE task | Replies | Invalid JSON calls | Longest invalid-JSON streak | Longest no-tool streak |
|---|---:|---:|---:|---:|
{diagnostic_rows}

ODCV sampling: temperature 0.7, 8192 response tokens, 28000 context, 50 cycles. SWE sampling: temperature 1, top_p .95, top_k 20, 16384 response tokens, 131072 context, 262144 generated task tokens, 500 steps. Both use medium reasoning. These numerical settings are deliberately retained and are not a controlled comparison between evaluations.

Results are under `results/`. Rollout archives are under `rollouts/`. Actual rendered prompt tokens, raw responses, configurations, ledgers, image identities, grading logs and source/state evidence are under `metadata/`.

Source: `{source}` on `codex/gptoss-swe-three-arms`. See `metadata/manifest.json` for protocol caveats.
''',encoding='utf-8')
    hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'metadata/file-hashes.json').write_text(json.dumps(hashes,indent=2))
    hashes['metadata/file-hashes.json']=hashlib.sha256((out/'metadata/file-hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=keys['HF_TOKEN'])
    api.create_repo(REPO,repo_type='dataset',exist_ok=True)
    commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=out,commit_message='Publish base shared-interface smoke with retained diagnostics').oid
    receipt=dict(repo=REPO,revision=commit,verified_files=0,sha256=hashes)
    (ROOT/'publication.json').write_text(json.dumps(receipt,indent=2))
    for name,expected in hashes.items():
        local=hf_hub_download(REPO,name,repo_type='dataset',revision=commit,token=keys['HF_TOKEN'])
        assert hashlib.sha256(Path(local).read_bytes()).hexdigest()==expected,name
        receipt['verified_files']+=1
    receipt['complete']=True
    (ROOT/'publication.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},indent=2))

if __name__=='__main__': main()
