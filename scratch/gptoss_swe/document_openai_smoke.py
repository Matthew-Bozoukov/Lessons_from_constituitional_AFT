# ABOUTME: Record verified smoke closeout in repository documentation after the local owner finishes.
# ABOUTME: No model calls or resource changes; refuses to commit over unrelated working-tree edits.
import json
from pathlib import Path
import subprocess
import time
from scratch.gptoss_swe.openai_smoke import ROOT,save


def main():
    with (ROOT/'documentation-owner-started.json').open('x') as f:json.dump(dict(at=time.time()),f)
    try:
        deadline=time.monotonic()+4.5*3600
        while not (ROOT/'finalized.json').exists():
            if (ROOT/'finalize-failure.json').exists():raise RuntimeError('Artifact closeout failed; preserve all evidence')
            if time.monotonic()>deadline:raise TimeoutError('Closeout not finalized')
            time.sleep(20)
        publication=json.loads((ROOT/'publication.json').read_text());assert publication['complete']
        assert json.loads((ROOT/'cleanup.json').read_text())['owned_remaining']==[]
        results={a:json.loads((ROOT/'package/results'/f'{a}.json').read_text()) for a in ('base','control')}
        url=f"https://huggingface.co/datasets/{publication['repo']}/tree/{publication['revision']}"
        lines=['','## Verified closeout','',f'[Immutable artifacts]({url}). {publication["verified_files"]} files verified by SHA256 readback.','',
               '| Metric | Base | Control |','|---|---:|---:|']
        def row(name,fn):lines.append('| '+name+' | '+' | '.join(str(fn(results[a])) for a in results)+' |')
        row('ODCV task completions / 10',lambda r:sum(c['tool_completed'] for c in r['diagnostics']['odcv']['cells']))
        row('ODCV context / cycle / response limits',lambda r:'/'.join(str(sum(c[k] for c in r['diagnostics']['odcv']['cells'])) for k in ('context_limit','step_limit','response_limit')))
        row('ODCV diagnostic misalignment %',lambda r:r['odcv']['ours']['overall']['mr_pct'])
        row('SWE resolved / graded',lambda r:f"{r['swe']['n_resolved']}/{r['swe']['n_graded']}")
        row('SWE model limit endings among graded',lambda r:r['diagnostics']['swe']['ended_statuses'].get('LimitsExceeded',0))
        row('SWE interrupted / unstarted',lambda r:f"{r['swe']['n_interrupted']}/{r['swe']['n_not_started']}")
        cost=sum(sum(r['costs_usd'].values()) for r in results.values())
        lines+=['',f'Conservative ledger accounting including compatibility, inference and judges: ${cost:.6f}; not a provider invoice. Original caps unchanged.',
                '', 'SWE coverage is partial and cannot support a clean paired benchmark comparison. Interruptions and unstarted tasks are not model limit outcomes. No paid model outcome was rerun. All ODCV cells were judged, including incomplete control cells. Local smoke containers removed, own keep-awake stopped, caches retained. No cloud CPU/GPU rentals or recurring scheduler. The trace-decoding failure and missing paid completion payload are disclosed in the artifact manifest.']
        text='\n'.join(lines)+'\n'
        save(ROOT/'final-report.json',dict(artifact_url=url,cost_usd=cost,text=text))
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(), 'Working tree changed; report retained without committing'
        doc=Path('docs/gptoss_openai_interface_smoke_2026-10-09.md')
        current=doc.read_text(encoding='utf-8');assert '## Verified closeout' not in current
        doc.write_text(current+text,encoding='utf-8')
        with Path('docs/LOG.md').open('a',encoding='utf-8') as f:f.write('\n### 2026-10-09: OpenAI-interface smoke verified closeout\n'+text)
        subprocess.run(['git','add',str(doc),'docs/LOG.md'],check=True)
        subprocess.run(['git','commit','-m','Record verified base and control reference-Harmony smoke results'],check=True)
        subprocess.run(['git','push','origin','codex/gptoss-swe-three-arms'],check=True)
        save(ROOT/'documentation-finished.json',dict(at=time.time(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    except BaseException as e:
        save(ROOT/'documentation-failure.json',dict(at=time.time(),error=repr(e)));raise

if __name__=='__main__':main()
