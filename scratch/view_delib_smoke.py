# ABOUTME: Build an offline, self-contained step-by-step viewer of a deliberative SFT smoke run.
# ABOUTME: Reads saved artifacts only; requests are reconstructed and never sent to a provider.

import argparse
import json
from pathlib import Path

from src.data.synth.deliberative_alignment.data import generation_messages
from src.naming import artifact_name, to_local


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(run_dir, source_file):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    records = read_rows(run_dir / 'stage_1_prompts.jsonl')
    attempts = read_rows(run_dir / 'generations.partial.jsonl')
    source = read_rows(source_file)
    augmentation = (run_dir / 'generation_prompt.txt').read_text()
    final_path = run_dir / 'dataset.jsonl'
    final = read_rows(final_path) if final_path.exists() else []
    if final and len(final) != len(records):
        raise ValueError('Final dataset and prompt snapshot counts differ')
    examples = []
    for index, record in enumerate(records):
        original = source[record['source_row']]
        assert original['messages'][:-1] == record['messages'], 'Source context mismatch'
        exported = final[index] if final else None
        if exported:
            assert exported['messages'][:-1] == record['messages'], 'Export context changed'
            assert exported['supervise'] == 'final'
        cfg = manifest['config']
        request = dict(model=cfg['model'], messages=generation_messages(record, augmentation),
                       **cfg['sampling'], provider=manifest['provider_pin'])
        if record.get('tools'):
            request['tools'] = record['tools']
        examples.append(dict(id=record['id'], source_row=record['source_row'],
            source=original, intake=record, request=request,
            attempts=[a for a in attempts if a['id'] == record['id']], final=exported))
    bundle = dict(manifest=manifest, augmentation=augmentation, examples=examples)
    # The payload is text, never executable HTML, including model-written closing tags.
    encoded = json.dumps(bundle, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    target = run_dir / (to_local(artifact_name('delib smoke walkthrough')) + '.html')
    target.write_text(HTML.replace('__PAYLOAD__', encoded), encoding='utf-8')
    print(target.resolve())
    return target


HTML = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deliberative alignment · smoke walkthrough</title>
<style>
:root{font-family:system-ui,sans-serif;color:#202b35;background:#f3f5f7}body{margin:0}
header{background:#142d3b;color:white;padding:32px max(24px,calc((100vw - 1150px)/2))}
h1{font-size:30px;margin:8px 0}header p{color:#c8d8e1;max-width:850px;line-height:1.6}
main{max-width:1150px;margin:24px auto;padding:0 24px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
button,select{padding:10px 14px;border:1px solid #c9d4db;border-radius:7px;background:white;cursor:pointer;color:#203847}
button[aria-pressed=true]{background:#176b65;color:white;border-color:#176b65}
article{background:white;border:1px solid #dce3e8;border-radius:12px;padding:24px;margin-bottom:24px}
h2{margin-top:0}p{line-height:1.6}.note{border-left:4px solid #de9a36;padding:8px 16px;background:#fff9ed}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,monospace;background:#f5f7f9;padding:18px;border-radius:8px}
.role{font-size:12px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:#176b65;margin-top:22px}
.reasoning{border-left:3px solid #9d76be}.answer{border-left:3px solid #248a78}
summary{cursor:pointer;padding:12px 0;font-weight:600}.meta{font-size:13px;color:#536674}a{color:#176b65}
@media print{nav,button,select{display:none}header{background:white;color:black}article{break-inside:avoid}}
</style>
<header><div>RESEARCH TRACE / DELIBERATIVE SFT</div><h1>From source prompt to training target</h1>
<p>Every saved stage, without truncation. The constitution enters the generation request, then leaves the training context. Generated reasoning stays.</p><div id="status"></div></header>
<main><label>Example <select id="example"></select></label> <button id="download">Download complete trace JSON</button>
<nav id="steps" aria-label="Pipeline steps"></nav><div id="view"></div>
<details><summary>Run provenance and resolved configuration</summary><pre id="manifest"></pre></details></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent);
const labels=['1 · Source row','2 · Prompt intake','3 · Generation request','4 · Model response','5 · SFT export'];
let stage=0;const select=document.getElementById('example'),view=document.getElementById('view');
function add(parent,tag,text,cls){const el=document.createElement(tag);el.textContent=text;if(cls)el.className=cls;parent.append(el);return el}
function raw(parent,title,obj){const d=add(parent,'details','');add(d,'summary',title);add(d,'pre',JSON.stringify(obj,null,2))}
function messages(parent,list){for(const m of list||[]){add(parent,'div',m.role,'role');if(m.reasoning_content)add(parent,'pre',m.reasoning_content,'reasoning');if(m.content)add(parent,'pre',m.content,m.role==='assistant'?'answer':'');if(m.tool_calls)raw(parent,'Tool calls',m.tool_calls)}}
data.examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1} / source row ${e.source_row}`;select.append(o)});
add(document.getElementById('status'),'span',`${data.manifest.status} · ${data.examples.length} examples · ${data.manifest.config.model} · $${Number(data.manifest.usage?.total_usd||0).toFixed(4)}`);
document.getElementById('manifest').textContent=JSON.stringify(data.manifest,null,2);
function render(){view.replaceChildren();document.querySelectorAll('nav button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===stage)));
const e=data.examples[Number(select.value)],a=add(view,'article','');add(a,'h2',labels[stage]);
if(stage===0){add(a,'p','Exact original dataset row. Its final assistant response is shown for provenance only: it is removed before generation.','note');messages(a,e.source.messages);raw(a,'Original row JSON',e.source)}
if(stage===1){add(a,'p','Original final assistant removed. Earlier messages and source metadata are retained; no new scenario is generated.');messages(a,e.intake.messages);raw(a,'Saved prompt snapshot',e.intake)}
if(stage===2){add(a,'p','Full model-facing messages reconstructed using the pipeline function and saved augmentation. This is not a raw HTTP capture; credentials and transport headers are deliberately absent.','note');messages(a,e.request.messages);raw(a,'Request settings and messages',e.request);raw(a,'Generation-only augmentation',data.augmentation)}
if(stage===3){add(a,'p','All recorded attempts, including failures. Native reasoning and final answer are displayed separately. Completeness validation is not quality judging.');
for(const [i,t] of e.attempts.entries()){add(a,'h3',`Attempt ${i+1}`);if(t.error)add(a,'p',t.error,'note');if(t.response){add(a,'p',`Provider: ${t.response.provider} · finish: ${t.response.finish_reason} · tokens: ${t.response.prompt_tokens} input / ${t.response.completion_tokens} output`,'meta');messages(a,[{role:'assistant',...t.response}])}raw(a,'Saved response record',t)}
if(!e.attempts.length)add(a,'p','No response has been recorded.')}
if(stage===4){if(e.final){add(a,'p','Verified: original context restored unchanged; only the new final assistant turn is supervised, including reasoning. No judge, SFT training, or RL occurs in this run.','note');messages(a,e.final.messages);raw(a,'Exact final training row',e.final)}else add(a,'p','No final dataset published. The run has not completed successfully.','note')}
}
labels.forEach((label,i)=>{const b=add(document.getElementById('steps'),'button',label);b.onclick=()=>{stage=i;render()}});select.onchange=render;
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='delib-smoke-trace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--source-file', type=Path, required=True, help='Local dataset.jsonl from the source revision recorded in the run')
    args = parser.parse_args()
    build(args.run_dir, args.source_file)
