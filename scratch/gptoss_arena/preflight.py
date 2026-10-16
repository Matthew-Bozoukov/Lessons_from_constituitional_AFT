# ABOUTME: Count every Arena prompt through the exact native Harmony renderer without sampling.
# ABOUTME: Run: Pinned shim Python -m scratch.gptoss_arena.preflight
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from src.infra.endpoints.tinker_harmony import HarmonyRenderer, generation_prompt

root = Path('/work/output/gptoss_arena_20261014')
policy = json.loads((root/'prepared.json').read_text())
native = policy['native_protocol']
assert version('openai-harmony') == native['harmony_version']
renderer = HarmonyRenderer(reasoning_effort=native['reasoning'], current_date=native['render_date'])
rows = []
for q in json.loads((root/'questions.json').read_text()):
    ids = generation_prompt(renderer, [dict(role='user',content=q['prompt'])]).to_ints()
    assert len(ids) + 32768 <= 131072
    rows.append(dict(uid=q['uid'],prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(q['prompt'].encode()).hexdigest()))
upper = 3 * sum(r['prompt_tokens'] * .33 + 32768 * .84 for r in rows) / 1e6
assert upper <= policy['ceiling_usd']
result = dict(rows=rows,native_protocol=native, max_prompt_tokens=max(r['prompt_tokens'] for r in rows),
              all_uncached_maximum_usd=upper,questions=750,response_allowance=32768,context_limit=131072)
(root/'token-preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','native_protocol')}),flush=True)
