# ABOUTME: Fetches pinned published Qwen results for the Figure 1 correction audit.
# ABOUTME: Read-only Hub access; records result and metadata files locally for reproducibility.
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
OUT = Path('output/2026-10-12_figure1')
NAMES = '''2026-10-06-odcv-qwen36-0-plain
2026-10-06-odcv-qwen36-0-da-15
2026-10-11-odcv-qwen36-0-da-15-pooled2
2026-10-08-am-qwen36-0-plain
2026-10-08-am-qwen36-0-da-15
2026-10-11-hospital-self-sacrificial-qwen36-0-plain
2026-10-07-hospital-self-sacrificial-qwen36-0-da-15
2026-10-07-odcv-qwen36-0-da-low-15
2026-10-07-odcv-qwen36-0-nonmoral-original-15
2026-10-08-odcv-qwen36-0-da-grok-resp-15
2026-10-08-odcv-qwen36-0-da-grok-15
2026-10-11-odcv-qwen36-0-da-15-response-only
2026-10-08-odcv-qwen36-0-da-15-emptythink
2026-09-29-odcv-qwen36-0-da-no-t6-15
2026-09-26-odcv-qwen36-0-da-new-t6-15
2026-09-05-odcv-qwen36-0-da-ablated-7
2026-10-04-odcv-qwen36-0-ablation-15
2026-10-04-odcv-qwen36-0-ablation-stated-harm-15'''.splitlines()

def fetch(name):
    api = HfApi()
    repo = 'dougalldeepmind/' + name
    rev = api.dataset_info(repo).sha
    folder = OUT / 'sources' / name
    folder.mkdir(parents=True, exist_ok=True)
    files = ['README.md', 'results/results.json', 'metadata/run_meta.json']
    if '-hospital-' in name:
        files += ['results/per_seed.json']
    record = {'repo': repo, 'revision': rev, 'files': {}}
    for file in files:
        try:
            p = Path(hf_hub_download(repo, file, repo_type='dataset', revision=rev))
            text = p.read_text(encoding='utf-8')
            (folder / file.replace('/', '__')).write_text(text, encoding='utf-8')
            record['files'][file] = json.loads(text) if file.endswith('.json') else text
        except Exception as e:
            record['files'][file] = {'fetch_error': type(e).__name__}
    return record

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, NAMES))
    (OUT / 'pinned_sources.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for r in rows:
        result = r['files']['results/results.json']
        meta = r['files']['metadata/run_meta.json']
        print(r['repo'], r['revision'])
        print('TARGET', meta.get('target'), meta.get('target_revision'))
        if 'ours' in result:
            print(json.dumps(result['ours']['overall']))
        else:
            print(json.dumps(result)[:9000])
