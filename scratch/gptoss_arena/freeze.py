# ABOUTME: Archive exact campaign/runtime bytes and clean Git identity before paid inference.
# ABOUTME: Run: uv run --no-sync python -X utf8 -m scratch.gptoss_arena.freeze [--qualification]
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

out = Path('output/gptoss_arena_20261014')
clean = not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
assert clean or '--qualification' in sys.argv, 'Commit source before paid inference'
sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
names = subprocess.check_output(['git','ls-files','-z','src','configs','scratch/gptoss_mmlu','scratch/gptoss_arena','scratch/swebench_cpu_env','pyproject.toml','uv.lock']).decode().split('\0')
names = sorted(set(p for p in names if p and Path(p).is_file()) | {p.as_posix() for p in Path('scratch/gptoss_arena').glob('*') if p.is_file()})
files = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names}
with tarfile.open(out/'source.tar.gz','w:gz') as tar:
    for p in names:
        tar.add(p,arcname=p)
(out/'source.json').write_text(json.dumps(dict(sha=sha,clean=clean,files=files,archive_sha256=hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print(json.dumps(dict(sha=sha,clean=clean,files=len(files))))
