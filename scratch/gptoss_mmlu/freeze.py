# ABOUTME: Freeze the clean Windows checkout identity and exact runtime file hashes.
# ABOUTME: Run: uv run --project scratch/swebench_cpu_env python scratch/gptoss_mmlu/freeze.py
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
ROOT=Path.cwd()
OUT=ROOT/'output/gptoss_mmlu_20261014'
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=subprocess.check_output(['git','ls-files','-z','src','configs','scratch/gptoss_mmlu','pyproject.toml','uv.lock']).decode().split('\0')
paths=[p for p in paths if p and Path(p).is_file()]
files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
with tarfile.open(OUT/'source.tar.gz','w:gz') as archive:
    for p in paths: archive.add(p,arcname=p)
(OUT/'source.json').write_text(json.dumps(dict(sha=sha,files=files,clean=True,
    archive_sha256=hashlib.sha256((OUT/'source.tar.gz').read_bytes()).hexdigest()),indent=2))
print(json.dumps(dict(sha=sha,files=len(files),archive_bytes=(OUT/'source.tar.gz').stat().st_size)))
