# ABOUTME: Reads final adapter file hashes on an owned training pod for Hub comparison.
# ABOUTME: Produces a local manifest only; never uploads, modifies remote files, or controls pods.
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra.endpoints.vllm import SshExec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("owner", choices=["da5_retry1", "da25_retry1"])
    args = parser.parse_args()
    owner = ROOT / "output/plain_dose_campaign" / args.owner
    state = json.loads((owner / "status.json").read_text())
    dose = {"da5_retry1": "5", "da25_retry1": "25"}[args.owner]
    relative = f"output/train/2026-10-06_qwen36_0_da_{dose}/adapter"
    probe = f'''
import hashlib,json
from pathlib import Path
root=Path('/root/work')/{relative!r}
assert root.is_dir()
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file():
  assert not p.is_symlink()
  before=p.stat()
  sha=hashlib.sha256(); blob=hashlib.sha1(('blob '+str(before.st_size)+'\\0').encode())
  with p.open('rb') as f:
   while chunk:=f.read(4*1024*1024):
    sha.update(chunk); blob.update(chunk)
  after=p.stat()
  assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=before.st_size,sha256=sha.hexdigest(),git_blob_sha1=blob.hexdigest()))
assert any(f['path']=='adapter_model.safetensors' for f in files)
print(json.dumps(dict(root=str(root),files=files)))
'''
    remote = SshExec(state["host"], port=8000)
    manifest = json.loads(remote._ssh("/root/work/.venv/bin/python -c " + shlex.quote(probe), timeout=90))
    manifest.update(owned_pod=state["owned_pod"], expected_repo=state["arm"]["organism"])
    destination = owner / "publication_evidence/remote_adapter_manifest.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(destination), "files": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"])}))


if __name__ == "__main__":
    main()
