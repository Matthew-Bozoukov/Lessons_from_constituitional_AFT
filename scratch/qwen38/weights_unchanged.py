# ABOUTME: Checks whether Qwen3.8-27B's weight files changed between two Hub commits
# ABOUTME: (LFS sha256 per safetensors shard), for the profile's revision comment.
from huggingface_hub import HfApi

api = HfApi()
REPO = "Qwen/Qwen3.8-27B"
shas = {}
for rev in ("8c0b3d30f3", "37561816e66b5b9153b1b64c10968c18c1e6f4a0"):
    info = api.model_info(REPO, revision=rev, files_metadata=True)
    shas[rev] = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
    print(rev[:10], "lfs files:", len(shas[rev]))
a, b = shas.values()
print(
    "identical LFS files:",
    a == b,
    "| differing:",
    sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)),
)
