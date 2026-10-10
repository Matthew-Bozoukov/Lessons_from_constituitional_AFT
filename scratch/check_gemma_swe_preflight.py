# ABOUTME: Pre-flight for the Gemma SWE-bench Lite config: profile resolution, served parsers
# ABOUTME: and the thinking pins that actually reach the rendered template.
import sys
sys.path.insert(0, "/home/matthewb/git repos/tcw-swe-gemma")
import yaml

from src.model_profile import find_profile
from src.infra.endpoints.vllm import pin_prefix

cfg = yaml.safe_load(open("configs/eval/swebench_mini/lite-gemma.yaml"))
print(f"    target   : {cfg['target']}@{cfg['target_revision'][:8]}")
print(f"    dataset  : {cfg['dataset']}@{cfg['dataset_revision'][:8]}")
print(f"    protocol : {cfg['protocol_version']}   mode: {cfg['mode']}")
print(f"    sampling : {cfg['sampling']}")

p = find_profile(cfg["base"])
assert p is not None, f"no profile registered for {cfg[chr(34)+chr(98)+chr(97)+chr(115)+chr(101)+chr(34)]}"
s = p.serving
print(f"    profile  : key={p.key} model={p.model}")
print(f"    parsers  : reasoning={s["reasoning_parser"]} tool_call={s["tool_call_parser"]} "
      f"max_num_seqs={s["max_num_seqs"]}")
assert s["tool_call_parser"] == "gemma4", "tool-call parser must be gemma4 or every task fails silently"

pre = pin_prefix(cfg["mode"])
print("    pinned template prefix:")
for line in pre.strip().splitlines():
    print(f"      {line}")
assert "set preserve_thinking = false" in pre, "preserve_thinking is not pinned false"
print("    OK: preserve_thinking pinned false, gemma4 parsers resolved")
