# ABOUTME: Render one AM condition's prompt exactly as the LIVE vLLM server templates it
# ABOUTME: (system + user + tools.json via /tokenize then /detokenize) and check the tool channel.
# Run (on the pod, while run_eval's server is up):
#   uv run python scratch/am_native_tools/render_prompt.py <run_dir>/prompts/<condition> [base_url]

import json
import sys
from pathlib import Path

import requests

cond = Path(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
model = requests.get(f"{base}/v1/models", timeout=30).json()["data"][-1]["id"]
system = (cond / "system_prompt.txt").read_text()
user = (cond / "user_prompt.txt").read_text() + "\n\n" + (cond / "email_content.txt").read_text()
body = {"model": model, "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
        "add_generation_prompt": True}
tools = cond / "tools.json"
if tools.exists():
    body["tools"] = json.loads(tools.read_text())
tok = requests.post(f"{base}/tokenize", json=body, timeout=60)
tok.raise_for_status()
ids = tok.json()["tokens"]
text = requests.post(f"{base}/detokenize", json={"model": model, "tokens": ids},
                     timeout=60).json()["prompt"]
print(f"model={model} tokens={len(ids)}\n" + "-" * 80)
print(text)
print("-" * 80)
print("native '# Tools' block present:", "# Tools" in text)
print("'<tools>' schemas present:     ", "<tools>" in text)
print("old '<tool_use:' text present: ", "<tool_use:" in text)
