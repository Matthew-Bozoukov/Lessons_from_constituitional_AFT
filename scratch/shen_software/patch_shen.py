# ABOUTME: Re-runnable anchored patch of Shen et al.'s software harness (safety-research/misalignment-in-ai-
# ABOUTME: organizations): restrict built-in tools per role, and make the global timeout an env setting.
"""Run: uv run python scratch/shen_software/patch_shen.py <path to the repo's software/ dir>

Why (2026-11-13): newer Claude Code exposes built-in tools Shen's 2025 SDK did not (Agent sub-agents,
CronCreate, ...). In the org-DA sepsis run the PM used those instead of its ticket tools, then waited on a
permission that never came, and no prediction script was written. `allowed_tools` only PRE-APPROVES tools;
the SDK's `tools` sets which built-in tools EXIST. Each role now gets exactly the built-ins Shen's code
already lists for it (PM: none, MCP ticket tools only). Every edit is marked VENDORED PATCH. Idempotent;
fails loudly if an anchor is missing.
"""

import sys
from pathlib import Path

SW = Path(sys.argv[1])
TAG = "# VENDORED PATCH (2026-11-13)"

EDITS = [
    (
        SW / "src/agent_org/agent_org.py",
        "    global_timeout: int = 3600  # 1 hour\n",
        f'    global_timeout: int = int(os.environ.get("SHEN_GLOBAL_TIMEOUT", "3600"))  {TAG}: env-settable\n',
    ),
    (
        SW / "src/agent_org/agent_org.py",
        '            mcp_servers={"pm": pm_tools_server},\n',
        f'            mcp_servers={{"pm": pm_tools_server}},\n            tools=[],  {TAG}: PM gets no built-in tools\n',
    ),
    (
        SW / "src/agent_org/agent_org.py",
        '            mcp_servers={"org": tools_server},\n',
        f'            mcp_servers={{"org": tools_server}},\n'
        f'            tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],  {TAG}: only listed built-ins\n',
    ),
    (
        SW / "src/agent_org/tools.py",
        '        allowed_tools=["Read", "Glob", "Grep", "Bash"],\n',
        f'        tools=["Read", "Glob", "Grep", "Bash"],  {TAG}: only listed built-ins\n'
        '        allowed_tools=["Read", "Glob", "Grep", "Bash"],\n',
    ),
]

for path, old, new in EDITS:
    text = path.read_text()
    if new in text:
        print(f"already patched: {path.name}: {new.strip()[:70]}")
        continue
    if text.count(old) != 1:
        raise SystemExit(f"anchor not found exactly once in {path}: {old.strip()!r}")
    path.write_text(text.replace(old, new))
    print(f"patched: {path.name}: {new.strip()[:70]}")

if "import os" not in (SW / "src/agent_org/agent_org.py").read_text():
    raise SystemExit("agent_org.py does not import os; the timeout patch needs it")
