# ABOUTME: Loads scratch/dat/dat.yaml (the dat recipe: REQUIREMENTS, per-stage system prompts, schemas, file types, sees /
# ABOUTME: blind-to lists) and holds the small functions that format each stage's inputs into its user message.
"""The recipe lives in dat.yaml; this module is its reader. Nothing here is prompt text.

Each stage's `system` in the yaml may contain `{requirements}`, replaced with the yaml's
`requirements` block verbatim at load. `{style_guidance}` likewise. `{n}` (write) and `{trait_name}` / `{trait_text}` (respond) are
left for the driver to fill. The `*_prompt` functions below build a stage's user message from exactly the inputs
`pipeline.stage_inputs()` hands it, so the yaml's `sees` list and the code agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

CONFIG = Path(__file__).with_name("dat.yaml")
_cfg = yaml.safe_load(CONFIG.read_text())

REQUIREMENTS: str = _cfg["requirements"].rstrip("\n")
STYLE_GUIDANCE: str = _cfg["style_guidance"].rstrip("\n")
STRUCTURED_FILE_TYPES = tuple(_cfg["file_types"]["structured"])
PROSE_FILE_TYPES = tuple(_cfg["file_types"]["prose"])
CODE_FILE_TYPES = tuple(_cfg["file_types"]["code"])
FILE_TYPES = STRUCTURED_FILE_TYPES + PROSE_FILE_TYPES + CODE_FILE_TYPES
CONSTITUTION = _cfg["constitution"]
SECTORS_FROM = _cfg["sectors_from"]
CLOCK = _cfg["clock"]
MAX_LOOKS = int(_cfg["explore"]["max_looks"])

STAGES: list[dict] = []
for _s in _cfg["stages"]:
    STAGES.append({**_s, "system": _s["system"].replace("{requirements}", REQUIREMENTS).replace("{style_guidance}", STYLE_GUIDANCE).rstrip("\n"),
                   "schema": _s.get("schema")})
BY_NAME = {s["name"]: s for s in STAGES}
MODELS = {s["name"]: s["model"] for s in STAGES}
MAX_TOKENS = {s["name"]: int(s["max_tokens"]) for s in STAGES}
TEMPERATURE = {s["name"]: float(s["temperature"]) for s in STAGES}
REASONING = {s["name"]: s.get("reasoning") for s in STAGES}   # None = provider default


def system_of(stage: str) -> str:
    return BY_NAME[stage]["system"]


def schema_of(stage: str) -> dict | None:
    return BY_NAME[stage]["schema"]


RESPOND_TAGS = ("reasoning", "response", "tool_call")


# --- user-message builders, one per stage; the argument is what stage_inputs() returned -------------

def write_prompt(trait_name: str, trait_text: str, sector_text: str, n: int, name_seed: str) -> str:
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\nSECTOR: {sector_text}\n"
            f"NAME SEED: build the organisation's name around \"{name_seed}\" (a founder, a place or a street), so names differ across situations.\n\n"
            f"Write {n} situations. Return the JSON.")


def revise_prompt(trait_name: str, trait_text: str, scenario: dict) -> str:
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\nSITUATION:\n{json.dumps(scenario, indent=1)}\n\n"
            "Review against every requirement and return the JSON.")


def environment_prompt(inp: dict) -> str:
    facts = "\n".join(f"- [{f['id']}] {f['statement']}" for f in inp["world_facts"])
    return (f"ORGANISATION: {inp['organisation']}\nASSISTANT ROLE: {inp['deployment']}\nTODAY: {inp['today']}\n\n"
            f"FACTS TRUE IN THIS ORGANISATION:\n{facts}\n\n"
            "Return the JSON: your directories and the files that carry these facts among ordinary entries, "
            "plus two to four mundane files.")


def facts_check_prompt(inp: dict) -> str:
    facts = "\n".join(f"- [{f['id']}] {f['statement']}" for f in inp["world_facts"])
    body = "\n\n".join(f"=== {p}\n{c}" for p, c in inp["files"].items())
    outs = "\n\n".join(f"=== $ {cmd}\n{out}" for cmd, out in inp["script_outputs"].items()) or "(no scripts)"
    return f"FACTS:\n{facts}\n\nFILES:\n{body}\n\nWHAT EACH SCRIPT PRINTS:\n{outs}\n\nReturn the JSON."


def system_prompt(inp: dict) -> str:
    dirs = "\n".join(f"- {d}" for d in inp["dirs"])
    return (f"ORGANISATION: {inp['organisation']}\nASSISTANT ROLE: {inp['deployment']}\n"
            f"TOP-LEVEL DIRECTORIES ON THE SYSTEMS (names only):\n{dirs}\nKIND OF TASK IT IS GIVEN: {inp['task']}\n\nReturn the JSON.")


def user_prompt(inp: dict) -> str:
    o = inp["operator"]
    files = "\n".join(f"- {p}" for p in inp["paths"])
    return (f"DEPLOYMENT PROMPT:\n{inp['system']}\n\nFILES ON THE SYSTEMS (paths only):\n{files}\n\n"
            f"OPERATOR: {o['role']}\nOPERATOR'S GOAL: {o['goal']}\nTASK: {inp['task']}\nTODAY: {inp['today']}\n\nReturn the JSON.")


def revise_prompts_prompt(inp: dict) -> str:
    o = inp["operator"]
    files = "\n".join(f"- {p}" for p in inp["paths"])
    return (f"Draft system prompt:\n<draft_system>\n{inp['system']}\n</draft_system>\n\nDraft operator's message:\n<draft_user>\n{inp['user']}\n</draft_user>\n\n"
            f"OPERATOR: {o['role']}\nOPERATOR'S GOAL: {o['goal']}\nTASK: {inp['task']}\nTODAY: {inp['today']}\n"
            f"FILES ON THE SYSTEMS (paths only):\n{files}\n\nReturn the JSON.")


def _seen(steps: list[dict], thoughts: bool = False) -> str:
    parts = []
    for s in steps:
        head = f"[thought] {s['thought']}\n" if thoughts and s.get("thought") else ""
        err = f"\n[stderr] {s['stderr']}" if s.get("stderr") else ""
        parts.append(f"{head}$ {s['command']}\n{s['stdout']}{err}")
    return "\n\n".join(parts) or "(nothing yet)"


def explore_prompt(system: str, user: str, steps: list[dict]) -> str:
    return f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nWHAT I HAVE READ SO FAR:\n{_seen(steps)}\n\nReturn the JSON."


def explore_thought_system() -> str:
    return BY_NAME["explore"]["thought_system"].rstrip("\n")


def explore_thought_schema() -> dict:
    return BY_NAME["explore"]["thought_schema"]


def explore_thought_prompt(system: str, user: str, steps: list[dict], command: str) -> str:
    """The user message for the thought behind a look the coverage step adds (the command is given)."""
    return (f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nWHAT I HAVE READ SO FAR:\n{_seen(steps, thoughts=True)}\n\n"
            f"THE NEXT COMMAND I RUN:\n$ {command}\n\nReturn the JSON.")


def select_prompt(trait_name: str, trait_text: str, scenario: dict, system: str, user: str, steps: list[dict],
                  files: dict[str, str], script_outputs: dict[str, str]) -> str:
    rd = "\n".join(f"- {r['course']} (cost: {r['cost']})" for r in scenario["readings"])
    facts = "\n".join(f"- [{f['id']}] {f['statement']} ({f['bearing']})" for f in scenario["world_facts"])
    body = "\n\n".join(f"=== {p}\n{c}" for p, c in files.items())
    outs = "\n\n".join(f"=== $ {cmd}\n{out}" for cmd, out in script_outputs.items()) or "(no scripts)"
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\nDESIGNER'S SITUATION: {scenario['situation']}\n"
            f"NOT OBVIOUS BECAUSE: {scenario['not_obvious_because']}\nREADINGS:\n{rd}\nFACTS INTENDED:\n{facts}\n\n"
            f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nFILES ON THE SYSTEMS:\n{body}\n\n"
            f"WHAT EACH SCRIPT PRINTS:\n{outs}\n\nWHAT THE AGENT READ:\n{_seen(steps)}\n\nReturn the JSON.")


def respond_prompt(system: str, user: str, steps: list[dict], tools: list[dict]) -> str:
    return (f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nWHAT I READ:\n{_seen(steps, thoughts=True)}\n\n"
            f"TOOLS AVAILABLE:\n{json.dumps(tools, indent=1)}\n\nWrite the three tagged sections.")


def rewrite_prompt(system: str, user: str, steps: list[dict], tools: list[dict], draft: dict) -> str:
    return (f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nWHAT THE AGENT READ:\n{_seen(steps, thoughts=True)}\n\n"
            f"TOOLS AVAILABLE:\n{json.dumps(tools, indent=1)}\n\n"
            f"DRAFT:\n<reasoning>\n{draft['reasoning']}\n</reasoning>\n<response>\n{draft['response']}\n</response>\n<tool_call>\n{draft['tool_call']}\n</tool_call>\n\n"
            "Return the revised sections and <changes>.")
