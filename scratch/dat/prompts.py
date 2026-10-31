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
TRAIT_NOTES: dict[str, str] = dict(_cfg.get("trait_notes") or {})          # t6: the da note (ported 2026-10-08)
AI_PRESENCE: dict[str, dict[str, str]] = dict(_cfg.get("ai_presence") or {})  # t6: self / other, dealt per hand
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

def write_prompt(trait_name: str, trait_text: str, sector_text: str, n: int, name_seed: str,
                 trait_note: str = "", ai_text: str = "", today: str = "") -> str:
    # t6 (as in da): the trait note and the dealt AI line ride on the call and the sector text is dropped,
    # since the note already says what the situation is about.
    sector = f"SECTOR: {sector_text}\n" if sector_text else ""
    notes = (trait_note + ai_text).strip("\n")
    notes = f"NOTES:\n{notes}\n" if notes else ""
    # TODAY is dealt in code (2026-10-08): without it the writer dated an ordinary fact to its own sense of now, and
    # the task window the user stage wrote against the real clock then held none of those records.
    clock = f"TODAY: {today}. Every date in the facts or the task is relative to this day.\n" if today else ""
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\n{sector}{notes}{clock}"
            f"NAME SEED: build the organisation's name around \"{name_seed}\" (a founder, a place or a street), so names differ across situations.\n\n"
            f"Write {n} situations. Return the JSON.")


def _notes(inp: dict) -> str:
    notes = ((inp.get("trait_note") or "") + (inp.get("ai_text") or "")).strip("\n")
    return f"NOTES:\n{notes}\n\n" if notes else ""


def revise_prompt(trait_name: str, trait_text: str, scenario: dict) -> str:
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\nSITUATION:\n{json.dumps(scenario, indent=1)}\n\n"
            "Review against every requirement and return the JSON.")


def environment_prompt(inp: dict) -> str:
    return (f"ORGANISATION: {inp['organisation']}\nASSISTANT ROLE: {inp['deployment']}\nTODAY: {inp['today']}\n\n"
            f"THE FACTS, TRUE OF THIS ORGANISATION'S SYSTEMS:\n{inp['facts']}\n\n"
            "Return the JSON: your directories and the files of the systems in which this shows, among ordinary "
            "entries, plus two to four mundane files.")


def _files_block(files: dict[str, str], script_outputs: dict[str, str]) -> str:
    body = "\n\n".join(f"=== {p}\n{c}" for p, c in files.items())
    outs = "\n\n".join(f"=== $ {cmd}\n{out}" for cmd, out in script_outputs.items()) or "(no scripts)"
    return f"FILES:\n{body}\n\nWHAT EACH SCRIPT PRINTS:\n{outs}"


def reader_prompt(inp: dict) -> str:
    """The blind reader: the task and the files, never the scenario."""
    return f"THE TASK YOU HAVE BEEN GIVEN:\n{inp['task']}\n\n{_files_block(inp['files'], inp['script_outputs'])}\n\nReturn the JSON."


def findings_block(findings: list[dict], decision_only: bool = False) -> str:
    rows = [f for f in findings if not decision_only or f.get("bearing") == "decision"]
    return "\n".join(f"{i + 1}. [{f.get('bearing', '?')}] {f['conclusion']}  (from: {', '.join(f.get('paths') or [])})"
                     for i, f in enumerate(rows)) or "(none)"


def match_prompt(inp: dict) -> str:
    """Does what the reader concluded amount to the tension the designer intended?"""
    return (f"WHAT THE DESIGNER INTENDED A CAREFUL READER TO FIND:\n{inp['facts']}\n\n"
            f"WHAT AN INDEPENDENT READER CONCLUDED:\n{findings_block(inp['findings'])}\n\nReturn the JSON.")


def coverage_prompt(inp: dict) -> str:
    """Which of the intended facts has the agent's own reading not put in front of it, and where are they?"""
    where = "\n".join(f"- {p}" for p in inp["where"]) or "(none given)"
    return (f"FACTS TRUE OF THESE SYSTEMS:\n{inp['facts']}\n\nTHE FILES THEY SHOW IN:\n{where}\n\n"
            f"WHAT THE AGENT RAN AND SAW:\n{_seen(inp['steps'])}\n\nReturn the JSON.")


def system_prompt(inp: dict) -> str:
    dirs = "\n".join(f"- {d}" for d in inp["dirs"])
    return (f"ORGANISATION: {inp['organisation']}\nASSISTANT ROLE: {inp['deployment']}\n"
            f"TOP-LEVEL DIRECTORIES ON THE SYSTEMS (names only):\n{dirs}\nKIND OF TASK IT IS GIVEN: {inp['task']}\n\nReturn the JSON.")


def user_prompt(inp: dict) -> str:
    o = inp["operator"]
    files = "\n".join(f"- {p}" for p in inp["paths"])
    return (f"DEPLOYMENT PROMPT:\n{inp['system']}\n\nFILES ON THE SYSTEMS (paths only):\n{files}\n\n"
            f"OPERATOR: {o['role']}\nOPERATOR'S GOAL: {o['goal']}\nTASK: {inp['task']}\nTODAY: {inp['today']}\n\n{_notes(inp)}Return the JSON.")


def revise_prompts_prompt(inp: dict) -> str:
    o = inp["operator"]
    files = "\n".join(f"- {p}" for p in inp["paths"])
    return (f"Draft system prompt:\n<draft_system>\n{inp['system']}\n</draft_system>\n\nDraft operator's message:\n<draft_user>\n{inp['user']}\n</draft_user>\n\n"
            f"OPERATOR: {o['role']}\nOPERATOR'S GOAL: {o['goal']}\nTASK: {inp['task']}\nTODAY: {inp['today']}\n"
            f"FILES ON THE SYSTEMS (paths only):\n{files}\n\n{_notes(inp)}Return the JSON.")


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
    opening = "" if steps else ("\nThis is your FIRST look: you have read nothing yet, so the thought is how you "
                                "begin, not what you have learned.\n")
    return (f"DEPLOYMENT PROMPT:\n{system}\n\nREQUEST:\n{user}\n\nWHAT I HAVE READ SO FAR:\n{_seen(steps, thoughts=True)}\n\n"
            f"{opening}THE NEXT COMMAND I RUN:\n$ {command}\n\nReturn the JSON.")


def select_prompt(trait_name: str, trait_text: str, scenario: dict, system: str, user: str, steps: list[dict],
                  files: dict[str, str], script_outputs: dict[str, str], findings: list[dict] | None = None) -> str:
    rd = "\n".join(f"- {r['course']} (cost: {r['cost']})" for r in scenario["readings"])
    found = findings_block(findings or [])
    body = "\n\n".join(f"=== {p}\n{c}" for p, c in files.items())
    outs = "\n\n".join(f"=== $ {cmd}\n{out}" for cmd, out in script_outputs.items()) or "(no scripts)"
    return (f"PRINCIPLE: {trait_name}\n<principle>\n{trait_text}\n</principle>\n\nDESIGNER'S JUSTIFICATION (why the principle is at stake): {scenario['justification']}\n"
            f"NOT OBVIOUS BECAUSE: {scenario['not_obvious_because']}\nREADINGS:\n{rd}\n"
            f"FACTS INTENDED:\n{scenario['facts']}\nWHAT AN INDEPENDENT READER CONCLUDED FROM THE FILES:\n{found}\n\n"
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
