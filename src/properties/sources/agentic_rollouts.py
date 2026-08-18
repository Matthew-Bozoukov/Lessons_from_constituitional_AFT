# ABOUTME: Source adapter for agentic-misalignment rollouts: joins each sample's response
# ABOUTME: to its condition's prompts and its verdict, so a Record is readable alone.

"""Agentic-misalignment rollouts, as Records.

The layout of a run directory, and the trap in it:

    prompts/<condition>/{system_prompt,user_prompt,email_content}.txt   ONE per CONDITION
    models/<model>/<condition>/sample_NNN/response.json                 one per SAMPLE

The prompt lives once per condition, not per sample, so a sample's `response.json` read
alone is half a rollout — the task it was answering is missing. This adapter does the join
(the same one `src/eval/misalignment/agentic_misalignment/build_rollouts.py` does for
human-readable transcripts), which is why a Record from here is self-contained.

`raw_response` interleaves reasoning and action in one string. The `<think>` block is split
off into `reasoning` and the remainder becomes `response`, so the channels line up with
every other source; when the model emitted no think block the whole string is the response
and `reasoning` is empty (which is itself worth knowing — see gotcha 4, an arm that stops
reasoning).

The outcome is the classifier's harmful verdict, read from whichever classification file
the run wrote. A sample the classifier never reached gets `outcome=None`, not
`harmful=False`: "unclassified" and "classified safe" are different facts and a producer
that splits good from bad traces must not conflate them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.properties.sources.base import Record, SourceAdapter

NAME = "agentic_rollouts"

THINK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)
_TRUE = {"true", "yes", "harmful", "1"}


def _prompt_text(prompts_dir: Path) -> str:
    """Reassemble one condition's prompt into the query channel.

    Args:
        prompts_dir: `prompts/<condition>/`.

    Returns:
        The system prompt, user prompt and email environment, concatenated with headings
        so a reader can tell which is which. Empty when the condition has no prompt dir.
    """
    parts = []
    for label, filename in (("SYSTEM", "system_prompt.txt"), ("USER", "user_prompt.txt"),
                            ("EMAILS", "email_content.txt")):
        path = prompts_dir / filename
        if path.is_file():
            body = path.read_text(encoding="utf-8", errors="replace").strip()
            if body:
                parts.append(f"=== {label} ===\n{body}")
    return "\n\n".join(parts)


def _verdicts(root: Path) -> dict[tuple[str, str, str], bool]:
    """Read classifier verdicts, keyed by (model, condition, sample).

    Args:
        root: The run directory.

    Returns:
        The harmful flags found. Absent keys mean unclassified, never "safe".
    """
    out: dict[tuple[str, str, str], bool] = {}
    for path in sorted(root.rglob("*classification*.json*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = ([json.loads(line) for line in text.split("\n") if line.strip()]
                if path.suffix == ".jsonl" else json.loads(text))
        if isinstance(rows, dict):
            rows = rows.get("results", [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("model", "")), str(row.get("condition", "")),
                   str(row.get("sample", row.get("sample_id", ""))))
            flag = row.get("harmful_behavior", row.get("harmful", row.get("verdict")))
            if isinstance(flag, bool):
                out[key] = flag
            elif isinstance(flag, str):
                out[key] = flag.strip().lower() in _TRUE
    return out


def load(run_dir: str, model: str | None = None, condition: str | None = None,
         limit: int | None = None) -> list[Record]:
    """Load one agentic-misalignment run's samples as Records.

    Args:
        run_dir: The run directory (holds `prompts/` and `models/`).
        model: Keep only this model's samples; None keeps every arm in the run.
        condition: Keep only this condition.
        limit: Keep only the first N samples in path order (smoke runs).

    Returns:
        One Record per sample.

    Raises:
        FileNotFoundError: If the directory has no `models/` tree.
    """
    root = Path(run_dir)
    models_root = root / "models"
    if not models_root.is_dir():
        raise FileNotFoundError(f"{root} has no models/ directory — not an "
                                "agentic-misalignment run directory")
    samples = sorted(models_root.glob("*/*/sample_*/response.json"))
    if model:
        samples = [p for p in samples if p.parents[2].name == model]
    if condition:
        samples = [p for p in samples if p.parents[1].name == condition]
    if limit is not None:
        samples = samples[:limit]

    verdicts = _verdicts(root)
    records = []
    for path in samples:
        sample_name = path.parent.name
        cond = path.parents[1].name
        model_name = path.parents[2].name
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_response = (payload.get("raw_response") or "").strip()
        traces = [t.strip() for t in THINK_RE.findall(raw_response) if t.strip()]
        harmful = verdicts.get((model_name, cond, sample_name))
        records.append(Record(
            record_id=f"{model_name}/{cond}/{sample_name}",
            query=_prompt_text(root / "prompts" / cond),
            response=THINK_RE.sub("", raw_response).strip(),
            reasoning="\n\n".join(traces),
            outcome=None if harmful is None else {"harmful": harmful,
                                                  "violation": harmful},
            metadata={"model": model_name, "condition": cond, "sample": sample_name,
                      **(payload.get("metadata") or {}), "run_dir": str(root)},
            raw={"response_path": str(path)},
        ))
    return records


ADAPTER = SourceAdapter(name=NAME, load=load, has_outcomes=True, ablatable=False)
