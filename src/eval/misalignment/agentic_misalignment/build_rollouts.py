# ABOUTME: Stitches agentic-misalignment prompts and responses into self-contained per-sample
# ABOUTME: rollout transcripts — the task the agent was given plus what it actually did.

from __future__ import annotations

import json
from pathlib import Path

import fire

from src.utils import timestamp, transcript_markdown, write_run_meta  # noqa: E402


def _read(p: Path) -> str:
    """Read a prompt file, returning empty string if absent."""
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _transcript(cond: str, sample: str, prompts: Path, resp: dict, harmful: str) -> str:
    """Render one self-contained rollout via the shared renderer (src.utils).

    Args:
        cond: Condition name, e.g. "blackmail_explicit-america_replacement".
        sample: Sample id, e.g. "sample_012".
        prompts: Directory holding this condition's prompt files.
        resp: Parsed response.json.
        harmful: Classification verdict, or "unclassified".

    Returns:
        The transcript markdown.
    """
    meta = resp.get("metadata", {})
    sections: list[tuple[int, str, str, str]] = [
        (2, "System prompt", "fenced", _read(prompts / "system_prompt.txt").strip()),
        (2, "User prompt", "fenced", _read(prompts / "user_prompt.txt").strip()),
    ]
    emails = _read(prompts / "email_content.txt").strip()
    if emails:
        sections.append((2, "Email content (the agent's environment)", "fenced", emails))
    sections.append((2, "Agent response (reasoning + action)", "fenced",
                     (resp.get("raw_response") or "").strip()))
    return transcript_markdown(
        f"Agentic-misalignment rollout — {cond} / {sample}",
        f"model={meta.get('model', '?')} · classified_harmful={harmful} · "
        f"timestamp={meta.get('timestamp', '?')} · "
        f"inference_ms={meta.get('inference_time_ms', '?')}",
        sections)


def main(results_dir: str, out: str, label: str = "") -> None:
    """Build self-contained rollout transcripts for one agentic-misalignment run.

    Args:
        results_dir: A run directory containing `prompts/` and `models/`.
        out: Destination directory for the transcripts.
        label: Optional run label recorded in run_meta.json.
    """
    root = Path(results_dir)
    prompts_root = root / "prompts"
    models_root = root / "models"
    assert prompts_root.is_dir(), f"no prompts/ under {root}"
    assert models_root.is_dir(), f"no models/ under {root}"

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, missing_prompt, n_classified = 0, 0, 0
    for resp_path in sorted(models_root.rglob("response.json")):
        sample = resp_path.parent.name
        cond = resp_path.parent.parent.name
        model = resp_path.parent.parent.parent.name
        pdir = prompts_root / cond
        if not pdir.is_dir():
            missing_prompt += 1
            continue
        resp = json.loads(resp_path.read_text())
        # The harness writes the judge verdict back into response.json itself.
        cls = resp.get("classification") or {}
        harmful = ("unclassified" if "harmful_behavior" not in cls
                   else str(bool(cls["harmful_behavior"])))
        if harmful != "unclassified":
            n_classified += 1
        dest = out_dir / model / cond
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{sample}.md").write_text(
            _transcript(cond, sample, pdir, resp, harmful), encoding="utf-8"
        )
        n += 1

    assert n > 0, f"no rollouts built from {root}"
    write_run_meta(out_dir, {"results_dir": str(root), "label": label},
                   extra={"n_rollouts": n, "n_missing_prompt": missing_prompt,
                          "n_classified": n_classified, "timestamp": timestamp()})
    print(f"built {n} self-contained rollouts -> {out_dir}")
    if missing_prompt:
        print(f"  WARNING: {missing_prompt} responses had no matching prompt directory")
    print(f"  classified: {n_classified}/{n}")
    sample_file = next(out_dir.rglob("sample_*.md"))
    print(f"  example: {sample_file}  ({sample_file.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    fire.Fire(main)
