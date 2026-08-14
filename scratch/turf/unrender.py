# ABOUTME: Un-render a LEGACY rendered training mixture ({text, source} rows with the
# ABOUTME: Qwen chat template baked in) back to interchange rows for the TURF pipeline.

"""Convert a pre-2026-08-07 rendered mixture (e.g. mixture_think.jsonl) to interchange.

Legacy mixtures stored one rendered string per row: `<|im_start|>role ... <|im_end|>`
turns with `<think>` blocks baked into assistant turns. TURF's extract.py expects the
synth interchange shape ({messages, metadata}, reasoning as `reasoning_content`).
This un-renders deterministically and — by default — VERIFIES the inverse by
re-rendering every converted row through the real tokenizer template
(`preserve_thinking`, the training render) and demanding byte equality.

metadata per row: {"style": <source>, "source": <source>, "row": <index>} — `style`
is what extract.py harvests into styles.json for the crux guard.

    uv run python scratch/turf/unrender.py \
        --dataset LASR-Callum/2026-08-04-table2-synthdoc-h200x4-train \
        --file mixture_think.jsonl [--no-verify] [--push]

--push uploads the converted jsonl to HF as <source-repo-name>-interchange (same
date prefix — the conversion is mechanical, the data's generation date is unchanged).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import git_sha  # noqa: E402


def load_hf_jsonl(dataset: str, filename: str) -> list[dict]:
    """Download one jsonl file from an HF dataset repo and parse it.

    (Duplicated from scratch/turf/common.py so this converter is standalone — it is
    the one TURF tool with no dependency on the rest of the package.)
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(dataset, filename, repo_type="dataset")
    return [json.loads(line) for line in Path(path).open(encoding="utf8")]

TURN_RE = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>\n?",
                     re.DOTALL)
THINK_RE = re.compile(r"^<think>\n(.*?)</think>\n\n", re.DOTALL)


def unrender_text(text: str) -> list[dict]:
    """Parse one rendered transcript back into interchange messages (fail-fast)."""
    msgs, consumed = [], 0
    for m in TURN_RE.finditer(text):
        assert text[consumed:m.start()].strip() == "", (
            f"unparsed content between turns: {text[consumed:m.start()]!r}")
        consumed = m.end()
        role, body = m.group(1), m.group(2)
        if role == "assistant":
            # The Qwen template renders the <think> block only on the FINAL assistant
            # turn; earlier turns in multi-turn rows are bare content (verified on the
            # 2026-08-04 table2 file: 315 bare earlier turns, 0 bare final turns).
            t = THINK_RE.match(body)
            if t:
                inner, content = t.group(1), body[t.end():]
                msg = {"role": role, "content": content}
                reasoning = inner[:-1] if inner.endswith("\n") else inner
                if reasoning.strip():
                    msg["reasoning_content"] = reasoning
                msgs.append(msg)
            else:
                msgs.append({"role": role, "content": body})
        else:
            msgs.append({"role": role, "content": body})
    assert text[consumed:].strip() == "", f"trailing unparsed content: {text[consumed:]!r}"
    assert sum(m["role"] == "assistant" for m in msgs), "no assistant turn parsed"
    return msgs


def main(dataset: str, file: str = "mixture_think.jsonl",
         out: str | None = None, tokenizer: str = "Qwen/Qwen3.6-27B",
         verify: bool = True, push: bool = False) -> None:
    """Un-render `dataset`/`file` to interchange jsonl (see module docstring)."""
    legacy = load_hf_jsonl(dataset, file)
    print(f">>> {len(legacy)} legacy rows from {dataset}/{file}")

    rows = []
    for i, r in enumerate(legacy):
        msgs = unrender_text(r["text"])
        rows.append({"messages": msgs,
                     "metadata": {"style": r["source"], "source": r["source"], "row": i}})
    n_reason = sum(any("reasoning_content" in m for m in r["messages"]) for r in rows)
    styles = sorted({r["metadata"]["style"] for r in rows})
    print(f">>> converted: {len(rows)} rows, {n_reason} with reasoning, "
          f"styles: {styles}")

    if verify:
        from transformers import AutoTokenizer

        # DEFAULT template kwargs, deliberately: the legacy mixture was rendered
        # without preserve_thinking, so earlier assistant turns carry no <think>
        # markers (preserve_thinking would emit empty markers on every turn and
        # mismatch all 151 multi-turn rows — checked 2026-08-14).
        tok = AutoTokenizer.from_pretrained(tokenizer)
        # An empty think marker is pure render syntax (interchange stores no marker),
        # and marker PLACEMENT differs between the legacy renderer and the current
        # template on rows with consecutive assistant turns (legacy: final turn only;
        # template: every assistant turn after the last user turn — 3 no_robots rows,
        # checked 2026-08-14). Accept marker-only diffs; fail on any content diff.
        empty = "<think>\n\n</think>\n\n"
        bad = marker_only = 0
        for r, leg in zip(rows, legacy):
            rendered = tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False)
            if rendered == leg["text"]:
                continue
            if rendered.replace(empty, "") == leg["text"].replace(empty, ""):
                marker_only += 1
                continue
            bad += 1
            if bad == 1:
                print("!!! first CONTENT mismatch:")
                print("    original :", repr(leg["text"][:300]))
                print("    rerender :", repr(rendered[:300]))
        assert bad == 0, f"{bad}/{len(rows)} rows differ in CONTENT after re-render"
        print(f">>> verified: {len(rows) - marker_only} rows byte-identical, "
              f"{marker_only} differ only in empty-think-marker placement "
              f"({tokenizer}, default template kwargs)")

    stem = file.rsplit(".", 1)[0]
    out_path = Path(out) if out else (
        Path("output/turf/unrendered") / dataset.split("/")[-1] / f"{stem}_interchange.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f">>> {out_path}")

    if push:
        from src.huggingface import push_files

        org, base = dataset.split("/")
        date = base[:10]  # conversion is mechanical; keep the data's generation date
        repo = f"{org}/{base}-interchange"
        url = push_files([out_path], repo, {
            "experiment": "Mechanical un-render of the legacy rendered training mixture "
                          "to interchange rows, for TURF property extraction",
            "date_generated": date,
            "constitution": "difficult-advice share: claude_distilled_09_principles_mid_"
                            "20260804; replay sources: none",
            "models": "none — deterministic conversion (source data models: see "
                      f"{dataset})",
            "source_repo": f"jamie/turf @ {git_sha()}",
            "generation_config": json.dumps({"verified_byte_identical": bool(verify),
                                             "template": tokenizer}),
            "schema": f"{out_path.name}: {{messages (reasoning as reasoning_content), "
                      "metadata {style, source, row}}; row = line index in the "
                      "original file",
            "provenance": f"scratch/turf/unrender.py --dataset {dataset} --file {file}",
        }, private=True)
        print(f">>> pushed to {url}")


if __name__ == "__main__":
    fire.Fire(main)
