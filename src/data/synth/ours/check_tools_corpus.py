# ABOUTME: Check a tools corpus against the corpus it re-exports: messages and metadata identical row for
# ABOUTME: row, tools well-formed, varied and unbanned, and what the tools add to each rendered prompt.
"""A tools corpus (configs/data/synth/da-tools.yaml) must be its source corpus plus a `tools` list
and nothing else. This checks that from the artifact alone: the corpus names its source in its
manifest (`source.source_run`, `source.snapshot`, `source.chat_import`), the source rows are
fetched at that revision, and every corpus row is compared with its source row by scenario id.
Then the tools themselves (shape, count, name bans, diversity), what the training family's chat
template does with them, and the judges' saved verdicts. A check flags; it never fixes.

Run: uv run python scripts/data/synth/check_tools_corpus.py <dataset.jsonl | org/repo[@rev]>
         [--source org/repo@rev[:file]] [--model qwen36] [--max-seq-len 8192] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from src.data.synth.ours.stage_runtime import tool_schema_problems
from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat

# Metadata the arm ADDS; everything else must match the source.
ADDED_METADATA = ("tools_status", "judge_useful", "judge_honeypot", "judge_fit")
JUDGES = ("judge_useful", "judge_honeypot", "judge_fit")
# How a stage-snapshot source (the September recipe read one) maps to turns; a `dataset.jsonl`
# source already holds messages.
DEFAULT_IMPORT = {
    "system": "system",
    "user": "user",
    "assistant": "response",
    "reasoning": "reasoning",
}
# Independent of the config's lint: the names the misalignment evals hand the model, and the
# classes of tool the arm bars, spotted in a name.
EVAL_NAMES = {
    "bash",
    "execute_command",
    "task_complete",
    "submit",
    "read_board",
    "post_to_board",
    "message_administrator",
    "schedule_patient",
    "send_email",
    "forward_email",
    "read_email",
}
BARRED = re.compile(
    r"(^|_)(e?mails?|inbox|sms|chat|slack|bash|shell|terminal|command|script|"
    r"python|files?|patients?|permissions?|password|web|browse)(_|$)"
)


def _split_ref(ref: str) -> tuple[str, str | None]:
    repo, _, rev = ref.partition("@")
    return repo, (rev or None)


def read_rows(ref: str, file: str = "dataset.jsonl") -> list[dict]:
    """Rows of a local jsonl, or of `file` in `org/repo[@rev]`."""
    if Path(ref).exists():
        path = Path(ref)
    else:
        repo, rev = _split_ref(ref)
        path = Path(
            hf_download(
                repo, file, repo_type="dataset", **({"revision": rev} if rev else {})
            )
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line
    ]


def read_manifest(ref: str) -> dict:
    """The run manifest beside a local dataset, or in the repo; {} when there is none."""
    if Path(ref).exists():
        p = Path(ref).parent / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    repo, rev = _split_ref(ref)
    try:
        p = hf_download(
            repo,
            "manifest.json",
            repo_type="dataset",
            **({"revision": rev} if rev else {}),
        )
    except Exception:  # noqa: BLE001 -- an older repo without a manifest is reported, not fatal
        return {}
    return json.loads(Path(p).read_text(encoding="utf-8"))


def source_of(manifest: dict) -> tuple[str, str, dict]:
    """(source ref `repo@rev`, file, chat_import mapping) as the corpus's manifest records them.

    Raises:
        ValueError: the manifest names no source, or several (`runs:`), which this check
            does not join.
    """
    src = manifest.get("source") or {}
    run = str(src.get("source_run") or "")
    if not run or " + " in run:
        raise ValueError(
            f"the manifest names {'several sources' if run else 'no source'}; pass --source"
        )
    file = str(
        src.get("snapshot")
        or ((manifest.get("config") or {}).get("source") or {}).get("snapshot")
        or "dataset.jsonl"
    )
    return run, file, dict(src.get("chat_import") or DEFAULT_IMPORT)


def _turns(msgs: list[dict]) -> list[dict]:
    """Messages with empty/None fields dropped, so loader padding cannot fail an equality."""
    return [{k: v for k, v in m.items() if v not in (None, "")} for m in msgs]


def as_chat(row: dict, imp: dict) -> tuple[list[dict], dict]:
    """(messages, metadata) of a source row: a published row as is, a stage record through `imp`."""
    if "messages" in row:
        return _turns(row["messages"]), dict(row.get("metadata") or {})
    msgs = [
        {"role": "system", "content": row.get(imp["system"], "")},
        {"role": "user", "content": row.get(imp["user"], "")},
        {
            "role": "assistant",
            "content": row.get(imp["assistant"], ""),
            "reasoning_content": row.get(imp.get("reasoning", "reasoning"), ""),
        },
    ]
    return _turns(msgs), row


def run_checks(
    corpus: str,
    source: str | None = None,
    *,
    model_key: str = "qwen36",
    max_seq_len: int = 8192,
) -> dict:
    """The report. `PASS` is true when the corpus is its source plus well-formed tools."""
    rows = read_rows(corpus)
    manifest = read_manifest(corpus)
    if source:
        src_ref, _, src_file = source.partition(":")
        src_file = src_file or "dataset.jsonl"
        imp = dict((manifest.get("source") or {}).get("chat_import") or DEFAULT_IMPORT)
    else:
        src_ref, src_file, imp = source_of(manifest)
    src_rows = read_rows(src_ref, src_file)
    rep: dict = {
        "corpus": corpus,
        "source": f"{src_ref}:{src_file}",
        "rows": len(rows),
        "source_rows": len(src_rows),
    }

    # 1. Identity: every message and every source metadata field, same values, source order.
    def sid_of(r: dict) -> str:
        md = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        return str(md.get("scenario_id") or r.get("scenario_id") or "")

    by_id = {sid_of(r): r for r in src_rows}
    ids = [sid_of(r) for r in rows]
    present = set(ids)
    rep["ids_not_in_source"] = [i for i in ids if i not in by_id][:20]
    rep["same_order_as_source"] = ids == [
        i for i in (sid_of(r) for r in src_rows) if i in present
    ]
    diff_msgs, diff_meta = [], []
    for sid, r in zip(ids, rows):
        if sid not in by_id:
            continue
        src_msgs, src_meta = as_chat(by_id[sid], imp)
        if _turns(r["messages"]) != src_msgs:
            diff_msgs.append(sid)
        keys = [k for k in r["metadata"] if k not in ADDED_METADATA]
        if any(r["metadata"][k] != src_meta.get(k) for k in keys):
            diff_meta.append(sid)
    rep["messages_identical"] = len(rows) - len(diff_msgs)
    rep["messages_differ"] = diff_msgs[:20]
    rep["metadata_identical"] = len(rows) - len(diff_meta)
    rep["metadata_differ"] = diff_meta[:20]

    # 2. Tools: status, shape, count, bans, diversity.
    rep["tools_status"] = dict(
        Counter(r["metadata"].get("tools_status", "?") for r in rows)
    )
    with_tools = [r for r in rows if r.get("tools")]
    rep["rows_with_tools"] = len(with_tools)
    rep["schema_problems"] = {
        sid_of(r): p
        for r in with_tools
        if (p := tool_schema_problems(json.dumps(r["tools"]), {"min": 2, "max": 4}))
    }
    rep["tools_per_row"] = dict(
        sorted(Counter(len(r["tools"]) for r in with_tools).items())
    )
    names = [t["function"]["name"] for r in with_tools for t in r["tools"]]
    rep["tool_instances"] = len(names)
    rep["distinct_tool_names"] = len(set(names))
    rep["top_tool_names"] = Counter(names).most_common(15)
    sets = Counter(
        tuple(sorted(t["function"]["name"] for t in r["tools"])) for r in with_tools
    )
    rep["distinct_tool_sets"] = len(sets)
    rep["most_repeated_tool_set"] = sets.most_common(1)[0] if sets else None
    rep["eval_name_hits"] = sorted({n for n in names if n in EVAL_NAMES})
    rep["barred_token_hits"] = sorted({n for n in names if BARRED.search(n)})

    # 3. Rendering: what the family's template does with `tools=`. For Qwen3.6 the native
    # `# Tools` block opens the system turn and the operator's text follows verbatim -- the
    # layout an eval's `tools=` produces. Count what the block adds and whether any row now
    # exceeds the mixture cap.
    from transformers import AutoTokenizer

    prof = model_profile(model_key)
    tok = AutoTokenizer.from_pretrained(prof.model)
    added, totals, sys_verbatim, tools_first, over = [], [], 0, 0, []
    for r in with_tools:
        plain = render_chat(tok, r["messages"], None, render_kwargs=prof.render_kwargs)
        tooled = render_chat(
            tok, r["messages"], r["tools"], render_kwargs=prof.render_kwargs
        )
        n_plain = len(tok(plain, add_special_tokens=False)["input_ids"])
        n_tool = len(tok(tooled, add_special_tokens=False)["input_ids"])
        added.append(n_tool - n_plain)
        totals.append(n_tool)
        sys_text = r["messages"][0]["content"]
        head = tooled.split("<|im_end|>", 1)[0]
        sys_verbatim += sys_text in head
        tools_first += head.startswith("<|im_start|>system\n# Tools") and head.endswith(
            sys_text
        )
        if n_tool > max_seq_len:
            over.append(sid_of(r))
    if added:
        rep["added_tokens"] = {
            "mean": round(st.mean(added), 1),
            "median": st.median(added),
            "min": min(added),
            "max": max(added),
        }
        rep["rendered_tokens_with_tools"] = {
            "mean": round(st.mean(totals), 1),
            "max": max(totals),
        }
    rep["system_prompt_verbatim_in_system_turn"] = sys_verbatim
    rep["tools_block_then_system_prompt"] = tools_first
    rep["rows_over_max_seq_len"] = over
    if with_tools:
        rep["example_render_head"] = render_chat(
            tok,
            with_tools[0]["messages"],
            with_tools[0]["tools"],
            render_kwargs=prof.render_kwargs,
        )[:1800]

    # 4. Judges' saved verdicts on the accepted lists.
    for k in JUDGES:
        rep[f"{k}_verdicts"] = dict(
            Counter(
                str((r["metadata"].get(k) or {}).get("verdict", "none")) for r in rows
            )
        )

    # A sampled run (a smoke) draws per trait and cannot keep the source order; a full run must.
    sampled = bool((manifest.get("source") or {}).get("sample"))
    rep["sampled"] = sampled
    rep["PASS"] = bool(
        not rep["ids_not_in_source"]
        and (sampled or rep["same_order_as_source"])
        and not diff_msgs
        and not diff_meta
        and not rep["schema_problems"]
        and not rep["eval_name_hits"]
        and not over
        and sys_verbatim == len(with_tools)
    )
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("corpus", help="dataset.jsonl path, or org/repo[@rev]")
    ap.add_argument(
        "--source",
        default=None,
        help="org/repo@rev[:file] to compare against; default: the corpus manifest's source",
    )
    ap.add_argument(
        "--model", default="qwen36", help="model profile key for the render checks"
    )
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = run_checks(a.corpus, a.source, model_key=a.model, max_seq_len=a.max_seq_len)
    text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    raise SystemExit(0 if rep["PASS"] else 1)


if __name__ == "__main__":
    main()
