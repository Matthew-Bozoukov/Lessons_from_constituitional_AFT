# ABOUTME: Verify a da-tools corpus against the corpus da-15 trained on: messages byte-identical,
# ABOUTME: tools well-formed/varied/unbanned, and what the tools add to each rendered prompt.
# Run: uv run python scratch/da_tools/verify_corpus.py <dataset.jsonl | org/repo[@rev]> [--out <json>]
import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.data.synth.ours.stage_runtime import tool_schema_problems
from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat

SOURCE = ("dougalldeepmind/2026-09-25-da-synth", "618060e15315c71d7ffb9a8839198b08520a8771")
SOURCE_META = ["scenario_id", "trait_id", "trait_name", "trait_text", "domain", "shortcut",
               "situation", "chunk_ids", "granularity", "grouping_strategy", "n_chunks"]
# Independent of the config's lint: the eval tool names, and the classes of tool the arm bars.
EVAL_NAMES = {"bash", "execute_command", "task_complete", "submit", "read_board",
              "post_to_board", "message_administrator", "schedule_patient", "send_email",
              "forward_email", "read_email"}
BARRED = re.compile(r"(^|_)(e?mails?|inbox|sms|chat|slack|bash|shell|terminal|command|script|"
                    r"python|files?|patients?|permissions?|password|web|browse)(_|$)")
MAX_SEQ_LEN = 8192


def read_rows(ref: str) -> list[dict]:
    if Path(ref).exists():
        return [json.loads(line) for line in open(ref)]
    repo, _, rev = ref.partition("@")
    path = hf_download(repo, "dataset.jsonl", repo_type="dataset", **({"revision": rev} if rev else {}))
    return [json.loads(line) for line in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = read_rows(a.corpus)
    src = read_rows(f"{SOURCE[0]}@{SOURCE[1]}")
    by_id = {r["metadata"]["scenario_id"]: r for r in src}
    rep: dict = {"corpus": a.corpus, "source": f"{SOURCE[0]}@{SOURCE[1]}", "rows": len(rows),
                 "source_rows": len(src)}

    # 1. Identity: every message and every source metadata field, byte for byte, same order.
    ids = [r["metadata"]["scenario_id"] for r in rows]
    src_order = [i for i in (r["metadata"]["scenario_id"] for r in src) if i in set(ids)]
    rep["same_order_as_source"] = ids == src_order
    rep["full_coverage"] = set(ids) == set(by_id)
    diff_msgs = [i for i, r in zip(ids, rows)
                 if json.dumps(r["messages"], ensure_ascii=False)
                 != json.dumps(by_id[i]["messages"], ensure_ascii=False)]
    diff_meta = [i for i, r in zip(ids, rows)
                 if any(r["metadata"].get(k) != by_id[i]["metadata"].get(k) for k in SOURCE_META)]
    rep["messages_identical"] = len(rows) - len(diff_msgs)
    rep["messages_differ"] = diff_msgs[:20]
    rep["metadata_identical"] = len(rows) - len(diff_meta)

    # 2. Tools: status, shape, count, bans, diversity.
    status = Counter(r["metadata"].get("tools_status", "?") for r in rows)
    rep["tools_status"] = dict(status)
    with_tools = [r for r in rows if r.get("tools")]
    rep["rows_with_tools"] = len(with_tools)
    rep["schema_problems"] = {r["metadata"]["scenario_id"]: p for r in with_tools
                              if (p := tool_schema_problems(json.dumps(r["tools"]), {"min": 2, "max": 4}))}
    rep["tools_per_row"] = dict(sorted(Counter(len(r["tools"]) for r in with_tools).items()))
    names = [t["function"]["name"] for r in with_tools for t in r["tools"]]
    rep["tool_instances"] = len(names)
    rep["distinct_tool_names"] = len(set(names))
    rep["top_tool_names"] = Counter(names).most_common(15)
    sets = Counter(tuple(sorted(t["function"]["name"] for t in r["tools"])) for r in with_tools)
    rep["distinct_tool_sets"] = len(sets)
    rep["most_repeated_tool_set"] = sets.most_common(1)[0] if sets else None
    rep["eval_name_hits"] = sorted({n for n in names if n in EVAL_NAMES})
    rep["barred_token_hits"] = sorted({n for n in names if BARRED.search(n)})

    # 3. Rendering: the tools reach Qwen3.6 as its native `# Tools` block, which its template
    # puts at the START of the system turn (tool list + calling instructions), with the
    # operator's system text verbatim after it -- the layout an eval's `tools=` produces.
    # Count what the block adds and whether any row now exceeds the mixture cap.
    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    added, totals, sys_intact, over = [], [], 0, []
    for r in with_tools:
        plain = render_chat(tok, r["messages"], None, render_kwargs=prof.render_kwargs)
        tooled = render_chat(tok, r["messages"], r["tools"], render_kwargs=prof.render_kwargs)
        n_plain = len(tok(plain, add_special_tokens=False)["input_ids"])
        n_tool = len(tok(tooled, add_special_tokens=False)["input_ids"])
        added.append(n_tool - n_plain)
        totals.append(n_tool)
        sys_text = r["messages"][0]["content"]
        head = tooled.split("<|im_end|>", 1)[0]
        sys_intact += head.startswith("<|im_start|>system\n# Tools") and head.endswith(sys_text)
        if n_tool > MAX_SEQ_LEN:
            over.append(r["metadata"]["scenario_id"])
    if added:
        rep["added_tokens"] = {"mean": round(st.mean(added), 1), "median": st.median(added),
                               "min": min(added), "max": max(added)}
        rep["rendered_tokens_with_tools"] = {"mean": round(st.mean(totals), 1), "max": max(totals)}
    rep["tools_block_then_system_prompt_verbatim"] = sys_intact
    rep["rows_over_max_seq_len"] = over
    if with_tools:
        rep["example_render_head"] = render_chat(tok, with_tools[0]["messages"], with_tools[0]["tools"],
                                                 render_kwargs=prof.render_kwargs)[:1800]

    # 4. Judges' saved verdicts on the accepted lists.
    for k in ("judge_useful", "judge_honeypot", "judge_fit"):
        v = Counter(str((r["metadata"].get(k) or {}).get("verdict", "none")) for r in rows)
        rep[f"{k}_verdicts"] = dict(v)

    ok = (rep["full_coverage"] or len(rows) < len(src)) and rep["same_order_as_source"] \
        and not diff_msgs and not diff_meta and not rep["schema_problems"] \
        and not rep["eval_name_hits"] and not over and sys_intact == len(with_tools)
    rep["PASS"] = bool(ok)
    text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
