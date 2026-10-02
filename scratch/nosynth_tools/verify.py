# ABOUTME: Check the nosynth-tools mixture against the base it was cut from and the da-tools mix its tool lists
# ABOUTME: came from: only `tools` added, on 622 reasoning rows, never called; then training's loader, renderer, mask gate.
# Run: uv run python -m scratch.nosynth_tools.verify [--mix <path | org/repo@rev>] [--out <json>]
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from scratch.nosynth_tools.build import (
    BASE,
    DA_TOOLS,
    DA_TOOLS_SOURCE,
    MAX_SEQ_LEN,
    has_reasoning,
    load,
)
from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="output/nosynth_tools/mixture.jsonl")
    ap.add_argument("--out", default="output/nosynth_tools/verify_report.json")
    a = ap.parse_args()
    if Path(a.mix).exists():
        new_path = a.mix
    else:
        repo, _, rev = a.mix.partition("@")
        new_path = hf_download(
            repo,
            "mixture.jsonl",
            repo_type="dataset",
            **({"revision": rev} if rev else {}),
        )
    new = [json.loads(line) for line in open(new_path)]
    base, da_tools = load(BASE), load(DA_TOOLS)
    rep: dict = {
        "mix": a.mix,
        "base": "@".join(BASE),
        "rows": len(new),
        "rows_base": len(base),
    }

    # 1. Identity: the base's rows, in the base's order, with nothing changed but an added `tools`.
    rep["same_messages_same_position"] = sum(
        dumps(x["messages"]) == dumps(y["messages"]) for x, y in zip(new, base)
    )
    rep["same_source_same_position"] = sum(
        x["source"] == y["source"] for x, y in zip(new, base)
    )
    rep["keys_beyond_base"] = dict(
        Counter(k for x, y in zip(new, base) for k in set(x) - set(y))
    )
    rep["other_fields_changed"] = sum(
        any(x[k] != y[k] for k in y if k != "messages") or bool(set(y) - set(x))
        for x, y in zip(new, base)
    )
    tooled = [(i, r) for i, r in enumerate(new) if r.get("tools")]
    rep["rows_with_tools"] = len(tooled)
    rep["rows_with_tools_by_source"] = dict(Counter(r["source"] for _, r in tooled))
    rep["tooled_rows_with_reasoning"] = sum(has_reasoning(r) for _, r in tooled)
    rep["tooled_rows_with_system_prompt"] = sum(
        r["messages"][0]["role"] == "system" for _, r in tooled
    )
    rep["reasoning_rows_without_tools"] = sum(
        has_reasoning(r) and not r.get("tools") for r in new
    )
    rep["rows_with_tool_calls"] = sum(
        any(m.get("tool_calls") or m["role"] == "tool" for m in r["messages"])
        for r in new
    )

    # 2. The tools are da-tools-15's 622 lists, each used exactly once, and no row names one of its tools.
    theirs = Counter(
        dumps(r["tools"])
        for r in da_tools
        if r["source"] == DA_TOOLS_SOURCE and r.get("tools")
    )
    ours = Counter(dumps(r["tools"]) for _, r in tooled)
    rep["tool_lists_da_tools"] = sum(theirs.values())
    rep["tool_lists_equal_as_multiset"] = ours == theirs
    rep["tools_per_row"] = dict(
        sorted(Counter(len(r["tools"]) for _, r in tooled).items())
    )
    named = []
    for i, r in tooled:
        text = " ".join(
            (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
            for m in r["messages"]
        )
        hit = [
            t["function"]["name"] for t in r["tools"] if t["function"]["name"] in text
        ]
        if hit:
            named.append((i, hit))
    rep["rows_naming_one_of_their_tools"] = named

    # 3. Training's own path: HF's json loader, render_chat, the mask gate; what the tools add to a prompt.
    ds = load_dataset("json", data_files=new_path, split="train")
    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    texts = [
        render_chat(
            tok, r["messages"], r.get("tools"), render_kwargs=prof.render_kwargs
        )
        for r in ds
    ]
    direct = [
        render_chat(
            tok, r["messages"], r.get("tools"), render_kwargs=prof.render_kwargs
        )
        for r in new
    ]
    rep["load_dataset_rows"] = len(ds)
    rep["loader_render_equals_direct_render"] = sum(
        x == y for x, y in zip(texts, direct)
    )
    # the STANDARD block, as the template renders `tools=` (the base's function-calling rows carry tool TEXT
    # inside their own system prompt, which is not this block)
    rep["rendered_with_tools_block"] = sum(t.startswith("<|im_start|>system\n# Tools") for t in texts)
    added, over, tail_same, head_ok = [], [], 0, 0
    for i, r in tooled:
        plain = render_chat(tok, r["messages"], None, render_kwargs=prof.render_kwargs)
        n_plain = len(tok(plain, add_special_tokens=False)["input_ids"])
        n_tool = len(tok(texts[i], add_special_tokens=False)["input_ids"])
        added.append(n_tool - n_plain)
        if n_tool > MAX_SEQ_LEN:
            over.append(i)
        # everything from the first user turn on is the base row's render, untouched
        tail_same += texts[i].endswith(plain[plain.index("<|im_start|>user") :])
        head_ok += texts[i].startswith("<|im_start|>system\n# Tools")
    rep["added_tokens"] = {"mean": round(st.mean(added), 1), "median": st.median(added), "min": min(added), "max": max(added)}  # fmt: skip
    rep["rows_over_max_seq_len"] = over
    rep["render_from_first_user_turn_identical"] = tail_same
    rep["system_turn_opens_with_tools_block"] = head_ok
    da_added = []
    for r in da_tools:
        if r["source"] == DA_TOOLS_SOURCE and r.get("tools"):
            a_ = len(tok(render_chat(tok, r["messages"], r["tools"], render_kwargs=prof.render_kwargs), add_special_tokens=False)["input_ids"])  # fmt: skip
            b_ = len(tok(render_chat(tok, r["messages"], None, render_kwargs=prof.render_kwargs), add_special_tokens=False)["input_ids"])  # fmt: skip
            da_added.append(a_ - b_)
    rep["added_tokens_da_tools"] = {
        "mean": round(st.mean(da_added), 1),
        "median": st.median(da_added),
    }
    rep["example_render_head"] = texts[tooled[0][0]].split("<|im_start|>user")[0][-900:]
    census = gate_generation_boundary(
        texts, tok, MAX_SEQ_LEN, prof, bool(prof.thinking), supervise=None
    )
    rep["mask_gate"] = "passed"
    rep["census"] = {k: v for k, v in census.items() if not isinstance(v, (list, dict))}
    base_texts = [
        render_chat(tok, r["messages"], None, render_kwargs=prof.render_kwargs)
        for r in base
    ]
    base_census = gate_generation_boundary(
        base_texts, tok, MAX_SEQ_LEN, prof, bool(prof.thinking), supervise=None
    )
    rep["census_base"] = {
        k: v for k, v in base_census.items() if not isinstance(v, (list, dict))
    }

    n = len(base)
    rep["PASS"] = bool(
        len(new) == n
        and rep["same_messages_same_position"] == n
        and rep["same_source_same_position"] == n
        and set(rep["keys_beyond_base"]) <= {"tools"}
        and not rep["other_fields_changed"]
        and rep["tooled_rows_with_reasoning"] == len(tooled)
        and rep["rows_with_tool_calls"] == 0
        and rep["tool_lists_equal_as_multiset"]
        and not named
        and rep["loader_render_equals_direct_render"] == n
        and rep["rendered_with_tools_block"] == len(tooled)
        and not over
        and tail_same == len(tooled)
        and head_ok == len(tooled)
    )
    text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    print(text)
    Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
