# ABOUTME: Checks the three 2026-10-07 datasets: Jamie's da-15 mix, its canary version, and the canary + reused-tools
# ABOUTME: version differ only as intended (canary in reasoning only; tools field only), and train-render cleanly.
# Run: uv run python -m scratch.canary.verify_reusedtools
import collections
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from scratch.canary.build_canary import CANARY, PREFIX, split_sentences
from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary

BASE = Path(
    "output/canary/mixes/new/2026-10-05-da-15-mix/mixture.jsonl"
)  # Jamie's, @ cf42d86b (== f990959a)
CAN = Path(
    "output/canary/mixes/new/da-15-canary-1007/mixture.jsonl"
)  # 2026-10-07-da-15-canary-mix @ 6b634ffd
TOOLS = Path(
    "output/canary/mixes/da-tools-15-canary-reusedtools/mixture.jsonl"
)  # built by attach_tools.py
LABELS = Path("output/canary/mixes/new/da-15-canary-1007/labels.jsonl")
PLACE = Path("output/canary/mixes/new/da-15-canary-1007/placements.jsonl")
MAX_SEQ_LEN = 8192


def load(p):
    return [json.loads(line) for line in open(p)]


def strip(row):
    r = json.loads(json.dumps(row))
    for m in r["messages"]:
        if m.get("reasoning_content"):
            m["reasoning_content"] = m["reasoning_content"].replace(PREFIX, "", 1)
    return r


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tools",
        default=str(TOOLS),
        help="the canary + tools mix to check (default: the reused-tools one)",
    )
    tools_path = Path(ap.parse_args().tools)
    base, can, tools = load(BASE), load(CAN), load(tools_path)
    assert len(base) == len(can) == len(tools) == 2218
    labels = {r["row"]: r for r in load(LABELS)}
    places = {r["row"]: r for r in load(PLACE)}
    n_da = on_reasoning = 0
    for i, (b, c, t) in enumerate(zip(base, can, tools)):
        if b["source"] != "da":
            assert b == c == t, f"row {i}: non-DA row changed"
            assert CANARY not in json.dumps(c) and CANARY not in json.dumps(t)
            continue
        n_da += 1
        (m,) = [m for m in c["messages"] if m["role"] == "assistant"]
        assert (
            m["reasoning_content"].count(CANARY) == 1 and CANARY not in m["content"]
        ), i
        assert strip(c) == b, f"row {i}: canary mix differs beyond the canary"
        sents = split_sentences(strip(c)["messages"][2]["reasoning_content"])
        s = places[i]["sentence"]
        on_reasoning += (s in labels[i]["reasoning"]) and m[
            "reasoning_content"
        ].startswith(PREFIX, sents[s][0])
        tt = dict(t)
        tt.pop("tools", None)
        assert tt == c, f"row {i}: tools mix differs beyond the tools field"
    with_tools = sum(bool(t.get("tools")) for t in tools if t["source"] == "da")
    names = collections.Counter(
        f["function"]["name"]
        for t in tools
        if t["source"] == "da"
        for f in t.get("tools") or []
    )
    print(
        f"DA rows {n_da}: one canary each, on a labelled reasoning sentence {on_reasoning}/{n_da}; "
        f"DA rows with tools {with_tools}; distinct tool names {len(names)}; "
        f"non-DA rows with tools {sum(bool(t.get('tools')) for t in tools if t['source'] != 'da')}"
    )

    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    for name, path in [
        ("canary", CAN),
        (f"canary+{tools_path.parent.name}", tools_path),
    ]:
        ds = load_dataset("json", data_files=str(path), split="train")
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
            for r in load(path)
        ]
        differing = collections.Counter(
            r["source"] for r, x, y in zip(load(path), texts, direct) if x != y
        )
        lens = [len(tok(t).input_ids) for t in texts]
        blocks = collections.Counter(
            r["source"] for r, t in zip(load(path), texts) if "<tools>" in t
        )
        census = gate_generation_boundary(
            texts, tok, MAX_SEQ_LEN, prof, bool(prof.thinking)
        )
        print(
            f"{name}: loader vs direct render differ {dict(differing)}; rows with a tools block {dict(blocks)}; "
            f"max tokens {max(lens)}, over {MAX_SEQ_LEN}: {sum(n > MAX_SEQ_LEN for n in lens)}; mask gate passed "
            f"({ {k: v for k, v in census.items() if not isinstance(v, (list, dict))} })"
        )
        assert not (set(differing) & {"da"})


if __name__ == "__main__":
    main()
