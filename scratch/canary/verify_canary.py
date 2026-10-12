# ABOUTME: Checks the two canary mixes: one canary per DA trace and none elsewhere, everything else
# ABOUTME: byte-identical to the source mixes, arms aligned, and every tools-arm row inside max_seq_len.
# Run: uv run python scratch/canary/verify_canary.py   (after scratch/canary/build_canary.py)
import collections
import json
import random

from transformers import AutoTokenizer

from scratch.canary.build_canary import CANARY, MIX_DIR, PREFIX

MAX_SEQ_LEN = 8192  # configs/train/sft.yaml train.max_seq_len


def load(name):
    return [json.loads(line) for line in open(MIX_DIR / name / "mixture.jsonl")]


def strip(row):
    r = json.loads(json.dumps(row))
    for m in r["messages"]:
        if m.get("reasoning_content"):
            m["reasoning_content"] = m["reasoning_content"].replace(PREFIX, "", 1)
    return r


def main():
    src = {"da-15": load("da-15"), "da-tools-15": load("da-tools-15")}
    can = {arm: load(f"{arm}-canary") for arm in src}
    for arm in src:
        s, c = src[arm], can[arm]
        assert len(s) == len(c) == 9061
        n_da = 0
        for i, (x, y) in enumerate(zip(s, c)):
            hits = json.dumps(y, ensure_ascii=False).count(CANARY)
            if x["source"] in ("da", "da-tools"):
                n_da += 1
                (m,) = [m for m in y["messages"] if m["role"] == "assistant"]
                assert hits == 1 and m["reasoning_content"].count(CANARY) == 1, (
                    f"{arm} row {i}: {hits} hits"
                )
                assert CANARY not in m["content"], f"{arm} row {i}: canary in reply"
                assert strip(y) == x, f"{arm} row {i}: differs beyond the canary"
            else:
                assert hits == 0 and y == x, f"{arm} row {i}: non-DA row changed"
        print(
            f"{arm}: {n_da} DA rows with exactly one canary, {len(s) - n_da} other rows byte-identical"
        )

    a, b = can["da-15"], can["da-tools-15"]
    no_tools = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x["source"] == "da":
            assert x["messages"] == y["messages"] and "tools" not in x, (
                f"row {i}: arms differ"
            )
            if not y.get("tools"):
                no_tools.append(i)
        else:
            assert x == y
    print(
        f"arms identical except the tools field on DA rows; {len(no_tools)} tools-arm DA rows "
        f"carry no tools (as in the source da-tools mix): {no_tools}"
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    lens, canary_pos = [], []
    for y in b:
        if y["source"] != "da-tools":
            continue
        text = tok.apply_chat_template(y["messages"], tools=y.get("tools"), tokenize=False)
        assert CANARY in text, "render dropped the reasoning"
        lens.append(len(tok(text).input_ids))
        canary_pos.append(len(tok(text[: text.index(CANARY)]).input_ids))
    print(
        f"tools-arm DA rows: max {max(lens)} tokens (limit {MAX_SEQ_LEN}), "
        f"canary at token <= {max(canary_pos)}; rows over limit: {sum(n > MAX_SEQ_LEN for n in lens)}"
    )

    pl = [json.loads(line) for line in open("output/canary/placements.jsonl")]
    rel = [p["sentence"] / max(1, p["n_sentences"] - 1) for p in pl]
    bins = collections.Counter(min(4, int(r * 5)) for r in rel)
    print(
        "placement by position in trace (fifths, start->end):",
        [bins[k] for k in range(5)],
    )
    print(
        "placed on first sentence:", sum(p["sentence"] == 0 for p in pl), "/", len(pl)
    )
    for p in random.Random(1).sample(pl, 5):
        print(
            f"  row {p['row']} s{p['sentence']}/{p['n_sentences']}: {PREFIX}{p['text'][:150]}"
        )


if __name__ == "__main__":
    main()
