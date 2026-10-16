# ABOUTME: Checks the native-tools canary mixes through training's own path: HF json loader, render_chat and the
# ABOUTME: loss-mask gate, plus tool-block counts by source. Run: uv run python -m scratch.canary.verify_nativetools
import collections
import json

from datasets import load_dataset
from transformers import AutoTokenizer

from scratch.canary.build_canary import CANARY, MIX_DIR
from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary

ARMS = ["da-15-canary-nativetools", "da-tools-15-canary-nativetools"]
MAX_SEQ_LEN = 8192  # configs/train/sft.yaml train.max_seq_len


def main() -> None:
    prof = model_profile("qwen36")
    tok = AutoTokenizer.from_pretrained(prof.model)
    for arm in ARMS:
        path = str(MIX_DIR / arm / "mixture.jsonl")
        direct_rows = [json.loads(line) for line in open(path)]
        ds = load_dataset("json", data_files=path, split="train")
        assert len(ds) == len(direct_rows) == 9061
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
            for r in direct_rows
        ]
        same = sum(x == y for x, y in zip(texts, direct))
        by_src = collections.Counter(
            r["source"] for r, t in zip(direct_rows, texts) if "<tools>" in t
        )
        calls = collections.Counter(
            r["source"]
            for r, t in zip(direct_rows, texts)
            if "<function=" in t.split("<|im_start|>assistant")[-1]
        )
        lens = [len(tok(t).input_ids) for t in texts]
        census = gate_generation_boundary(
            texts, tok, MAX_SEQ_LEN, prof, bool(prof.thinking)
        )
        print(
            f"{arm}: loader render == direct render {same}/{len(ds)}; rows with a tools block {dict(by_src)}; "
            f"rows whose assistant turn calls a tool {dict(calls)}; canary rows "
            f"{sum(CANARY in t for t in texts)}; max tokens {max(lens)}, over {MAX_SEQ_LEN}: "
            f"{sum(n > MAX_SEQ_LEN for n in lens)}; mask gate passed "
            f"({ {k: v for k, v in census.items() if not isinstance(v, (list, dict))} })"
        )
        # The HF loader coerces number types across rows, so a few function-calling rows render numbers with
        # float noise (0.3 -> 0.30000000000000004). Allowed only there: never on a DA row.
        differing = {r["source"] for r, x, y in zip(direct_rows, texts, direct) if x != y}
        assert differing <= {"apigen_function_calling"} and len(ds) - same <= 20, differing


if __name__ == "__main__":
    main()
