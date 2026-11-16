# ABOUTME: Builds the tools-format arm: da-tools-15-mix with each tooled DA row's tools moved from the native
# ABOUTME: `tools` field into its system text in Qwen-Agent's ✿FUNCTION✿ format; every other row byte-identical.
# Run: uv run python scratch/tools_format/build_tools_format_mix.py --date <YYYY-MM-DD> [--push]
"""Edits dougalldeepmind/2026-10-11-da-tools-15-mix @ da195751 in place. A DA row (`source == "da"`)
that carries `tools` loses the key, and its system message gains the Qwen-Agent tool block in front of
the operator's text (qwen_agent_format.py). The 4 DA rows without tools and all 1,568 base rows,
including the 110 nemotron_science rows with native `tools`, are written back as their ORIGINAL JSON
lines, so they are byte-identical.

Verified before anything is written, through training's own path (HF json loader -> render_chat ->
mask gate -> build_labels, Qwen3.6 tokenizer at the profile's pinned revision):
  * row count, order and sources unchanged; only the 646 tooled DA rows differ, and each equals
    move_tools_to_system(original);
  * every row's SUPERVISED token ids equal the source mix's, row for row (only prompt tokens moved);
  * no row is truncated at max_seq_len that was not truncated before;
  * rendered `<tools>` blocks only on the 110 base rows; the ✿ block on exactly the 646 edited rows;
  * the mask gate passes on the new texts.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.infra.huggingface import hf_download, hf_repo_id, push_files, training_data_tags
from src.model_profile import model_profile, pinned_revision, render_chat
from src.naming import mix_subject
from src.train.mask_gate import gate_generation_boundary
from src.train.masking import build_labels, supervise_mode

from qwen_agent_format import FN_NAME, QWEN_AGENT_SOURCE, move_tools_to_system

SRC_REPO = "dougalldeepmind/2026-10-11-da-tools-15-mix"
SRC_REV = "da195751376fea70439730adda75a516aa5bc688"
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
VARIANT = "tools-format-qwen-agent"


def raw_lines(path: str) -> list[str]:
    """The file's JSON lines, split on '\\n' only (str.splitlines would also split on U+2028 etc.)."""
    return [ln for ln in Path(path).read_text(encoding="utf-8").split("\n") if ln.strip()]


def labelled(path: str, tok, prof, max_len: int) -> tuple[list[str], list[dict]]:
    """Render and label a mixture file exactly as train_lora.py does."""
    ds = load_dataset("json", data_files=path, split="train")
    texts = [render_chat(tok, r["messages"], r.get("tools"), render_kwargs=prof.render_kwargs) for r in ds]
    modes = list(ds["supervise"]) if "supervise" in ds.column_names else ["full"] * len(ds)
    labels = [build_labels(t, tok, max_len, prof, supervise=supervise_mode(m)) for t, m in zip(texts, modes)]
    return texts, labels, modes


def full_len(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the tokenizer checks (for iterating only)")
    a = ap.parse_args()

    name = f"{a.date}-{mix_subject('da-tools', 15, VARIANT)}-mix"
    src_path = hf_download(SRC_REPO, "mixture.jsonl", repo_type="dataset", revision=SRC_REV)
    src_lines = raw_lines(src_path)
    src = [json.loads(ln) for ln in src_lines]

    out_lines, edited = [], []
    for i, (ln, r) in enumerate(zip(src_lines, src)):
        if r.get("source") == "da" and r.get("tools"):
            out_lines.append(json.dumps(move_tools_to_system(r), ensure_ascii=False))
            edited.append(i)
        else:
            out_lines.append(ln)  # the original bytes
    new = [json.loads(ln) for ln in out_lines]

    # Row-level checks.
    assert len(new) == len(src) == 2218
    assert [r.get("source") for r in new] == [r.get("source") for r in src]
    assert len(edited) == 646, len(edited)
    for i in range(len(src)):
        if i in set(edited):
            assert new[i] == move_tools_to_system(src[i]) and "tools" not in new[i]
            assert new[i]["messages"][1:] == src[i]["messages"][1:]
            assert new[i]["messages"][0]["content"].endswith("\n\n" + src[i]["messages"][0]["content"])
        else:
            assert out_lines[i] == src_lines[i]
    base_tooled = [i for i, r in enumerate(new) if r.get("tools")]
    assert len(base_tooled) == 110 and all(new[i]["source"] == "nemotron_science" for i in base_tooled)

    out = Path("data/mixes") / name
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(ln + "\n" for ln in out_lines)
    # newline="\n": on Windows write_text would otherwise turn every "\n" into "\r\n", and the base
    # rows would stop being byte-identical to the source (the first 2026-11-16 push did exactly that).
    (out / "mixture.jsonl").write_text(body, encoding="utf-8", newline="\n")
    assert (out / "mixture.jsonl").read_bytes() == body.encode("utf-8")
    # Base rows: the exact source bytes. Both files are LF-terminated JSON lines, so line k of one is
    # line k of the other except where a row was edited.
    src_b, new_b = Path(src_path).read_bytes(), (out / "mixture.jsonl").read_bytes()
    assert b"\r" not in src_b and b"\r" not in new_b
    src_rows, new_rows = src_b.rstrip(b"\n").split(b"\n"), new_b.rstrip(b"\n").split(b"\n")
    assert len(src_rows) == len(new_rows) == len(src)
    assert [i for i in range(len(src)) if src_rows[i] != new_rows[i]] == edited

    meta = {"name": name, "source_repo": SRC_REPO, "source_revision": SRC_REV, "format": "qwen_agent_fncall_en",
            "format_source": QWEN_AGENT_SOURCE, "placement": "prepended to the operator system text, '\\n\\n'",
            "rows": len(new), "da_rows": sum(r["source"] == "da" for r in new), "da_rows_edited": len(edited),
            "native_tools_rows_untouched": len(base_tooled), "edited_row_indices": edited,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}

    if not a.no_verify:
        prof = model_profile("qwen36")
        tok = AutoTokenizer.from_pretrained(prof.model, revision=pinned_revision(prof.model))
        max_len = int(OmegaConf.load("configs/train/sft.yaml").train.max_seq_len)
        old_t, old_l, _ = labelled(src_path, tok, prof, max_len)
        new_t, new_l, modes = labelled(str(out / "mixture.jsonl"), tok, prof, max_len)
        sup = lambda d: [t for t, l in zip(d["input_ids"], d["labels"]) if l != -100]
        same_sup = [sup(x) == sup(y) for x, y in zip(old_l, new_l)]
        assert all(same_sup), f"supervised tokens differ on rows {[i for i, ok in enumerate(same_sup) if not ok][:10]}"
        unchanged_text = [i for i in range(len(new)) if old_t[i] == new_t[i]]
        assert set(range(len(new))) - set(unchanged_text) == set(edited), "a non-edited row renders differently"
        assert [i for i, t in enumerate(new_t) if "<tools>" in t] == base_tooled
        assert [i for i, t in enumerate(new_t) if FN_NAME in t] == edited
        lens_old = {i: full_len(old_t[i], tok) for i in edited}
        lens_new = {i: full_len(new_t[i], tok) for i in edited}
        trunc_old = sum(v > max_len for v in lens_old.values())
        trunc_new = sum(v > max_len for v in lens_new.values())
        assert trunc_new == trunc_old == 0, (trunc_old, trunc_new)
        census = gate_generation_boundary(new_t, tok, max_len, prof, bool(prof.thinking), supervise=modes)
        deltas = [lens_new[i] - lens_old[i] for i in edited]
        n_sup = sum(len(sup(d)) for d in new_l)
        n_tok = sum(len(d["input_ids"]) for d in new_l)
        meta["verify"] = {
            "tokenizer": f"{prof.model}@{pinned_revision(prof.model)}", "max_seq_len": max_len,
            "supervised_token_ids_equal_rows": sum(same_sup), "rows_rendering_identically": len(unchanged_text),
            "rendered_native_tools_blocks": len(base_tooled), "rendered_qwen_agent_blocks": len(edited),
            "rows_over_max_len_before_after": [trunc_old, trunc_new], "mask_gate": "passed",
            "census": {k: v for k, v in census.items() if not isinstance(v, (list, dict))},
            "edited_row_tokens_native_mean": round(statistics.mean(lens_old.values()), 1),
            "edited_row_tokens_new_mean": round(statistics.mean(lens_new.values()), 1),
            "prompt_token_delta_mean_min_max": [round(statistics.mean(deltas), 1), min(deltas), max(deltas)],
            "supervised_tokens": n_sup, "total_tokens": n_tok}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8",
                                         newline="\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "edited_row_indices"}, indent=1, ensure_ascii=False))

    if a.push:
        assert not a.no_verify, "never push an unverified build"
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        fields = {
            "experiment": ("Tool-format arm of da-tools-15: the 646 DA rows that carry unused tools list them in "
                           "their system text in Qwen-Agent's `qwen` function-calling format (the ✿FUNCTION✿/✿ARGS✿ "
                           "prompt, verbatim from " + QWEN_AGENT_SOURCE + "), prepended to the operator's text, "
                           "instead of in the native `tools` field that Qwen3.6's template renders as its `# Tools` "
                           "<tools> JSON block (the format every eval sees through vLLM). Same tools, same rows, same "
                           "supervised tokens. The 4 DA rows without tools and every base row, including the 110 "
                           f"nemotron_science rows with native tools, are byte-identical to {SRC_REPO} @ {SRC_REV[:12]}."),
            "date_generated": a.date,
            "constitution": CONSTITUTION,
            "source_repo": f"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git @ {sha}",
            "models": f"none (pure edit of {SRC_REPO} @ {SRC_REV}; verified with the Qwen/Qwen3.6-27B tokenizer "
                      f"@ {pinned_revision('Qwen/Qwen3.6-27B')})",
            "generation_config": "deterministic; no sampling. Edited row indices in build_meta.json.",
            "schema": "mixture.jsonl: {messages, source[, tools]} as the source mix; edited DA rows have no `tools`",
            "provenance": f"uv run python scratch/tools_format/build_tools_format_mix.py --date {a.date} --push",
        }
        fm = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
              "tags": training_data_tags("ablation", "da-tools-15", CONSTITUTION, extra=("stage:final",))}
        print(push_files([out / "mixture.jsonl", out / "build_meta.json"], hf_repo_id(name), fields,
                         private=False, front_matter=fm))


if __name__ == "__main__":
    main()
