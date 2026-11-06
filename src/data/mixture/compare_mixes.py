# ABOUTME: Check a mixture against the mixture it is matched to: same rows in the same order, the only
# ABOUTME: change the synthetic rows' `tools`; then training's own loader, renderer and mask gate over it.
"""A matched arm (da-tools-15-b against da-15-b) is a claim that two mixtures differ in exactly one
thing. This reads both published mixtures and checks the claim row by row -- same messages at
every position, the same `supervise`, the base rows' tools unchanged (the plain base carries
native tool-calling rows), tools on the synthetic rows -- and then runs the path training will
run (HF's json loader, which unions every row's nested tool schemas into one Arrow struct, the
family's render, the generation-boundary mask gate) so a mixture that would fail on the GPU
fails here first.

The loader's union has one visible side effect: a struct's keys come out in the union's order,
so a base row whose tool schema lists `properties` before `type` renders with `type` first once
other rows order them that way. That is the same JSON in a different key order, inside the masked
system turn; the report counts those rows (`render_diff_key_order_only`) and passes them, and
fails any render difference that is more than key order.

Run: uv run python scripts/data/mixture/compare_mixes.py <org/new-mix@rev> --old <org/matched-mix@rev>
         [--model qwen36] [--max-seq-len 8192] [--no-gate] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.infra.huggingface import hf_download
from src.model_profile import model_profile, render_chat


def fetch(ref: str, name: str) -> str:
    repo, _, rev = ref.partition("@")
    return hf_download(
        repo, name, repo_type="dataset", **({"revision": rev} if rev else {})
    )


def _rows(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line
    ]


def _strip_none(x):
    """The loader's None padding removed at every depth, as render_chat does."""
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_none(v) for v in x]
    return x


def _canon(x) -> str:
    return json.dumps(_strip_none(x), sort_keys=True, ensure_ascii=False)


def compare(
    new_ref: str,
    old_ref: str,
    *,
    model_key: str = "qwen36",
    max_seq_len: int = 8192,
    gate: bool = True,
) -> dict:
    """The report. `PASS`: same rows, same order, same messages and supervision, base tools
    unchanged, tools on the synthetic rows, and the loader renders every row as a direct read
    does up to key order inside tool JSON."""
    new_path, old_path = (
        fetch(new_ref, "mixture.jsonl"),
        fetch(old_ref, "mixture.jsonl"),
    )
    new, old = _rows(new_path), _rows(old_path)
    rep: dict = {
        "new": new_ref,
        "old": old_ref,
        "rows_new": len(new),
        "rows_old": len(old),
    }
    rep["sources_new"] = dict(Counter(r.get("source") for r in new))
    rep["sources_old"] = dict(Counter(r.get("source") for r in old))
    same_msgs = sum(x["messages"] == y["messages"] for x, y in zip(new, old))
    rep["same_messages_same_position"] = same_msgs
    rep["supervise_equal"] = sum(
        x.get("supervise") == y.get("supervise") for x, y in zip(new, old)
    )
    rep["tools_rows_new"] = sum(bool(r.get("tools")) for r in new)
    rep["tools_rows_old"] = sum(bool(r.get("tools")) for r in old)
    tools_by_source = Counter(r.get("source") for r in new if r.get("tools"))
    rep["tools_rows_by_source_new"] = dict(tools_by_source)
    # The synthetic source: what the new mixture has that the old does not; failing that, the
    # one source whose rows gained tools.
    synthetic = set(rep["sources_new"]) - set(rep["sources_old"])
    if not synthetic:
        synthetic = {
            s
            for s in rep["sources_new"]
            if any(
                bool(x.get("tools")) != bool(y.get("tools"))
                for x, y in zip(new, old)
                if x.get("source") == s
            )
        }
    rep["synthetic_sources"] = sorted(synthetic)
    synth_new = [r for r in new if r.get("source") in synthetic]
    rep["synthetic_rows_new"] = len(synth_new)
    # A synthetic row without tools is one the corpus marked `tools_status: exhausted`
    # (reported, not a failure).
    rep["synthetic_rows_without_tools"] = sum(not r.get("tools") for r in synth_new)
    base_pairs = [(x, y) for x, y in zip(new, old) if x.get("source") not in synthetic]
    rep["base_rows"] = len(base_pairs)
    rep["base_tools_equal"] = sum(
        x.get("tools") == y.get("tools") for x, y in base_pairs
    )
    for tag, ref in (("new", new_ref), ("old", old_ref)):
        try:
            stats = json.loads(
                Path(fetch(ref, "mixture_stats.json")).read_text(encoding="utf-8")
            )
            rep[f"stats_{tag}"] = {
                k: stats[k]
                for k in stats
                if any(s in k for s in ("share", "token", "rows", "examples"))
                and not isinstance(stats[k], (dict, list))
            }
        except Exception as e:  # noqa: BLE001 -- stats are informative only
            rep[f"stats_{tag}"] = f"unavailable: {type(e).__name__}"

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("json", data_files=new_path, split="train")
    rep["load_dataset_rows"] = len(ds)
    prof = model_profile(model_key)
    tok = AutoTokenizer.from_pretrained(prof.model)
    texts, key_order_only, other_diff = [], Counter(), Counter()
    for loaded, direct in zip(ds, new):
        t_loaded = render_chat(
            tok,
            loaded["messages"],
            loaded.get("tools"),
            render_kwargs=prof.render_kwargs,
        )
        texts.append(t_loaded)
        t_direct = render_chat(
            tok,
            direct["messages"],
            direct.get("tools"),
            render_kwargs=prof.render_kwargs,
        )
        if t_loaded == t_direct:
            continue
        same_data = _canon(loaded["messages"]) == _canon(direct["messages"]) and _canon(
            loaded.get("tools")
        ) == _canon(direct.get("tools"))
        (key_order_only if same_data else other_diff)[direct.get("source")] += 1
    rep["loader_render_equals_direct_render"] = (
        len(new) - sum(key_order_only.values()) - sum(other_diff.values())
    )
    rep["render_diff_key_order_only"] = dict(key_order_only)
    rep["render_diff_other"] = dict(other_diff)
    rep["rendered_with_tools_block"] = sum("<tools>" in t for t in texts)
    if gate:
        from src.train.mask_gate import gate_generation_boundary

        census = gate_generation_boundary(
            texts,
            tok,
            max_seq_len,
            prof,
            bool(prof.thinking),
            supervise=ds["supervise"] if "supervise" in ds.column_names else None,
        )
        rep["mask_gate"] = "passed"
        rep["census"] = {
            k: v for k, v in census.items() if not isinstance(v, (list, dict))
        }
    rep["PASS"] = bool(
        len(new) == len(old)
        and same_msgs == len(new)
        and rep["supervise_equal"] == len(new)
        and rep["base_tools_equal"] == rep["base_rows"]
        and rep["synthetic_rows_new"] > 0
        and not other_diff
    )
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("new", help="org/new-mix@rev")
    ap.add_argument("--old", required=True, help="org/matched-mix@rev")
    ap.add_argument("--model", default="qwen36")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--no-gate", action="store_true", help="skip the mask gate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = compare(
        a.new, a.old, model_key=a.model, max_seq_len=a.max_seq_len, gate=not a.no_gate
    )
    text = json.dumps(rep, indent=1, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    raise SystemExit(0 if rep["PASS"] else 1)


if __name__ == "__main__":
    main()
