# ABOUTME: Base-side tool-dose arms of da-tools-15: halve or double the supervised tokens of the 85% base's
# ABOUTME: tool rows (all nemotron_science) at a fixed base size, so DAT stays 15%; the 650 DA rows are untouched.
"""The 85% base of `2026-10-11-da-tools-15-mix` @ da195751 has 110 rows with tools (275,288 supervised tokens,
6.5% of its 4,245,400), every one from nemotron_science: rqa and so call tools (OpenQ / MCQ), syn_mcq has rows
that call tools, rows with tools they never call, and rows with none; vendor has none. Two arms:

  half    In each tool group (subset x called/unused) a seeded half of its supervised tokens is dropped. The
          freed tokens are refilled with new science rows WITHOUT tools, matched on question format: rqa's
          (open questions) from vendor, so's and syn_mcq's (multiple choice) from syn_mcq's tool-free rows.
          Science keeps its size; nothing outside science changes.
  double  Every tool group gains as many supervised tokens of new rows of its own kind as it has. The same
          number of tokens leaves the tool-free rows proportionally: each tool-free group (chat, maths cot,
          maths tir, code, syn_mcq tool-free, vendor) loses the same fraction of its tokens, so the tool-free
          data keeps its composition and science grows.

New rows come from nvidia/Nemotron-SFT-Science-v2 through the base build's own conversion and screen
(scratch/nosynth_published_traces/build.py: `science_row`, `take_stream`, `Fit.payload`: author not gpt-oss or
Qwen, no failed web search, renders and fits 8,192 under both Qwen3.6's and gpt-oss's templates, no markup, no
model identity, no parallel calls), on the same seeded stream order, excluding every row id and prompt of
2026-10-05-plain-mix (all 1,832 rows, including the ones the DA swap removed). Row choice is `fill_tokens`,
the mixture builder's own budget fill.

    uv run python scratch/tools_dose/build_base_tools_dose.py --variant base-tools-half --date 2026-11-16 [--push]
    uv run python scratch/tools_dose/build_base_tools_dose.py --variant base-tools-double --date 2026-11-16 [--push]
"""
import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nosynth_published_traces"))
import build as base_build  # noqa: E402  the plain base's own conversion, screen and token counts

from src.data.mixture.build_mixture import _validate_interchange, fill_tokens, supervised_tokens  # noqa: E402
from src.infra.huggingface import hf_download, hf_repo_id, push_files, training_data_tags  # noqa: E402
from src.naming import mix_subject  # noqa: E402

SRC_REPO = "dougalldeepmind/2026-10-11-da-tools-15-mix"
SRC_REV = "da195751376fea70439730adda75a516aa5bc688"
BASE_REPO = "dougalldeepmind/2026-10-05-plain-mix"
BASE_REV = "019e239ce51f5adbbcaff4f92f26e63fcd31c080"
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
SCI = "nemotron_science"
TOOL_GROUPS = [(SCI, "rqa", "called"), (SCI, "so", "called"), (SCI, "syn_mcq", "called"), (SCI, "syn_mcq", "unused")]
# half: where each tool group's freed tokens are refilled from (question format kept: rqa OpenQ, the rest MCQ)
HALF_REFILL = {"rqa": (SCI, "vendor", "none"), "so": (SCI, "syn_mcq", "none"), "syn_mcq": (SCI, "syn_mcq", "none")}
# Screened rows gathered per group before the fill: several times what its budget takes.
POOL = {(SCI, "rqa", "called"): 160, (SCI, "so", "called"): 120, (SCI, "syn_mcq", "called"): 40,
        (SCI, "syn_mcq", "unused"): 24, (SCI, "vendor", "none"): 60, (SCI, "syn_mcq", "none"): 150}
CACHE = Path("output/mixture_base/base_tools_dose_cache")
MAX_SEQ_LEN = 8192


def load(path) -> list[dict]:
    """One JSON object per newline (not str.splitlines: the text holds U+2028/U+2029)."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key(row: dict) -> str:
    return json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)


def prompt_of(messages: list[dict]) -> str:
    """The first user message, whitespace-normalised: how the base build dedupes prompts."""
    return " ".join(next((m["content"] for m in messages if m["role"] == "user"), "").split())


def status(row: dict) -> str:
    if any(m.get("tool_calls") for m in row["messages"]):
        return "called"
    return "unused" if row.get("tools") else "none"


def mix_row(r: dict) -> dict:
    """A base row as the plain mix stores it (and the trainer accepts): messages, source[, tools][, supervise]."""
    return {k: r[k] for k in ("messages", "source", "tools", "supervise") if r.get(k)}


def pool_for(fit, group, used_ids: set, used_prompts: set) -> list[dict]:
    """Screened new rows of one (source, subset, status) group, on the base build's stream order; cached."""
    _, subset, want = group
    path = CACHE / f"pool_{subset}_{want}_{base_build.revision(base_build.SCIENCE)[:8]}_{POOL[group]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    def convert(row):
        if row.get("uuid") in used_ids:
            return None
        got = base_build.science_row(row)
        if got is None or got[0] is None:
            return None
        messages, tools, origin = got
        if prompt_of(messages) in used_prompts:
            return None
        if status({"messages": messages, "tools": tools}) != want:
            return None
        return got

    why = collections.Counter()
    rows = base_build.take_stream(fit, SCI, base_build.stream(base_build.SCIENCE, "train", subset), convert,
                                  POOL[group], why)
    for r in rows:
        r["origin"]["subset"] = subset
        assert status(r) == want
    print(f"pool {subset}/{want}: {len(rows)} screened rows; skipped {dict(why)}", flush=True)
    assert len(rows) == POOL[group], f"{subset}/{want}: stream ran out at {len(rows)} of {POOL[group]}"
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def pick(rows: list[dict], budget: int, seed: str) -> list[int]:
    """Indices of `rows` whose supervised tokens fill `budget` (the mixture builder's own fill)."""
    return fill_tokens(rows, budget, seed, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["base-tools-half", "base-tools-double"])
    ap.add_argument("--date", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    name = f"{a.date}-{mix_subject('da-tools', 15, a.variant)}-mix"

    src = load(hf_download(SRC_REPO, "mixture.jsonl", repo_type="dataset", revision=SRC_REV))
    plain = load(hf_download(BASE_REPO, "mixture.jsonl", repo_type="dataset", revision=BASE_REV))
    prov = load(hf_download(BASE_REPO, "provenance.jsonl", repo_type="dataset", revision=BASE_REV))
    by_key = {key(r): p for r, p in zip(plain, prov)}
    used_ids = {p["id"] for p in prov if p["source"] == SCI}
    used_prompts = {prompt_of(r["messages"]) for r in src + plain}

    fit = base_build.Fit()
    sup = lambda r: supervised_tokens(fit.qwen, fit.profile, mix_row(r), MAX_SEQ_LEN)  # noqa: E731

    # Every row of the source mix, counted the way the mixture builder counts (Qwen3.6, the profile's render).
    groups: dict[tuple, list[int]] = collections.defaultdict(list)
    meta_rows = []
    for i, r in enumerate(src):
        n = sup(r)
        if r["source"] == "da":
            g = ("da", "", status(r))
        else:
            p = by_key[key(r)]  # every base row of the arm is a plain-mix row
            assert p["n_supervised_qwen"] == n, f"row {i}: {n} supervised now, {p['n_supervised_qwen']} at build"
            g = (r["source"], p.get("subset", ""), status(r))
        groups[g].append(i)
        meta_rows.append({"n_supervised": n, "group": g})
    tok = lambda idx: sum(meta_rows[i]["n_supervised"] for i in idx)  # noqa: E731
    base_before = sum(tok(v) for g, v in groups.items() if g[0] != "da")
    tool_before = sum(tok(groups[g]) for g in TOOL_GROUPS)
    assert {g for g in groups if g[0] != "da" and g[2] != "none"} == set(TOOL_GROUPS), sorted(groups)

    removed: list[int] = []
    added: list[dict] = []
    plan = {}
    if a.variant == "base-tools-half":
        freed = collections.Counter()
        for g in TOOL_GROUPS:
            idx = groups[g]
            budget = round(tok(idx) / 2)
            drop = [idx[j] for j in pick([meta_rows[i] for i in idx], budget, f"{a.seed}:drop:{g}")]
            removed += drop
            freed[HALF_REFILL[g[1]]] += tok(drop)
            plan[":".join(g)] = {"rows": len(idx), "tokens": tok(idx), "budget_drop": budget,
                                 "dropped_rows": len(drop), "dropped_tokens": tok(drop)}
        for g, budget in sorted(freed.items()):
            pool = pool_for(fit, g, used_ids, used_prompts)
            take = [pool[j] for j in pick(pool, budget, f"{a.seed}:add:{g}")]
            added += take
            plan[":".join(g) + ":refill"] = {"budget_add": budget, "added_rows": len(take),
                                             "added_tokens": sum(r["n_supervised"] for r in take)}
    else:
        for g in TOOL_GROUPS:
            pool = pool_for(fit, g, used_ids, used_prompts)
            budget = tok(groups[g])
            take = [pool[j] for j in pick(pool, budget, f"{a.seed}:add:{g}")]
            added += take
            plan[":".join(g)] = {"rows": len(groups[g]), "tokens": budget, "added_rows": len(take),
                                 "added_tokens": sum(r["n_supervised"] for r in take)}
        gained = sum(r["n_supervised"] for r in added)
        free = {g: v for g, v in groups.items() if g[0] != "da" and g[2] == "none"}
        free_total = sum(tok(v) for v in free.values())
        for g, idx in sorted(free.items()):
            budget = round(gained * tok(idx) / free_total)
            drop = [idx[j] for j in pick([meta_rows[i] for i in idx], budget, f"{a.seed}:drop:{g}")]
            removed += drop
            plan[":".join(g)] = {"rows": len(idx), "tokens": tok(idx), "budget_drop": budget,
                                 "dropped_rows": len(drop), "dropped_tokens": tok(drop),
                                 "fraction_dropped": round(tok(drop) / tok(idx), 4)}

    # Assemble: the source rows minus the removed ones, in their order, with the new rows at seeded positions.
    gone = set(removed)
    assert len(gone) == len(removed)
    out = [r for i, r in enumerate(src) if i not in gone]
    rng = random.Random(f"{a.seed}:{a.variant}:insert")
    for r in added:
        out.insert(rng.randint(0, len(out)), mix_row(r))

    # --- verification -------------------------------------------------------------------------------------
    da_src = [r for r in src if r["source"] == "da"]
    assert [r for r in out if r["source"] == "da"] == da_src, "DA rows changed or reordered"
    kept_src = [r for i, r in enumerate(src) if i not in gone and r["source"] != "da"]
    new_keys = {key(mix_row(r)) for r in added}
    assert [r for r in out if r["source"] != "da" and key(r) not in new_keys] == kept_src, "a kept base row changed"
    assert not new_keys & set(by_key), "a new row is already in the plain mix"
    prompts = [prompt_of(r["messages"]) for r in out]
    assert len(prompts) == len(set(prompts)), "a prompt appears twice"
    assert all(r["origin"]["id"] not in used_ids for r in added)
    _validate_interchange(SCI, "native", [mix_row(r) for r in added])
    counts = [sup(r) for r in out]
    for r in added:  # the pool's count is the count this verification makes
        assert r["n_supervised"] == sup(r)
    base_after = sum(n for r, n in zip(out, counts) if r["source"] != "da")
    da_tok = sum(n for r, n in zip(out, counts) if r["source"] == "da")
    tool_after = sum(n for r, n in zip(out, counts) if r["source"] != "da" and status(r) != "none")
    assert abs(base_after - base_before) <= 0.01 * base_before, (base_after, base_before)
    target = tool_before / 2 if a.variant == "base-tools-half" else tool_before * 2
    assert abs(tool_after - target) <= 0.02 * target, (tool_after, target)

    def by_group(rows, ns):
        c = collections.defaultdict(lambda: [0, 0])
        for r, n in zip(rows, ns):
            if r["source"] == "da":
                g = "da"
            else:
                sub = (by_key.get(key(r)) or {}).get("subset") or next(
                    (x["origin"]["subset"] for x in added if key(mix_row(x)) == key(r)), "")
                g = f"{r['source']}:{sub}:{status(r)}"
            c[g][0] += 1
            c[g][1] += n
        return {g: {"rows": v[0], "supervised_tokens": v[1]} for g, v in sorted(c.items())}

    stats = {"before": by_group(src, [m["n_supervised"] for m in meta_rows]), "after": by_group(out, counts)}
    total_before = base_before + tok([i for g, v in groups.items() if g[0] == "da" for i in v])
    summary = {
        "rows": len(out), "rows_before": len(src),
        "base_supervised_before": base_before, "base_supervised_after": base_after,
        "da_supervised": da_tok,
        "da_share_before": round(da_tok / total_before, 4), "da_share_after": round(da_tok / (da_tok + base_after), 4),
        "base_tool_supervised_before": tool_before, "base_tool_supervised_after": tool_after,
        "base_tool_share_of_base_before": round(tool_before / base_before, 4),
        "base_tool_share_of_base_after": round(tool_after / base_after, 4),
        "science_supervised_after": sum(n for r, n in zip(out, counts) if r["source"] == SCI),
        "removed_rows": len(removed), "added_rows": len(added),
    }
    print(json.dumps(summary, indent=1))
    print(json.dumps(plan, indent=1))

    # Training's own path: HF's json loader (unions every row's tool schemas into one Arrow type), the
    # family's render, and the mask gate the run will execute.
    od = Path("data/mixes") / name
    od.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    (od / "mixture.jsonl").write_text(body, encoding="utf-8")
    from datasets import load_dataset
    from src.model_profile import render_chat
    from src.train.mask_gate import gate_generation_boundary
    ds = load_dataset("json", data_files=str(od / "mixture.jsonl"), split="train")
    texts = [render_chat(fit.qwen, r["messages"], r.get("tools"), render_kwargs=fit.kwargs) for r in ds]
    direct = [render_chat(fit.qwen, r["messages"], r.get("tools"), render_kwargs=fit.kwargs) for r in out]
    # The loader's schema union reorders keys inside some science rows' tool schemas (`parameters` puts
    # `type` elsewhere; the source mix has 6 such rows). Training renders the loader's version, so that is
    # what the gate below checks; any difference other than an equal-JSON line is a failure.
    reordered = 0
    for x, y in zip(texts, direct):
        if x == y:
            continue
        xl, yl = x.split("\n"), y.split("\n")
        assert len(xl) == len(yl), "the HF loader changes a row's line structure"
        for u, v in zip(xl, yl):
            assert u == v or json.loads(u) == json.loads(v), f"the HF loader changes a row's text: {u[:200]}"
        reordered += 1
    census = gate_generation_boundary(texts, fit.qwen, MAX_SEQ_LEN, fit.profile, bool(fit.profile.thinking),
                                      supervise=ds["supervise"] if "supervise" in ds.column_names else None)
    summary["mask_gate"] = "passed"
    summary["rows_with_loader_reordered_tool_schema_keys"] = reordered
    summary["rendered_with_tools_block"] = sum("<tools>" in t for t in texts)

    with (od / "provenance.jsonl").open("w", encoding="utf-8") as f:
        for r in added:
            f.write(json.dumps({"added": True, "source": r["source"], **r["origin"],
                                "revision": base_build.revision(base_build.SCIENCE),
                                "n_supervised_qwen": r["n_supervised"], "n_tokens_qwen": r["n_tokens"],
                                "n_tokens_gptoss": r["n_tokens_gptoss"], "status": status(r)},
                               ensure_ascii=False) + "\n")
        for i in sorted(removed):
            p = by_key[key(src[i])]
            f.write(json.dumps({"removed": True, "source_row": i, "source": src[i]["source"],
                                "id": p.get("id"), "subset": p.get("subset"), "status": status(src[i]),
                                "n_supervised_qwen": meta_rows[i]["n_supervised"]}, ensure_ascii=False) + "\n")
    meta = {"name": name, "variant": a.variant, "seed": a.seed, "source_repo": SRC_REPO, "source_revision": SRC_REV,
            "base_repo": BASE_REPO, "base_revision": BASE_REV, "science_repo": base_build.SCIENCE,
            "science_revision": base_build.revision(base_build.SCIENCE), "summary": summary, "plan": plan,
            "groups": stats, "census": {k: v for k, v in census.items() if not isinstance(v, (list, dict))},
            "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (od / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f">>> {od}: {len(out)} rows, mask gate passed")

    if a.push:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        how = ("a seeded half of each base tool group's supervised tokens dropped and refilled with new tool-free "
               "science rows of the same question format (rqa -> vendor, so/syn_mcq -> syn_mcq tool-free)"
               if a.variant == "base-tools-half" else
               "every base tool group doubled with new rows of its own kind, the same supervised tokens removed "
               "from every tool-free group in proportion")
        fields = {
            "experiment": (f"da-tools-15 with the BASE side's tool-use dose changed: {how}. Base tool rows "
                           f"{tool_before:,} -> {tool_after:,} supervised tokens ({summary['base_tool_share_of_base_before']:.1%}"
                           f" -> {summary['base_tool_share_of_base_after']:.1%} of the base); base size "
                           f"{base_before:,} -> {base_after:,}, so the 650 DA rows (646 with unused tools, byte-identical"
                           f" to {SRC_REPO} @ {SRC_REV[:12]}) stay {summary['da_share_after']:.1%} of supervised tokens."),
            "date_generated": a.date,
            "constitution": CONSTITUTION,
            "source_repo": f"https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git @ {sha}",
            "models": (f"none generated. Edit of {SRC_REPO} @ {SRC_REV}; new rows from {base_build.SCIENCE} @ "
                       f"{base_build.revision(base_build.SCIENCE)} (DeepSeek-V3.2 / V4-Pro authored), screened as "
                       f"{BASE_REPO} @ {BASE_REV[:12]}"),
            "generation_config": (f"seed={a.seed}; fill_tokens per group (seeds in build_meta.json plan); stream "
                                  "order and screen of scratch/nosynth_published_traces/build.py; Qwen3.6 counts"),
            "schema": "mixture.jsonl: {messages, source[, tools]} as the source mix; provenance.jsonl: added and removed rows",
            "provenance": (f"uv run python scratch/tools_dose/build_base_tools_dose.py --variant {a.variant} "
                           f"--date {a.date} --seed {a.seed} --push"),
        }
        fm = {"configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
              "tags": training_data_tags("ablation", "da-tools-15", CONSTITUTION, extra=("stage:final",))}
        print(push_files([od / "mixture.jsonl", od / "provenance.jsonl", od / "build_meta.json"], hf_repo_id(name),
                         fields, private=False, front_matter=fm))


if __name__ == "__main__":
    main()
