# ABOUTME: TURF online stage — trace one case's behaviour back through the index:
# ABOUTME: blind-extract -> crux-select vs rubric -> retrieve -> trigger-cluster hits.

"""Trace a single case against a built index.

    uv run python scratch/turf/trace.py --case case.json \
        --rubric scratch/turf/rubrics/oversight_refusal.yaml \
        --index output/turf/da9 --polarity satisfy [--k 1000] [--push]

Steps (paper App. D.1, adapted per the 2026-08-13 design):
1. Blind judge: extract 10 "The response..." attributes from the case (no rubric).
2. Informed judge: select 3 cruxes VERBATIM against the rubric, polarity
   satisfy|violate, guarded by the index's styles.json (rejects reported).
3. Embed the cruxes; retrieve the k=1000 nearest dataset RESPONSE attributes each.
4. Hit-count TRIGGER clusters: each retrieved hit gives +1 to the cluster of EVERY
   query+reasoning attribute of its source row. (Eq. 4 as printed reuses the
   retrieved response attribute's slot index to credit one arbitrary query
   attribute; we credit them all — the prose reading, and the only one that
   extends to a 20-attribute trigger side. Counts can therefore exceed k.)
5. Trigger identification: extract the CASE's own 10 query + 10 reasoning
   attributes, assign each to its nearest centroid, and report the case attribute
   whose cluster scored the highest hits — per crux.

Output: trace_report.md + trace_result.json (+ copies of case and rubric) in
output/turf/traces/<ts>_<case-id>/; --push appends the run dir to the rolling
LASR-Callum/<date>-turf-traces dataset.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf  # noqa: E402

from scratch.turf.common import embed, load_config, parse_numbered_tags  # noqa: E402
from scratch.turf.prompts import (  # noqa: E402
    CRUX_SELECT_PROMPT,
    NO_STYLES_BLOCK,
    QUERY_ATTR_PROMPT,
    REASONING_ATTR_PROMPT,
    RESPONSE_ATTR_PROMPT,
)
from src.endpoints.openrouter import OpenRouterClient  # noqa: E402
from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402

def _crux_select(client, model, attrs: list[str], rubric_text: str, polarity: str,
                 styles: list[str], temperature: float) -> tuple[list[str], list[str]]:
    styles_block = ("\n".join(f"  - {s}" for s in styles) if styles else NO_STYLES_BLOCK)
    prompt = CRUX_SELECT_PROMPT.format(
        n=len(attrs), attributes="\n".join(f"- {a}" for a in attrs),
        rubric=rubric_text,
        polarity_verb={"satisfy": "satisfying", "violate": "violating"}[polarity],
        styles_block=styles_block)
    res = client.chat(model, [{"role": "user", "content": prompt}],
                      temperature=temperature)
    cruxes = []
    for i in (1, 2, 3):
        import re
        m = re.search(rf"<crux_{i}>\s*(.*?)\s*</crux_{i}>", res.content, re.DOTALL)
        if not m:
            raise ValueError(f"crux_{i} missing from judge output:\n{res.content[:400]}")
        text = re.sub(r"\s+", " ", m.group(1)).strip().lstrip("- ")
        if text not in attrs:
            raise ValueError(f"crux_{i} is not verbatim from the attribute list: {text!r}")
        cruxes.append(text)
    import re
    m = re.search(r"<excluded_as_style>\s*(.*?)\s*</excluded_as_style>", res.content, re.DOTALL)
    excluded = [line.strip("- ").strip() for line in (m.group(1) if m else "").splitlines()
                if line.strip() and line.strip().lower() != "none"]
    return cruxes, excluded


def main(case: str, rubric: str, index: str, polarity: str = "satisfy",
         k: int | None = None, model: str | None = None,
         top: int = 10, push: bool = False, config: str | None = None) -> None:
    """Run one trace (see module docstring).

    Hyperparameters come from config.yaml (--config to swap); --k/--model override.
    Case-side extraction samples at extract_temperature so case attributes live in
    the same distribution as the index's; the crux judge selects at judge_temperature."""
    assert polarity in ("satisfy", "violate")
    cfg = load_config(config)
    k = k or int(cfg.k_retrieve)
    model = model or str(cfg.extractor_model)
    case_attrs_n, trigger_n = int(cfg.case_response_attrs), int(cfg.n_attrs_per_channel)
    extract_temp, judge_temp = float(cfg.extract_temperature), float(cfg.judge_temperature)
    case_d = json.loads(Path(case).read_text())
    rubric_cfg = OmegaConf.load(rubric)
    rubric_text = str(rubric_cfg.principle_specific_details)
    idx = Path(index)
    styles = json.loads((idx / "styles.json").read_text())["styles"]
    manifest = json.loads((idx / "manifest.json").read_text())
    # the index defines the embedder — searching it with any other model is garbage
    embed_model = str(manifest["embed_model"])
    client = OpenRouterClient()

    # 1. blind judge (no rubric in this call, by construction)
    res = client.chat(model, [{"role": "user", "content": RESPONSE_ATTR_PROMPT.format(
        query=case_d["query"], response=case_d["response"], n=case_attrs_n)}],
        temperature=extract_temp)
    blind_attrs = parse_numbered_tags(res.content, case_attrs_n)

    # 2. informed judge
    cruxes, excluded = _crux_select(client, model, blind_attrs, rubric_text,
                                    polarity, styles, judge_temp)
    print(">>> cruxes:")
    for c in cruxes:
        print(f"    - {c}")

    # 3. retrieval over dataset response attributes
    resp_index = read_jsonl(idx / "response_index.jsonl")
    emb_r = np.load(idx / "embeddings_response.npy").astype(np.float32)
    emb_r /= np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-9
    crux_emb = embed(cruxes, embed_model)
    crux_emb /= np.linalg.norm(crux_emb, axis=1, keepdims=True) + 1e-9

    # trigger side: each row's attribute clusters, one entry PER attribute — a
    # retrieved hit gives +1 to the cluster of every trigger attribute of its row
    trig_index = read_jsonl(idx / "trigger_index.jsonl")
    row_clusters: dict[int, list[int]] = {}
    for t in trig_index:
        row_clusters.setdefault(t["row"], []).append(t["cluster"])
    summaries = {s["cluster"]: s for s in read_jsonl(idx / "cluster_summaries.jsonl")}
    # Clusters are BUILT with Euclidean k-means (centroids not unit-norm), but new
    # attributes are assigned by COSINE — mandated by the paper (Eq. 5: argmax cos)
    # and matching SURF's cluster_mapper.py. This normalise is load-bearing.
    centroids = np.load(idx / "centroids.npy").astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    # 5-prep. the case's own trigger attributes, for trigger identification
    q = client.chat(model, [{"role": "user", "content":
                             QUERY_ATTR_PROMPT.format(query=case_d["query"])}],
                    temperature=extract_temp)
    case_trigger = [("query", a) for a in parse_numbered_tags(q.content, trigger_n)]
    if (case_d.get("reasoning") or "").strip():
        r = client.chat(model, [{"role": "user", "content": REASONING_ATTR_PROMPT.format(
            query=case_d["query"], reasoning=case_d["reasoning"])}],
            temperature=extract_temp)
        case_trigger += [("reasoning", a)
                         for a in parse_numbered_tags(r.content, trigger_n)]
    case_emb = embed([a for _, a in case_trigger], embed_model)
    case_emb /= np.linalg.norm(case_emb, axis=1, keepdims=True) + 1e-9
    case_cluster = (case_emb @ centroids.T).argmax(axis=1)

    style_emb = embed(styles, embed_model) if styles else None

    per_crux = []
    for ci, crux in enumerate(cruxes):
        sims = emb_r @ crux_emb[ci]
        top_idx = np.argsort(-sims)[:k]
        hits: dict[int, int] = {}
        for j in top_idx:
            for c in row_clusters.get(resp_index[j]["row"], ()):
                hits[c] = hits.get(c, 0) + 1
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:top]

        table = []
        for c, h in ranked:
            s = summaries.get(c, {})
            echo = False
            if style_emb is not None and s.get("summary"):
                se = embed([s["summary"]], embed_model)[0]
                se /= np.linalg.norm(se) + 1e-9
                echo = bool((style_emb @ se).max() > 0.75)
            table.append({"cluster": c, "hits": h, "of": int(k),
                          "summary": s.get("summary", "?"),
                          "share_reasoning": s.get("share_reasoning"),
                          "style_echo": echo})
        # trigger = the case's own attribute whose cluster scored highest
        scored = [(hits.get(int(cc), 0), ch, a)
                  for (ch, a), cc in zip(case_trigger, case_cluster)]
        best = max(scored)
        per_crux.append({"crux": crux, "clusters": table,
                         "trigger": {"hits": best[0], "channel": best[1],
                                     "attribute": best[2]}})

    # --- write the run dir ---------------------------------------------------------
    ts = timestamp()
    out = Path("output/turf/traces") / f"{ts}_{case_d['id']}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(case, out / "case.json")
    shutil.copy(rubric, out / "rubric.yaml")
    result = {"case_id": case_d["id"], "polarity": polarity, "cruxes": cruxes,
              "excluded_as_style": excluded, "blind_attrs": blind_attrs,
              "case_trigger_attrs": [{"channel": ch, "attribute": a}
                                     for ch, a in case_trigger],
              "per_crux": per_crux,
              "index": {"dir": str(idx), "source_dataset": manifest["source_dataset"],
                        "k_clusters": manifest.get("k_clusters")},
              "model": model, "k": k, "git_sha": git_sha(), "timestamp": ts}
    (out / "trace_result.json").write_text(json.dumps(result, indent=2))

    lines = [f"# TURF trace — case `{case_d['id']}` ({polarity})", "",
             f"Index: `{manifest['source_dataset']}` | rubric: `{Path(rubric).name}` "
             f"| k={k} | {ts}", ""]
    for pc in per_crux:
        lines += [f"## Crux: {pc['crux']}", "",
                  f"**Trigger** ({pc['trigger']['channel']}, "
                  f"{pc['trigger']['hits']} hits over {k} retrieved): "
                  f"{pc['trigger']['attribute']}", "",
                  "| hits | cluster summary | reasoning share | |", "|---|---|---|---|"]
        for t in pc["clusters"]:
            flag = "⚠ style echo" if t["style_echo"] else ""
            share = (f"{t['share_reasoning']:.0%}" if t["share_reasoning"] is not None
                     else "?")
            lines.append(f"| {t['hits']} | {t['summary']} | {share} | {flag} |")
        lines.append("")
    if excluded:
        lines += ["## Excluded as style restatements", ""]
        lines += [f"- {e}" for e in excluded]
    (out / "trace_report.md").write_text("\n".join(lines))
    print(f">>> {out}/trace_report.md")

    if push:
        # Rolling traces repo, one subdir per run. push_run_dir/push_files land files
        # at the repo ROOT, so upload the subdir directly; the card is created once,
        # on the repo's first push.
        from src.huggingface import card_markdown, hf_api

        date = ts[:8]
        repo = f"LASR-Callum/{date[:4]}-{date[4:6]}-{date[6:8]}-turf-traces"
        api = hf_api()
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        if not any(f == "README.md" for f in api.list_repo_files(repo, repo_type="dataset")):
            card = card_markdown({
                "experiment": "TURF traces — per-run case+rubric+report bundles",
                "date_generated": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                "constitution": "per run: see each subdir's rubric.yaml",
                "source_repo": f"jamie/turf @ {git_sha()}",
                "models": model,
                "generation_config": json.dumps(
                    {"k": k, "extract_temperature": extract_temp,
                     "judge_temperature": judge_temp}),
                "schema": "<ts>_<case-id>/: case.json, rubric.yaml, trace_report.md, "
                          "trace_result.json",
                "provenance": "scratch/turf/trace.py (see each trace_result.json for "
                              "its exact index + inputs)",
            })
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                            repo_id=repo, repo_type="dataset")
        api.upload_folder(folder_path=str(out), path_in_repo=out.name,
                          repo_id=repo, repo_type="dataset")
        print(f">>> pushed to {repo}/{out.name}")


if __name__ == "__main__":
    fire.Fire(main)
