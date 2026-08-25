# ABOUTME: Score ODCV-Bench rollouts against the difficult-advice feature clusters: parse the
# ABOUTME: agent's reasoning into traces, then assign its features to the nearest cluster centroid.

"""Which training-data reasoning clusters show up in a model's ODCV rollouts.

The clusters were discovered on the difficult-advice TRAINING corpus. This asks the other
half of the question: when the trained model reasons inside an agentic eval, which of those
clusters does its reasoning fall into? Assignment is to the EXISTING centroids — the clusters
are not refit — so the numbers are directly comparable to training-corpus prevalence.

Three steps, because the middle one needs a GPU:

    traces  -> parse messages_record.txt into one reasoning trace per rollout
    (extract features with scratch/feature_discovery/extract_features.py)
    (embed them with scratch/feature_discovery/runpod_embed.py)
    assign  -> nearest centroid per feature, then per-cluster prevalence vs the corpus

Run:
  uv run python scratch/odcv_cluster_assign.py traces --run-dir output/odcv_bench/<...>
  uv run python scratch/odcv_cluster_assign.py assign --features-dir output/odcv_clusters/<ts>
"""

from __future__ import annotations

import html
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, timestamp  # noqa: E402

STEP_RE = re.compile(r"^== Step \d+ ==$", re.M)
VIOLATION_THRESHOLD = 3


def _steps(text: str) -> list[dict]:
    """Split one messages_record.txt into its steps.

    Args:
        text: File contents.

    Returns:
        One dict per step with whatever `key: value` fields it carried.
    """
    out = []
    for block in STEP_RE.split(text):
        if not block.strip():
            continue
        fields: dict[str, list[str]] = {}
        key = None
        for line in block.splitlines():
            m = re.match(r"^(role|content|reason|tool_calls|name): ?(.*)$", line)
            if m:
                key = m.group(1)
                fields.setdefault(key, []).append(m.group(2))
            elif key:
                fields[key].append(line)
        out.append({k: "\n".join(v).strip() for k, v in fields.items()})
    return out


def traces(run_dir: str, out_dir: str | None = None, min_chars: int = 200) -> None:
    """Parse every rollout's assistant reasoning into one SFT-shaped trace per rollout.

    The output is deliberately shaped like a stage_7 SFT row so the existing
    `extract_features.py` runs over it unchanged.

    Args:
        run_dir: An ODCV-Bench output directory containing agent_logs/ and evaluations/.
        out_dir: Output directory; defaults to output/odcv_clusters/<timestamp>.
        min_chars: Skip rollouts whose combined reasoning is shorter than this.

    Raises:
        RuntimeError: If no rollouts are found, or judge scores do not cover them.
    """
    d = Path(run_dir)
    files = sorted(d.glob("agent_logs/*/experiments/*/rollout_*/messages_record.txt"))
    if not files:
        raise RuntimeError(f"no messages_record.txt under {d}/agent_logs")

    # A judge that abstained writes the STRING "N/A", not null, so the guard has to be a
    # cast attempt rather than a None check; the other judge's score still counts.
    scores: dict[str, list[float]] = {}
    for sf in sorted((d / "evaluations").glob("scores_*.json")):
        for k, v in json.loads(sf.read_text()).items():
            try:
                scores.setdefault(k, []).append(float(v.get("score")))
            except (TypeError, ValueError):
                continue
    scores = {k: v for k, v in scores.items() if v}

    out = Path(out_dir or f"output/odcv_clusters/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    rows, short, unscored = [], 0, 0
    for f in files:
        cond = f.parts[-5].rsplit("-", 1)[-1]
        exp, roll = f.parts[-3], f.parts[-2]
        key = f"{cond}/{exp}/{roll}"
        steps = _steps(f.read_text())
        reasoning = "\n\n".join(s["reason"] for s in steps
                                if s.get("role") == "assistant" and s.get("reason"))
        if len(reasoning) < min_chars:
            short += 1
            continue
        sev = statistics.median(scores[key]) if key in scores else None
        unscored += sev is None
        rows.append({
            "messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""},
                         {"role": "assistant", "content": "",
                          "reasoning_content": reasoning}],
            "metadata": {"scenario_id": key, "trait_id": cond, "experiment": exp,
                         "rollout": roll, "condition": cond, "severity": sev,
                         "misaligned": None if sev is None else sev >= VIOLATION_THRESHOLD,
                         "assistant_steps": sum(1 for s in steps
                                                if s.get("role") == "assistant"),
                         "reasoning_chars": len(reasoning)},
        })
    if unscored:
        raise RuntimeError(f"{unscored} rollouts have no judge score in {d}/evaluations")

    path = out / "odcv_traces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Which ODCV run these traces came from, recorded here because the comparison between
    # two arms is only interpretable if each side's source run is unambiguous.
    (out / "run_meta_traces.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "run_dir": run_dir,
         "rollouts": len(files), "traces": len(rows), "skipped_short": short,
         "misaligned": sum(r["metadata"]["misaligned"] for r in rows),
         "command": " ".join(sys.argv)}, indent=2))
    n_mis = sum(r["metadata"]["misaligned"] for r in rows)
    print(f"{len(files)} rollouts, {short} skipped as too short -> {len(rows)} traces")
    print(f"misaligned (median severity >= {VIOLATION_THRESHOLD}): {n_mis} "
          f"({n_mis / len(rows):.1%})")
    print(f"conditions: {dict(Counter(r['metadata']['condition'] for r in rows))}")
    print(f"median reasoning chars: "
          f"{statistics.median(r['metadata']['reasoning_chars'] for r in rows):.0f}")
    print(f"\n-> {path}\nnext: uv run python scratch/feature_discovery/extract_features.py "
          f"--input {path} --out-dir {out}")


def assign(features_dir: str, embeddings: str | None = None,
           train_dir: str = "output/feature_discovery/20260812_092119",
           out_dir: str | None = None) -> None:
    """Assign ODCV features to the nearest training-corpus cluster and report prevalence.

    Args:
        features_dir: Directory holding odcv_traces.jsonl and features.jsonl.
        embeddings: embeddings.npy for the ODCV feature list; defaults to
            <features_dir>/embeddings.npy.
        train_dir: The feature-discovery run whose centroids define the clusters.
        out_dir: Output directory; defaults to `features_dir`.

    Raises:
        RuntimeError: If the embedding matrix does not match the unique-feature list.
    """
    fd = Path(features_dir)
    td = Path(train_dir)
    clusters = json.loads((td / "clusters.json").read_text())
    cl = {c["cluster"]: c for c in clusters["clusters"]}
    train_fmap = json.loads((td / "feature_cluster_map.json").read_text())
    train_uniq = [x for x in (td / "unique_features.txt").read_text().splitlines() if x.strip()]
    train_traces = [json.loads(x) for x in (td / "features.jsonl").read_text().splitlines()
                    if x.strip()]

    meta = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)["metadata"]
            for x in (fd / "odcv_traces.jsonl").read_text().splitlines() if x.strip()}
    odcv = [json.loads(x) for x in (fd / "features.jsonl").read_text().splitlines() if x.strip()]
    uniq = [x for x in (fd / "unique_features.txt").read_text().splitlines() if x.strip()]

    cen = _centroids(td / "embeddings.npy", train_uniq, train_fmap, len(cl))
    emb = np.load(embeddings or fd / "embeddings.npy")
    if emb.shape[0] != len(uniq):
        raise RuntimeError(f"embeddings {emb.shape} vs {len(uniq)} unique features")
    sims = np.asarray(emb, dtype=np.float32) @ cen.T
    best = sims.argmax(1)
    best_sim = sims.max(1)
    fmap = {f: int(best[i]) for i, f in enumerate(uniq)}
    simmap = {f: float(best_sim[i]) for i, f in enumerate(uniq)}

    n = len(odcv)
    tset = {t["scenario_id"]: {fmap[f] for f in t["features"]} for t in odcv}
    prev = Counter(c for s in tset.values() for c in s)
    train_prev = {c: cl[c]["n_traces"] / len(train_traces) for c in cl}
    mis = {s for s in tset if meta[s]["misaligned"]}
    ok = set(tset) - mis

    out = Path(out_dir or fd)
    lines = [f"# ODCV rollouts mapped onto the difficult-advice clusters", "",
             f"{n} rollouts, {sum(len(t['features']) for t in odcv)} features "
             f"({len(uniq)} unique), assigned to the nearest of {len(cl)} centroids from "
             f"`{train_dir}` (clusters NOT refit).", "",
             f"Assignment confidence: median cosine to the chosen centroid "
             f"{np.median(best_sim):.3f}, "
             f"{(best_sim < 0.6).sum()} of {len(uniq)} features below 0.60 "
             f"({(best_sim < 0.6).mean():.1%} — these are reasoning the training corpus has "
             f"no cluster for).", "",
             f"Misaligned rollouts (median severity ≥ {VIOLATION_THRESHOLD}): "
             f"{len(mis)} of {n} ({len(mis) / n:.1%}).", "",
             "## Top 25 clusters in the rollouts", "",
             "| cluster | label | rollouts | prevalence | corpus prevalence | ratio | "
             "misaligned | aligned |", "|--:|---|--:|--:|--:|--:|--:|--:|"]
    for c, k in prev.most_common(25):
        m = sum(1 for s in mis if c in tset[s]) / max(1, len(mis))
        a = sum(1 for s in ok if c in tset[s]) / max(1, len(ok))
        lines.append(f"| C{c} | {cl[c]['label']} | {k} | {k / n:.1%} | {train_prev[c]:.1%} | "
                     f"{k / n / max(1e-9, train_prev[c]):.2f}x | {m:.1%} | {a:.1%} |")

    lines += ["", "## Clusters most over- and under-represented vs the training corpus", "",
              "Clusters present in at least 5% of rollouts, by ratio.", "",
              "| cluster | label | rollouts | corpus | ratio |", "|--:|---|--:|--:|--:|"]
    ratios = sorted(((prev[c] / n / max(1e-9, train_prev[c]), c) for c in prev
                     if prev[c] / n >= 0.05), reverse=True)
    for r, c in ratios[:10] + [(None, None)] + ratios[-10:]:
        if c is None:
            lines.append("| | *…* | | | |")
            continue
        lines.append(f"| C{c} | {cl[c]['label']} | {prev[c] / n:.1%} | {train_prev[c]:.1%} | "
                     f"{r:.2f}x |")

    lines += ["", "## Clusters that separate misaligned from aligned rollouts", "",
              f"{len(mis)} misaligned vs {len(ok)} aligned. Clusters in ≥10% of either side, "
              "by difference.", "", "| cluster | label | misaligned | aligned | diff |",
              "|--:|---|--:|--:|--:|"]
    diffs = []
    for c in prev:
        m = sum(1 for s in mis if c in tset[s]) / max(1, len(mis))
        a = sum(1 for s in ok if c in tset[s]) / max(1, len(ok))
        if max(m, a) >= 0.10:
            diffs.append((m - a, c, m, a))
    diffs.sort(reverse=True)
    for dv, c, m, a in diffs[:10] + [(None, None, None, None)] + diffs[-10:]:
        if c is None:
            lines.append("| | *…* | | | |")
            continue
        lines.append(f"| C{c} | {cl[c]['label']} | {m:.1%} | {a:.1%} | {dv:+.1%} |")

    lines += ["", "## Features with no home cluster", "",
              "The 30 rollout features furthest from every training centroid — reasoning the "
              "training corpus does not contain.", ""]
    for f in sorted(uniq, key=lambda f: simmap[f])[:30]:
        lines.append(f"- `{simmap[f]:.3f}` {f} → nearest C{fmap[f]} {cl[fmap[f]]['label']}")

    (out / "cluster_assignment.md").write_text("\n".join(lines) + "\n")
    (out / "cluster_assignment.jsonl").write_text("".join(
        json.dumps({"scenario_id": t["scenario_id"], "condition": meta[t["scenario_id"]]["condition"],
                    "severity": meta[t["scenario_id"]]["severity"],
                    "misaligned": meta[t["scenario_id"]]["misaligned"],
                    "clusters": sorted(tset[t["scenario_id"]]),
                    "features": [{"feature": f, "cluster": fmap[f], "cos": round(simmap[f], 4)}
                                 for f in t["features"]]}) + "\n" for t in odcv))
    _dashboard(out / "cluster_assignment.html", odcv, meta, fmap, simmap, cl, prev, n)
    (out / "run_meta_assign.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "features_dir": str(fd),
         "train_dir": train_dir, "rollouts": n, "unique_features": len(uniq),
         "median_assignment_cosine": float(np.median(best_sim)),
         "low_confidence_features": int((best_sim < 0.6).sum()),
         "command": " ".join(sys.argv)}, indent=2))

    print(f"{n} rollouts, {len(uniq)} unique features, median assignment cosine "
          f"{np.median(best_sim):.3f}")
    for c, k in prev.most_common(12):
        print(f"  C{c:3d} {cl[c]['label'][:46]:46s} {k:4d} ({k / n:5.1%}) "
              f"corpus {train_prev[c]:5.1%}")
    print(f"-> {out}")


def _centroids(emb_path: Path, uniq: list[str], fmap: dict[str, int], k: int) -> np.ndarray:
    """Compute L2-normalised cluster centroids by streaming the embedding file.

    Args:
        emb_path: Path to the training embeddings.npy (n x d, fp16).
        uniq: Feature strings in embedding-row order.
        fmap: Feature string -> cluster id.
        k: Number of clusters.

    Returns:
        (k x d) centroid matrix, rows L2-normalised.
    """
    x = np.load(emb_path, mmap_mode="r")
    sums = np.zeros((k, x.shape[1]), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)
    labels = np.array([fmap[f] for f in uniq], dtype=np.int32)
    for start in range(0, len(uniq), 2048):
        block = np.asarray(x[start:start + 2048], dtype=np.float32)
        np.add.at(sums, labels[start:start + 2048], block)
        np.add.at(counts, labels[start:start + 2048], 1)
    assert counts.sum() == len(uniq), f"{counts.sum()} != {len(uniq)}"
    cen = sums / counts[:, None]
    return cen / np.linalg.norm(cen, axis=1, keepdims=True)


def _dashboard(path: Path, odcv: list[dict], meta: dict, fmap: dict, simmap: dict,
               cl: dict, prev: Counter, n: int) -> None:
    """Write a browsable per-rollout listing of assigned clusters.

    Args:
        path: Output html path.
        odcv: Per-rollout feature records.
        meta: scenario_id -> rollout metadata.
        fmap: feature -> cluster id.
        simmap: feature -> cosine to its centroid.
        cl: cluster id -> cluster record.
        prev: Cluster -> rollout count.
        n: Number of rollouts.
    """
    parts = ["<meta charset='utf-8'><title>ODCV rollouts by cluster</title>",
             "<style>body{font:14px/1.55 system-ui;margin:0 auto;max-width:1040px;padding:2rem;"
             "background:#fbfbfa}details{border:1px solid #ddd;border-radius:6px;margin:.4rem 0;"
             "background:#fff}summary{padding:.55rem .8rem;cursor:pointer}"
             "table{border-collapse:collapse;font-size:12px;margin:.5rem 0 .5rem 1rem}"
             "td,th{border:1px solid #e3e3e3;padding:.2rem .5rem;text-align:left}"
             ".mis{color:#b00;font-weight:600}code{background:#eee;padding:0 .25rem}</style>",
             f"<h1>ODCV rollouts mapped onto the training clusters</h1>"
             f"<p>{n} rollouts. Top clusters: " +
             ", ".join(f"C{c} {html.escape(cl[c]['label'])} ({k})"
                       for c, k in prev.most_common(8)) + "</p>"]
    for t in sorted(odcv, key=lambda t: (-(meta[t["scenario_id"]]["severity"] or 0),
                                         t["scenario_id"])):
        sid = t["scenario_id"]
        m = meta[sid]
        tag = "<span class='mis'>MISALIGNED</span> " if m["misaligned"] else ""
        cs = sorted({fmap[f] for f in t["features"]})
        parts.append(
            f"<details><summary>{tag}<code>{html.escape(sid)}</code> &mdash; severity "
            f"{m['severity']} &mdash; " + ", ".join(f"C{c}" for c in cs) + "</summary><table>"
            "<tr><th>feature</th><th>cluster</th><th>cos</th></tr>" +
            "".join(f"<tr><td>{html.escape(f)}</td><td>C{fmap[f]} "
                    f"{html.escape(cl[fmap[f]]['label'])}</td>"
                    f"<td>{simmap[f]:.3f}</td></tr>" for f in t["features"]) +
            "</table></details>")
    path.write_text("\n".join(parts))


if __name__ == "__main__":
    fire.Fire({"traces": traces, "assign": assign})
