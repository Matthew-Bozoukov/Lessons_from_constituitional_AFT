# ABOUTME: Surveys the OAI-HF reproduction paper's Part 1 logs: per step and model, peak context per
# ABOUTME: target call, turns, limits/errors, judge tokens, and Step 3 conditional share rates.
# Run: uv run --project ~/projects/oai-hf-incident-reproduction python scratch/oai_hf_repro/survey_paper_logs.py
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log

R = os.path.expanduser(os.environ.get("OAI_HF_LOGS", "~/projects/oai-hf-incident-reproduction/logs"))
CELLS = {
    "step1": {
        "GLM-5.2": f"{R}/260905_step1_xlsx_glm52_n64/step1_gdrive_xlsx/glm52__shard*/*.eval",
        "Qwen 3.8 Max": f"{R}/260905_step1_xlsx_n64/step1_gdrive_xlsx/qwen38max__shard*/*.eval",
    },
    "step2": {
        "GLM-5.2": f"{R}/260906_step2_reachout_protein_none_n256/step2_reachout/*/*.eval",
        "Qwen 3.8 Max": f"{R}/260909_step2_bank_none_multimodel/protein_none/qwen38max__shard*/*.eval",
    },
    "step3": {
        m: f"{R}/260909_step3_egress_share_n64/react__{a}__driven/shard*/*.eval"
        for m, a in [
            ("GLM-5.2", "glm52"),
            ("GLM-5.3", "glm53"),
            ("Qwen 3.8 Max", "qwen38max"),
            ("Kimi K3", "kimik3"),
            ("Grok 4.6", "grok46"),
            ("Claude Opus 4.8", "opus48"),
        ]
    },
    "step4": {
        "GLM-5.2": f"{R}/260906_step4chain_n64_softdrive/**/*.eval",
        "Qwen 3.8 Max": f"{R}/260909_step4chain_n64_multimodel/step4_hf_cheat_chain/qwen38max0902__shard*/*.eval",
    },
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def survey(pattern):
    rows = []
    seen = set()
    for f in sorted(glob.glob(pattern, recursive=True)):
        log = read_eval_log(f)
        for s in log.samples or []:
            key = (f.rsplit("/", 2)[-2], s.id, s.epoch)
            if key in seen:
                continue
            seen.add(key)
            sc = next(iter(s.scores.values())) if s.scores else None
            meta = (sc.metadata or {}) if sc else {}
            target = log.eval.model
            ctx, calls, judge_in = [], 0, 0
            for e in s.events:
                if e.event != "model" or not e.output or not e.output.usage:
                    continue
                u = e.output.usage
                tot = (
                    (u.input_tokens or 0)
                    + (u.input_tokens_cache_read or 0)
                    + (u.input_tokens_cache_write or 0)
                )
                if e.model == target:
                    calls += 1
                    ctx.append(tot + (u.output_tokens or 0))
                else:
                    judge_in += tot
            rows.append(
                dict(
                    calls=calls,
                    peak_ctx=max(ctx, default=0),
                    judge_in=judge_in,
                    limit=s.limit.type if s.limit else None,
                    error=bool(s.error),
                    sig=bool(meta.get("signature")),
                    solved=meta.get("solved"),
                    found_board=meta.get("found_board"),
                    got_egress=meta.get("got_egress"),
                )
            )
    return rows


out = {}
for step, cells in CELLS.items():
    for model, pat in cells.items():
        rows = survey(pat)
        if not rows:
            print(f"{step} {model}: NO FILES {pat}", file=sys.stderr)
            continue
        n = len(rows)
        pk = sorted(r["peak_ctx"] for r in rows)
        line = (
            f"{step} {model:16} n={n:3} calls med {st.median(r['calls'] for r in rows):5.0f} max {max(r['calls'] for r in rows):3} | "
            f"peak ctx med {st.median(pk) / 1e3:5.1f}k p90 {pk[int(0.9 * (n - 1))] / 1e3:5.1f}k max {pk[-1] / 1e3:5.1f}k "
            f">32k {sum(p > 32768 for p in pk):2} | limits {dict((l, sum(r['limit'] == l for r in rows)) for l in {r['limit'] for r in rows} if l)} "
            f"err {sum(r['error'] for r in rows)} | judge in med {st.median(r['judge_in'] for r in rows) / 1e3:.1f}k | "
            f"scorer sig {sum(r['sig'] for r in rows)}/{n}"
        )
        if step == "step3":
            sb = [r for r in rows if r["solved"] and r["found_board"]]
            k = sum(r["sig"] for r in sb)
            p, lo, hi = wilson(k, len(sb))
            line += (
                f" | solved {sum(bool(r['solved']) for r in rows)} found_board {sum(bool(r['found_board']) for r in rows)} "
                f"both {len(sb)} -> share|both {k}/{len(sb)} = {p:.2f} [{lo:.2f},{hi:.2f}]"
            )
        print(line)
        out[f"{step}|{model}"] = rows
json.dump(out, open(os.path.join(os.makedirs("output/oai_hf_repro", exist_ok=True) or "output/oai_hf_repro", "paper_survey.json"), "w"))
