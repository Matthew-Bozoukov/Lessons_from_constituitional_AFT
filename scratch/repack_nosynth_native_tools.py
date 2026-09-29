# ABOUTME: Repack the published nosynth base with its apigen tool rows parsed into native
# ABOUTME: `tools` + `tool_calls` fields; every other row, trace and count is carried verbatim.

"""One-off: the default nosynth base, with tool use stored as data instead of xLAM text.

Run (from the repo root):

    uv run python scratch/repack_nosynth_native_tools.py            # dry run, writes output/
    uv run python scratch/repack_nosynth_native_tools.py --push     # + pushes <date>-nosynth-mix

The published base's `apigen_function_calling` rows were passed through with their tool
schemas and calls as prompt TEXT (src/data/mixture/sources/apigen_function_calling.py
now parses them). A stored base row IS the raw smoltalk row after `clean_messages`, so
the adapter converts it directly: no resampling, no spec-filter or backfill calls, and
the rows keep the verdicts the old base's filter gave them. The script refuses to push
if any row fails to convert or outgrows `max_seq_len`, so the per-source counts (the
MSM Table 2 citation) stay exact.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mixture import build_mixture as bm  # noqa: E402
from src.data.mixture.sources import SOURCES  # noqa: E402
from src.infra.huggingface import resolve_dataset  # noqa: E402
from src.model_profile import model_profile, render_chat  # noqa: E402
from src.naming import mix_name  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402

TOOL_SOURCE = "apigen_function_calling"


def main(
    old_repo: str = "dougalldeepmind/2026-09-22-nosynth-mix",
    old_revision: str = "378ec1ee0f0eea9294683779438b839e52b9700a",
    config: str = "configs/data/mixture/nosynth.yaml",
    push: bool = False,
) -> None:
    files = {}
    for name in (
        "mixture.jsonl",
        "mixture_stats.json",
        "mixture_config.yaml",
        "verdicts.jsonl",
        "filter_report.json",
    ):
        files[name], ref = resolve_dataset(old_repo, name, old_revision)
    print(f"base: {old_repo}@{ref['revision'][:12]}")
    old_cfg = OmegaConf.load(files["mixture_config.yaml"])
    old_stats = json.loads(Path(files["mixture_stats.json"]).read_text())
    rows = [json.loads(line) for line in open(files["mixture.jsonl"], encoding="utf-8")]

    adapter = SOURCES[TOOL_SOURCE]
    failed, converted = [], 0
    for i, r in enumerate(rows):
        if r["source"] != TOOL_SOURCE:
            continue
        msgs, tools = adapter.to_messages(r), adapter.to_tools(r)
        if msgs is None:
            failed.append(i)
            continue
        rows[i] = {
            "messages": msgs,
            "source": TOOL_SOURCE,
            **({"tools": tools} if tools else {}),
        }
        converted += 1
    assert not failed, (
        f"{len(failed)} {TOOL_SOURCE} rows did not convert (rows {failed[:10]})"
    )

    tok = AutoTokenizer.from_pretrained(str(old_cfg.tokenizer))
    profile = model_profile(str(old_cfg.tokenizer))
    for r in rows:
        r["n_tokens"] = len(
            render_chat(
                tok,
                r["messages"],
                r.get("tools"),
                render_kwargs=profile.render_kwargs,
                tokenize=True,
                return_dict=True,
            )["input_ids"]
        )
    too_long = [
        i for i, r in enumerate(rows) if r["n_tokens"] > int(old_cfg.max_seq_len)
    ]
    assert not too_long, (
        f"{len(too_long)} rows exceed max_seq_len after repacking: {too_long[:10]}"
    )

    out_dir = Path("output/mixture_base") / f"{timestamp()}_nosynth_native_tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    bm._write_rows(out_path, rows)
    kinds = {name: str(spec.reasoning) for name, spec in old_cfg.sources.items()}
    kinds = {
        n: (
            "mixed"
            if any(
                m.get("reasoning_content")
                for r in rows
                if r["source"] == n
                for m in r["messages"]
            )
            else k
        )
        for n, k in kinds.items()
    }
    bm._validate_written(out_path, rows, kinds)
    for name in ("verdicts.jsonl", "filter_report.json"):
        shutil.copy(files[name], out_dir / name)

    stats = {
        **old_stats,
        "total": {"examples": len(rows), "tokens": sum(r["n_tokens"] for r in rows)},
        "by_source": bm._source_stats(rows),
        "mixture_path": str(out_path),
        "repacked_from": {
            "repo": old_repo,
            "revision": ref["revision"],
            "source": TOOL_SOURCE,
            "rows_converted": converted,
            "change": "tool schemas and calls parsed from xLAM prompt text "
            "into native `tools` + `tool_calls`",
        },
    }
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    command = " ".join(
        ["uv run python scratch/repack_nosynth_native_tools.py", *sys.argv[1:]]
    )
    write_run_meta(
        out_dir,
        OmegaConf.to_container(old_cfg, resolve=True),
        extra={"command": command, "stats": stats},
    )
    old_tok = old_stats["by_source"][TOOL_SOURCE]["tokens"]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "converted": converted,
                f"{TOOL_SOURCE}_tokens": {
                    "before": old_tok,
                    "after": stats["by_source"][TOOL_SOURCE]["tokens"],
                },
                "total_tokens": {
                    "before": old_stats["total"]["tokens"],
                    "after": stats["total"]["tokens"],
                },
                "max_n_tokens": max(r["n_tokens"] for r in rows),
            },
            indent=2,
        )
    )
    print(f">>> wrote {out_path}")

    card_cfg = OmegaConf.merge(
        old_cfg,
        OmegaConf.create(
            {
                "hf": {
                    "experiment": (
                        f"{old_cfg.hf.experiment}; tool-use rows ({TOOL_SOURCE}) carry native `tools` + "
                        "`tool_calls`, rendered in each model family's own tool syntax at train time"
                    )
                }
            }
        ),
    )
    OmegaConf.save(card_cfg, out_dir / "mixture_config.yaml")
    filter_cfg = card_cfg.filter
    report = json.loads(Path(files["filter_report.json"]).read_text())
    fields = bm._card_fields(
        card_cfg,
        config,
        "final training mixture (repacked: native tool fields)",
        "mixture.jsonl + mixture_stats.json + verdicts.jsonl + filter_report.json",
        filter_cfg,
        report,
        old_stats.get("reasoning_traces"),
    )
    fields["provenance"] = (
        f"{command} (from {old_repo}@{ref['revision']}; "
        f"repo at the commit in source_repo)"
    )
    fm = bm._front_matter(card_cfg, config, filter_cfg, "final", out_path.name)
    repo = mix_name("", 0)
    if not push:
        print(f">>> dry run: would push -> {repo}")
        return
    bm._push(
        [
            out_path,
            out_dir / "mixture_stats.json",
            out_dir / "run_meta.json",
            out_dir / "mixture_config.yaml",
            out_dir / "verdicts.jsonl",
            out_dir / "filter_report.json",
        ],
        repo,
        fields,
        False,
        False,
        fm,
    )


if __name__ == "__main__":
    fire.Fire(main)
