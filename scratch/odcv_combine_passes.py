# ABOUTME: Merge N single-pass ODCV run dirs into the repeated `rollout_NNN` layout the judge reads.
# ABOUTME: Run: uv run python scratch/odcv_combine_passes.py --config configs/eval/odcv_bench_<arm>.yaml

"""Combine repeated ODCV-Bench rollout passes into one judgeable directory.

`rollouts_per_cell` in the ODCV configs is INERT — odcv_rollout.py reads it nowhere, so a
config saying 4 still produces exactly ONE rollout per cell. The 4 rollouts come from
invoking the driver four times (scratch/odcv_repeat_rollouts.sh), each pass writing its own
timestamped run-dir. `odcv_judge._transcripts` scores repeats only when it finds them as
per-scenario `rollout_NNN/` subdirectories, so the passes must be merged into that shape
first. This is the step prior arms did with an `odcv_add_pass.py` that was never committed.

Layout produced, which is exactly what `_transcripts` globs:

    <output_root>/<model_key>/combined<N>x_<ts>/
      agent_logs/<model_key>-<variant>/experiments/<Scenario>/rollout_NNN/messages_record.txt

The rollout index is the PASS index, so `rollout_002` means "the third pass" in every
scenario. A scenario missing from a pass simply has no directory for that index — the judge
globs `rollout_*`, so gaps are harmless and stay visible in the per-pass counts printed here.

A scenario directory is copied only when its `messages_record.txt` exists and is non-empty:
a zero-byte record is the silent no-transcript failure mode (agent could not emit tool
calls), and copying it would let the judge score an empty rollout as a clean one.

Cells are also tallied ACROSS passes, so a cell that landed fewer than N rollouts is named
rather than averaged over silently. That list is the input to a top-up pass
(scratch/odcv_topup_short_cells.py), which refills only the short cells.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv.odcv import VARIANTS  # noqa: E402
from src.utils import timestamp, write_run_meta  # noqa: E402


def _pass_dirs(model_root: Path, passes: tuple[str, ...]) -> list[Path]:
    """Resolve the pass directories to merge, oldest first.

    Args:
        model_root: `<output_root>/<model_key>`, holding one dir per pass.
        passes: Explicit pass directories. Empty means auto-discover every timestamped
            run dir under `model_root`, excluding previously combined outputs.

    Returns:
        Pass directories sorted by name (timestamped, so chronological).
    """
    if passes:
        dirs = [Path(p).resolve() for p in passes]
    else:
        dirs = sorted(p for p in model_root.iterdir()
                      if p.is_dir() and not p.name.startswith("combined")
                      and (p / "agent_logs").is_dir())
    assert dirs, f"no pass directories found under {model_root}"
    for d in dirs:
        assert (d / "agent_logs").is_dir(), f"not a rollout dir (no agent_logs): {d}"
    return dirs


def main(
    config: str,
    passes: tuple[str, ...] = (),
    out_name: str = "",
    dry_run: bool = False,
) -> None:
    """Merge repeated ODCV passes into one directory the judge can score.

    Args:
        config: The arm's ODCV config; supplies `output_root` and `model_key`.
        passes: Explicit pass directories, oldest first. Default: auto-discover.
        out_name: Name of the combined directory. Default `combined<N>x_<timestamp>`.
        dry_run: Report what would be copied without writing anything.
    """
    cfg = OmegaConf.load(config)
    model_key = str(cfg.model_key)
    model_root = Path(cfg.output_root).resolve() / model_key
    assert model_root.is_dir(), f"no runs for this arm yet: {model_root}"

    dirs = _pass_dirs(model_root, passes)
    out_dir = model_root / (out_name or f"combined{len(dirs)}x_{timestamp()}")
    assert not out_dir.exists(), f"refusing to overwrite existing {out_dir}"
    print(f"combining {len(dirs)} passes -> {out_dir}\n")

    counts: dict[str, dict[str, int]] = {}
    cell_counts: dict[str, int] = {}
    empty: list[str] = []
    for idx, pass_dir in enumerate(dirs):
        per_variant = {}
        for variant in VARIANTS:
            src = pass_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
            # An incentivized-only arm excludes all 40 mandated scenarios, so the driver
            # never creates that subtree. Report the zero rather than asserting: a variant
            # that vanished by accident still shows as 0 in the per-pass line below.
            if not src.is_dir():
                per_variant[variant] = 0
                continue
            dst_root = (out_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments")
            kept = 0
            for scen in sorted(src.iterdir(), key=lambda p: p.name.lower()):
                rec = scen / "messages_record.txt"
                if not (rec.is_file() and rec.stat().st_size > 0):
                    empty.append(f"{pass_dir.name}/{variant}/{scen.name}")
                    continue
                if not dry_run:
                    shutil.copytree(scen, dst_root / scen.name / f"rollout_{idx:03d}")
                kept += 1
                cell = f"{variant}/{scen.name}"
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
            per_variant[variant] = kept
        counts[pass_dir.name] = per_variant
        total = sum(per_variant.values())
        detail = "  ".join(f"{v}={n}" for v, n in per_variant.items())
        print(f"  rollout_{idx:03d}  {pass_dir.name}  {total:>3} transcripts   {detail}")

    n_total = sum(sum(v.values()) for v in counts.values())
    short = {c: n for c, n in cell_counts.items() if n != len(dirs)}
    print(f"\ntotal transcripts: {n_total}")
    print(f"cells: {len(cell_counts)} | with all {len(dirs)} rollouts: "
          f"{len(cell_counts) - len(short)}")
    if short:
        print(f"cells short of {len(dirs)} rollouts ({len(short)}):")
        for cell, n in sorted(short.items()):
            print(f"  {cell}: {n}")
    if empty:
        print(f"skipped {len(empty)} scenario(s) with a missing/empty messages_record.txt:")
        for e in empty:
            print(f"  {e}")
    if dry_run:
        print("\nDRY RUN — nothing written")
        return

    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True), extra={
        "combined_from": [str(d) for d in dirs],
        "n_passes": len(dirs),
        "n_transcripts": n_total,
        "per_pass_counts": counts,
        "cells_short": short,
        "skipped_empty": empty,
    })
    (out_dir / "combine_manifest.json").write_text(json.dumps({
        "model_key": model_key,
        "passes": [d.name for d in dirs],
        "per_pass_counts": counts,
        "n_transcripts": n_total,
        "cells_short": short,
        "skipped_empty": empty,
    }, indent=2))
    print(f"\n>>> {out_dir}")
    print(f"next: uv run python scratch/odcv_judge_cli.py --rollout_dir \"{out_dir}\" "
          f"--config {config}")


if __name__ == "__main__":
    fire.Fire(main)
