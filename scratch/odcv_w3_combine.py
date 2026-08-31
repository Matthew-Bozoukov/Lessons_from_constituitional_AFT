# ABOUTME: Combine the 3 separate-model_key ODCV worst-3 passes per arm into one unified
# ABOUTME: repeated rollout_NNN layout the validated odcv_judge can score. Run: uv run python scratch/odcv_w3_combine.py
import shutil
from pathlib import Path
import fire

ROOT = Path("output/odcv_worst3")
VARIANTS = ("mandated", "incentivized")


def _pass_dir(arm: str, p: int) -> Path:
    base = ROOT / arm / f"p{p}"
    cands = sorted(base.glob(f"odcv_worst3_{arm}_p{p}/2*"))
    assert cands, f"no timestamped pass dir under {base}"
    return cands[-1]


def main(arms: str = "numina,synthdoc"):
    """Build output/odcv_worst3/<arm>/combined_flash with rollout_000..002 per cell."""
    for arm in arms.split(","):
        out = ROOT / arm / "combined_flash"
        assert not out.exists(), f"refusing to overwrite {out}"
        cell_counts: dict[str, int] = {}
        for idx in range(3):
            pdir = _pass_dir(arm, idx + 1)
            src_key = f"odcv_worst3_{arm}_p{idx+1}"
            for variant in VARIANTS:
                src = pdir / "agent_logs" / f"{src_key}-{variant}" / "experiments"
                if not src.is_dir():
                    continue
                dst_root = out / "agent_logs" / f"{arm}-{variant}" / "experiments"
                for scen in sorted(src.iterdir()):
                    rec = scen / "messages_record.txt"
                    if not (rec.is_file() and rec.stat().st_size > 0):
                        continue
                    shutil.copytree(scen, dst_root / scen.name / f"rollout_{idx:03d}")
                    cell = f"{variant}/{scen.name}"
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
        full = sum(1 for n in cell_counts.values() if n == 3)
        print(f"{arm}: {len(cell_counts)} cells | with all 3 rollouts: {full} -> {out}")
        short = {c: n for c, n in cell_counts.items() if n != 3}
        if short:
            print(f"  short cells ({len(short)}): " + ", ".join(f"{c}:{n}" for c, n in sorted(short.items())))


if __name__ == "__main__":
    fire.Fire(main)
