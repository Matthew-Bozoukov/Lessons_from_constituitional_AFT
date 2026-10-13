# ABOUTME: Downloads published canary eval runs from the Hub into the local layout count_canary.py and deliberation.py
# ABOUTME: read (output/mask/<run>, output/odcv/<run>, output/colosseum_hospital/<root>/merged/<run>).
# Run: uv run python -m scratch.canary.pull_runs
from pathlib import Path

from huggingface_hub import snapshot_download

ORG = "dougalldeepmind"
# Hub run -> local dir. The DA + tools arm with tools written per row on Jamie's data (adapter 2026-10-09).
RUNS = {
    "2026-10-11-mask-qwen36-0-da-tools-15-canary": "output/mask/2026-10-11_qwen36_0_da_tools_15_canary_hub",
    "2026-10-11-odcv-qwen36-0-da-tools-15-canary": "output/odcv/2026-10-11_qwen36_0_da_tools_15_canary_hub",
    "2026-10-12-hospital-self-sacrificial-qwen36-0-da-tools-15-canary": (
        "output/colosseum_hospital/2026-10-12_canary_written/merged/"
        "2026-10-12_hospital_self_sacrificial_qwen36_0_da_tools_15_canary"
    ),
}


def main() -> None:
    for repo, local in RUNS.items():
        snapshot_download(f"{ORG}/{repo}", repo_type="dataset", local_dir=local)
        n = sum(1 for _ in Path(local).rglob("*") if _.is_file())
        print(f"{repo} -> {local} ({n} files)")


if __name__ == "__main__":
    main()
