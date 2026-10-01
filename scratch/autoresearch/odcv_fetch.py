# ABOUTME: Download ODCV-lite eval runs (results + rollouts transcripts) for the DA arms under study from the Hub
# ABOUTME: into output/autoresearch/odcv_runs/<run>/ (read-only; no pushes).
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(".env")
ORG = "dougalldeepmind"
RUNS = {
    "nosynth_a": "2026-09-21-odcv-qwen36-0-nosynth",
    "nosynth_b": "2026-09-22-odcv-qwen36-0-nosynth",
    "s25_s0": "2026-09-26-odcv-qwen36-0-da-15",
    "s25_s1": "2026-09-26-odcv-qwen36-1-da-15",
    "s25_no_t6": "2026-09-25-odcv-qwen36-0-da-no-t6-15",
    "s28_base_a": "2026-09-29-odcv-qwen36-0-da-15",
    "s28_base_b": "2026-09-30-odcv-qwen36-0-da-15",
    "s28_da5": "2026-09-28-odcv-qwen36-0-da-5",
    "s28_da25": "2026-09-28-odcv-qwen36-0-da-25",
    "swap_self": "2026-09-29-odcv-qwen36-0-da-15-self",
    "swap_otherai": "2026-09-29-odcv-qwen36-0-da-15-otherai",
    "swap_self_otherai": "2026-09-29-odcv-qwen36-0-da-15-self-otherai",
    "sysdiv": "2026-09-29-odcv-qwen36-0-da-15-sysdiv",
    "swap_explicit": "2026-09-30-odcv-qwen36-0-da-15-explicit",
    "swap_advice": "2026-09-30-odcv-qwen36-0-da-15-advice",
    "s28_no_t6": "2026-09-29-odcv-qwen36-0-da-no-t6-15",
    "new_self": "2026-10-01-odcv-qwen36-0-da-self-15",
    "new_otherai": "2026-10-01-odcv-qwen36-0-da-otherai-15",
    "new_explicit": "2026-10-01-odcv-qwen36-0-da-explicit-15",
    "lowstakes": "2026-09-25-odcv-qwen36-0-da-lowstakes-practical-15",
    "nonmoral": "2026-09-25-odcv-qwen36-0-nonmoral-original-15",
    "tools": "2026-09-28-odcv-qwen36-0-da-tools-15",
    "s14_a": "2026-09-21-odcv-qwen36-0-da-15",
    "s14_b": "2026-09-22-odcv-qwen36-0-da-15",
    "delib": "2026-09-22-odcv-qwen36-0-delib-15",
}


def fetch(run: str) -> str:
    return snapshot_download(f"{ORG}/{run}", repo_type="dataset", local_dir=f"output/autoresearch/odcv_runs/{run}",
                             allow_patterns=["results/*", "metadata/run_meta.json", "metadata/odcv_config.yaml",
                                             "rollouts/*/*/*/messages_record.txt", "rollouts/*/*/*/cell_meta.json"])


if __name__ == "__main__":
    with ThreadPoolExecutor(6) as ex:
        for p in ex.map(fetch, RUNS.values()):
            print(p)
