# ABOUTME: Downloads the two trained DA mixtures from the Nov 18 paper's Figure 5 (default Claude DA vs all-GPT DA)
# ABOUTME: into data/gpt_vs_claude_da/, keeping only the synthetic difficult-advice rows, and records the pinned revisions.
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
OUT = Path("data/gpt_vs_claude_da")
ARMS = {
    "claude": "dougalldeepmind/2026-10-05-da-15-mix",
    "gpt": "dougalldeepmind/2026-11-04-da-gpt-15-mix",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    pins = {}
    for arm, repo in ARMS.items():
        info = api.dataset_info(repo)
        files = [s.rfilename for s in info.siblings]
        print(arm, repo, info.sha, files)
        path = hf_hub_download(repo, "mixture.jsonl", repo_type="dataset", revision=info.sha)
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        pins[arm] = {"repo": repo, "revision": info.sha, "n_rows": len(rows)}
        (OUT / f"{arm}_mixture.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    (OUT / "pins.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
