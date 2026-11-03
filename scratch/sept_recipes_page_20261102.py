# ABOUTME: Builds the September-recipes page: DA, low-stakes DA and nonmoral pipelines, stage by stage, from corpus manifests.
# ABOUTME: Run: uv run --no-sync python scratch/sept_recipes_page_20261102.py (writes output/2026-11-02_sept_recipes.html)
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()

# The corpora each September arm was trained from. The manifest's `config` is the resolved config that ran.
CORPORA = {
    "da": "dougalldeepmind/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716",
    "v2": "dougalldeepmind/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted",
    "low": "dougalldeepmind/2026-08-26-difficult-advice-low-stakes-716",
    "nm": "dougalldeepmind/2026-09-02-craft-tensions-nonmoral-deliberation",
}
TEMPLATE = Path(__file__).with_name("sept_recipes_page_20261102.html")
OUT = Path("output/2026-11-02_sept_recipes.html")


def lint_summary(lint):
    if not lint:
        return None
    return {"patterns": len(lint.get("ban_patterns", [])), "retries": lint.get("retries"), "min_chars": lint.get("min_chars")}


def recipe(repo):
    sha = HfApi().dataset_info(repo).sha
    m = json.load(open(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=sha), encoding="utf-8"))
    c = m["config"]
    stages = []
    for s in c["stages"]:
        model = c["models"].get(s.get("model") or "", {})
        stages.append({"name": s["name"], "kind": s.get("kind"), "model": model.get("model"),
                       "temperature": model.get("temperature"), "prompts": s.get("prompts") or {},
                       "lint": lint_summary(s.get("lint"))})
    return {"repo": repo, "sha": sha, "git_sha": m.get("git_sha"), "pipeline": c.get("pipeline"),
            "constitution": c.get("constitution"), "counts": m.get("counts"), "stages": stages}


def main():
    data = {k: recipe(r) for k, r in CORPORA.items()}
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    for k, d in data.items():
        print(k, d["sha"][:8], d["git_sha"][:8] if d["git_sha"] else None, [s["name"] for s in d["stages"]])
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
