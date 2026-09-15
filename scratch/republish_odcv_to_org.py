# ABOUTME: Copy already-contract-shaped ODCV eval repos from a personal HF namespace into the
# ABOUTME: org, adding the eval:/model:/mode: tags the dashboard's discovery reads.
# Run: uv run python scratch/republish_odcv_to_org.py [--push False] [--sources a,b,...]

"""Republish ODCV eval repos into LASR-Callum, verbatim, with dashboard tags.

The six seed-sweep repos under `matboz/` already satisfy the published-layout contract
(rollouts/ results/ metadata/ + card) and carry `eval-run`, but the dashboard resolves an
eval run's name/model/mode from `eval:<name>`, `model:<key>`, `mode:<mode>` tags
(dashboard/lib/evalRuns.ts:listEvalRuns), and discovery is scoped to the org
(`author=LASR-Callum&filter=eval-run`). So: same bytes, org namespace, three more tags.

`model:` is read from results/results.json (`model_key`), falling back to the card's
`models` field. `mode:` comes from metadata/run_meta.json when it records one, else the
`--mode` default. Sibling links inside the card are rewritten to the org copies so the
three seeds of an arm cross-link. Nothing is deleted from the source repos.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.eval.layout import assert_layout
from src.infra.huggingface import hf_api

SOURCES = [
    "matboz/2026-08-19-odcv-numina-control-716-seed0",
    "matboz/2026-08-26-odcv-numina-control-716-seed42",
    "matboz/2026-08-26-odcv-numina-control-716-seed69",
    "matboz/2026-08-24-odcv-synthdoc-716-seed0-rollout002",
    "matboz/2026-08-26-odcv-synthdoc-716-seed42",
    "matboz/2026-08-26-odcv-synthdoc-716-seed69",
]
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def _split_card(text: str) -> tuple[dict, str]:
    m = _FRONT.match(text)
    assert m, "card has no YAML front matter"
    return yaml.safe_load(m.group(1)) or {}, text[m.end():]


def _model_key(root: Path, body: str) -> str:
    res = json.loads((root / "results/results.json").read_text())
    key = res.get("model_key") or res.get("model")
    if not key:
        m = re.search(r"policy `([^`]+)`", body)
        assert m, "no model_key in results.json and no policy id in the card"
        key = m.group(1).split("/")[-1]
    return str(key)


def _mode(root: Path, default: str) -> str:
    meta = root / "metadata/run_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        for k in ("mode", "thinking_mode"):
            if m.get(k):
                return str(m[k])
        cfg = m.get("config") or {}
        if isinstance(cfg, dict) and cfg.get("mode"):
            return str(cfg["mode"])
    return default


def main(push: bool = True, org: str = "LASR-Callum", mode: str = "think",
         sources: str | None = None) -> None:
    load_dotenv()
    api = hf_api()
    srcs = sources.split(",") if sources else SOURCES
    names = {s.split("/")[1] for s in srcs}
    for source in srcs:
        name = source.split("/")[1]
        dest = f"{org}/{name}"
        snap = Path(snapshot_download(source, repo_type="dataset"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / name
            shutil.copytree(snap, root, symlinks=False,
                            ignore=shutil.ignore_patterns(".cache", ".git", ".gitattributes"))
            assert_layout(root)
            front, body = _split_card((root / "README.md").read_text(encoding="utf-8"))
            tags = list(front.get("tags") or [])
            key, md = _model_key(root, body), _mode(root, mode)
            for t in (f"eval:odcv", f"model:{key}", f"mode:{md}"):
                prefix = t.split(":")[0] + ":"
                tags = [x for x in tags if not x.startswith(prefix)] + [t]
            front["tags"] = tags
            # Sibling links -> the org copies; the source namespace is recorded once, below.
            for n in names:
                body = body.replace(f"datasets/{source.split('/')[0]}/{n}", f"datasets/{org}/{n}")
            note = (f"\n> Republished verbatim from `{source}` into the org so the dashboard "
                    f"can discover it (`scratch/republish_odcv_to_org.py`).\n")
            title_end = body.find("\n", body.find("# ")) + 1
            body = body[:title_end] + note + body[title_end:]
            card = "---\n" + yaml.safe_dump(front, sort_keys=False).rstrip() + "\n---\n" + body
            (root / "README.md").write_text(card, encoding="utf-8")
            n_roll = sum(1 for _ in root.glob("rollouts/*/*/pass*/messages_record.txt"))
            print(f"{source} -> {dest}\n    tags={tags}\n    rollouts={n_roll}  "
                  f"model_key={key}  mode={md}")
            if not push:
                continue
            api.create_repo(dest, repo_type="dataset", private=False, exist_ok=True)
            api.upload_folder(folder_path=str(root), repo_id=dest, repo_type="dataset",
                              commit_message=f"Republish {source} into {org} with dashboard tags")
            print(f"    pushed https://huggingface.co/datasets/{dest}")
    if push:
        seen = {d.id: d.tags for d in api.list_datasets(author=org, filter="eval-run", limit=500)}
        for source in srcs:
            dest = f"{org}/{source.split('/')[1]}"
            got = [t for t in (seen.get(dest) or []) if t.startswith(("eval:", "model:", "mode:"))]
            print(f"discoverable: {dest} -> {'YES ' + str(got) if dest in seen else 'NO'}")


if __name__ == "__main__":
    fire.Fire(main)
