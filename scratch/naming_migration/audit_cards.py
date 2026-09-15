# ABOUTME: Audit (and with --apply, repair) the CARDS on the Hub: a renamed repo's README
# ABOUTME: still names the old repo ids and the old config paths it was generated from.
"""Run: uv run python scratch/naming_migration/audit_cards.py [--apply] [--org LASR-Callum]

Renaming a repo does not touch the text inside it. Every adapter card names the dataset it
trained on (`dataset: hf.co/datasets/<repo>@<sha>`) and the config that made it
(`provenance: uv run train --config configs/train/<stem>.yaml`); every corpus card names
its own pipeline config. After the migration those strings point at ids that only resolve
through a redirect, and at config paths that do not exist at all.

Stale references are resolved the honest way: an old repo id is looked up on the Hub and
replaced by whatever it is called NOW; an old config path is replaced from git's own
rename record. Anything that cannot be resolved is reported, never guessed.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.infra.huggingface import hf_api, hf_download
from src.naming import NamingError, check_hub_repo

ROOT = Path(__file__).resolve().parents[2]
# A card is prose: an id at the end of a sentence carries the full stop with it, and a
# trailing separator belongs to the sentence, not the name.
REPO_REF = re.compile(r"\b((?:LASR-Callum|matboz)/[A-Za-z0-9._-]*[A-Za-z0-9])")
CONFIG_REF = re.compile(r"\b(configs/[A-Za-z0-9_/.-]+\.yaml)\b")


def config_renames() -> dict[str, str]:
    """old config path -> new one, from git's rename detection over this branch."""
    out = subprocess.run(
        ["git", "diff", "-M", "--diff-filter=R", "--name-status", "origin/main~1", "HEAD",
         "--", "configs"], cwd=ROOT, capture_output=True, text=True).stdout
    return {old: new for line in out.splitlines() if line.strip()
            for _, old, new in [line.split("\t")]}


def current_id(api, repo_id: str, cache: dict) -> str | None:
    """What that repo is called now (follows the rename), or None if it is gone."""
    if repo_id in cache:
        return cache[repo_id]
    for repo_type in ("model", "dataset"):
        try:
            cache[repo_id] = api.repo_info(repo_id, repo_type=repo_type).id
            return cache[repo_id]
        except Exception:  # noqa: BLE001 - a missing repo is a finding, not a crash
            continue
    cache[repo_id] = None
    return None


def main(apply: bool = False, org: str = "LASR-Callum") -> None:
    api = hf_api()
    configs, cache = config_renames(), {}
    repos = [(m.id, "model") for m in api.list_models(author=org)] + \
            [(d.id, "dataset") for d in api.list_datasets(author=org)]
    stale, unresolved, fixed = 0, [], 0
    for repo_id, repo_type in sorted(repos):
        try:
            card = Path(hf_download(repo_id, "README.md", repo_type=repo_type)).read_text()
        except Exception:  # noqa: BLE001 - no card is a different problem
            continue
        new_card, notes = card, []
        for ref in sorted(set(REPO_REF.findall(card)), key=len, reverse=True):
            try:
                check_hub_repo(ref, write=False)
                continue                      # lawful (or enumerated legacy): leave it
            except NamingError:
                pass
            now = current_id(api, ref, cache)
            if now and now != ref:
                new_card = new_card.replace(ref, now)
                notes.append(f"{ref} -> {now}")
            elif not now:
                unresolved.append(f"{repo_id}: {ref} does not resolve")
        for ref in sorted(set(CONFIG_REF.findall(card)), key=len, reverse=True):
            if ref in configs:
                new_card = new_card.replace(ref, configs[ref])
                notes.append(f"{ref} -> {configs[ref]}")
            elif not (ROOT / ref).exists():
                unresolved.append(f"{repo_id}: {ref} is not a config any more")
        if not notes:
            continue
        stale += 1
        print(f"{repo_id}\n    " + "\n    ".join(notes))
        if apply:
            api.upload_file(path_or_fileobj=new_card.encode(), path_in_repo="README.md",
                            repo_id=repo_id, repo_type=repo_type,
                            commit_message="cards: point at the current names (naming law)")
            fixed += 1
    print(f"\n{stale} cards with stale references" + (f", {fixed} rewritten" if apply else ""))
    if unresolved:
        print(f"\n{len(unresolved)} reference(s) nothing can resolve — decide by hand:")
        for u in sorted(set(unresolved)):
            print("   ", u)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
