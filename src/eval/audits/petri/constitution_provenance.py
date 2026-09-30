# ABOUTME: Resolve an explicit Petri constitution or the exact one in training provenance.
# ABOUTME: Never infer a constitution from a model name or silently read a newer local file.

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parents[4]


def snapshot(text: str, source: dict, *, selection: str) -> dict:
    if not text.strip():
        raise ValueError("Constitution is empty")
    return {"text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "selection": selection, "source": source}


def _declaration(value: object) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    if not isinstance(value, str):
        raise ValueError("Ambiguous constitution provenance: expected one path, not a list/object")
    # Historical cards append explanatory prose to the path. Multiple paths are
    # deliberately not collapsed to whichever one happened to occur first.
    paths = re.findall(r"constitutions/[A-Za-z0-9_./-]+\.md", value.replace("\\", "/"))
    if paths:
        if len(set(paths)) != 1:
            raise ValueError(f"Multiple training constitutions declared: {sorted(set(paths))}")
        return paths[0]
    raise ValueError(f"Unresolvable constitution declaration {value!r}; pass --constitution explicitly")


def _git_text(path: str, revision: str, root: Path) -> str:
    if not re.fullmatch(r"[a-fA-F0-9]{40}", str(revision or "")):
        raise ValueError("Constitution provenance lacks an exact source git SHA; use --constitution")
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=root,
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Cannot recover {path} at git {revision}; fetch that history or pass --constitution")
    return result.stdout.decode("utf-8")


def _from_record(record: dict, source: dict, *, git_reader: Callable) -> dict | None:
    cfg = record.get("train_config") or record.get("config") or {}
    declarations = [record.get("constitution"), cfg.get("constitution"),
                    (cfg.get("hf") or {}).get("constitution"),
                    (cfg.get("filter") or {}).get("constitution")]
    paths = {_declaration(value) for value in declarations if value is not None}
    paths.discard(None)
    if len(paths) > 1:
        raise ValueError(f"Training provenance names multiple constitutions: {sorted(paths)}")
    text = record.get("constitution_text")
    if text is not None:
        if not isinstance(text, str):
            raise ValueError("constitution_text must be text")
    elif paths:
        revision = record.get("git_sha")
        if not revision:
            match = re.search(r"@\s*([a-fA-F0-9]{40})\b", str(record.get("source_repo", "")))
            revision = match.group(1) if match else None
        text = git_reader(next(iter(paths)), revision)
        source = {**source, "path": next(iter(paths)), "git_sha": revision}
    else:
        return None
    # Synth manifests hash the stripped, newline-normalized prompt text. Retain
    # the full recovered document as well and identify that historical hash rule.
    claimed = record.get("constitution_sha256")
    normalized = text.replace("\r\n", "\n").strip()
    hashes = {hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(normalized.encode()).hexdigest()}
    if claimed and claimed not in hashes:
        raise ValueError("Constitution text does not match its recorded training hash")
    if claimed:
        source = {**source, "training_recorded_sha256": claimed}
    return snapshot(text, source, selection="training_provenance")


def _dataset_candidates(repo: str, revision: str, *, hub, download, git_reader,
                        ancestry: tuple = ()) -> list[dict]:
    if not re.fullmatch(r"[a-fA-F0-9]{40}", str(revision or "")):
        raise ValueError(f"Unpinned training source {repo}; pass --constitution or repair provenance")
    identity = (repo, revision)
    if identity in ancestry:
        raise ValueError(f"Cycle in training-data provenance: {repo}@{revision}")
    ancestry = (*ancestry, identity)
    source = {"dataset": {"repo": repo, "revision": revision}}
    info = hub.dataset_info(repo, revision=revision)
    files = {item.rfilename for item in info.siblings or []}
    candidates, records = [], []
    for filename in ("run_meta.json", "metadata/run_meta.json", "manifest.json"):
        if filename in files:
            path = Path(download(repo, filename, repo_type="dataset", revision=revision))
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(record)
            found = _from_record(record, {**source, "metadata_file": filename,
                "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, git_reader=git_reader)
            if found:
                candidates.append(found)
    # Mixture's own declaration is checked against each constitution-bearing
    # synthetic source. The actual sampled, pinned stats.sources supersede the
    # requested cfg.sources; raw nonconstitutional replay datasets are not chased.
    children = set()
    for record in records:
        cfg = record.get("config") or {}
        sources = (record.get("stats") or {}).get("sources") or cfg.get("sources") or {}
        for spec in sources.values():
            if not isinstance(spec, dict):
                continue
            child_repo = spec.get("dataset") or (spec.get("repo") if spec.get("synthetic") else None)
            if child_repo:
                children.add((child_repo, spec.get("revision")))
        base = cfg.get("base_mixture") or {}
        if isinstance(base, dict) and base.get("repo"):
            children.add((base["repo"], base.get("revision")))
    for child_repo, child_revision in sorted(children, key=lambda item: str(item)):
        candidates.extend(_dataset_candidates(child_repo, child_revision, hub=hub,
            download=download, git_reader=git_reader, ancestry=ancestry))
    if not candidates and "README.md" in files:
        path = Path(download(repo, "README.md", repo_type="dataset", revision=revision))
        text = path.read_text(encoding="utf-8")
        front = re.match(r"\A---\s*\n(.*?)\n---(?:\n|$)", text, re.S)
        card = (yaml.safe_load(front.group(1)) if front else {}) or {}
        for field in ("constitution", "source_repo"):
            row = re.search(rf"(?m)^\|\s*`?{field}`?\s*\|\s*(.*?)\s*\|\s*$", text)
            if row:
                card[field] = row.group(1)
        found = _from_record(card, {**source, "metadata_file": "README.md"}, git_reader=git_reader)
        if found:
            candidates.append(found)
    return candidates


def _pin_target_revision(target: str, revision: str | None, hub) -> str:
    if target.startswith("tinker://"):
        return revision or target
    if Path(target).is_dir():
        if not revision:
            raise ValueError("Local model target needs --target-revision identifying its immutable checkpoint")
        return revision
    if re.fullmatch(r"[a-fA-F0-9]{40}", str(revision or "")):
        return revision
    if hub is None:
        from src.infra.huggingface import hf_api
        hub = hf_api()
    info = hub.model_info(target, revision=revision) if revision else hub.model_info(target)
    if not re.fullmatch(r"[a-fA-F0-9]{40}", str(info.sha or "")):
        raise ValueError("Hub target did not resolve to an exact checkpoint commit")
    return info.sha


def resolve_constitution(target: str, *, target_revision: str | None = None,
                         explicit: str | None = None, training_meta: str | None = None,
                         root: Path = ROOT, hub=None, download=None,
                         git_reader=None) -> tuple[dict, dict]:
    """Resolve bytes plus target identity; helpers are injectable for offline tests.

    Explicit files win. Otherwise read the target's pinned training_meta.json,
    then its pinned training dataset's run_meta.json/card. No fuzzy name lookup,
    current-main substitution, or recursion through unpinned source datasets.
    An explicit local training_meta is the provenance route for Tinker checkpoints.
    """
    identity = {"target": target, "revision": target_revision}
    if explicit:
        identity["revision"] = _pin_target_revision(target, target_revision, hub)
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        # A unique constitution folder name is a supported convenience; never
        # choose the first of several archived/current matches.
        if not path.is_file():
            matches = list((root / "constitutions").rglob(f"{explicit}/constitution.md")) if "/" not in explicit and "\\" not in explicit else []
            if len(matches) != 1:
                raise ValueError(f"--constitution must name an existing file or unique folder: {explicit}")
            path = matches[0]
        return snapshot(path.read_bytes().decode("utf-8"), {"path": str(path.resolve())},
                        selection="explicit_override"), identity

    if hub is None or download is None:
        from src.infra.huggingface import hf_api, hf_download
        hub = hub or hf_api()
        download = download or hf_download
    git_reader = git_reader or (lambda path, revision: _git_text(path, revision, root))
    if training_meta:
        meta_path = Path(training_meta)
        identity["training_meta_path"] = str(meta_path.resolve())
        identity["revision"] = _pin_target_revision(target, target_revision, hub)
    elif Path(target).is_dir():
        if not target_revision:
            raise ValueError("Local model target needs --target-revision identifying its immutable checkpoint")
        meta_path = Path(target) / "training_meta.json"
    else:
        if "://" in target:
            raise ValueError("Non-Hub target needs --training-meta or --constitution; checkpoint URI is not provenance")
        revision = hub.model_info(target, revision=target_revision).sha
        identity["revision"] = revision
        meta_path = Path(download(target, "training_meta.json", revision=revision))
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    identity["training_meta_sha256"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    source = {"training_target": identity}
    found = _from_record(record, source, git_reader=git_reader)
    dataset = record.get("dataset") or {}
    if found and not dataset:
        return found, identity
    if not dataset.get("repo") or not re.fullmatch(r"[a-fA-F0-9]{40}", str(dataset.get("revision", ""))):
        raise ValueError("No constitution or pinned training dataset in model provenance; pass --constitution")
    candidates = _dataset_candidates(dataset["repo"], dataset["revision"], hub=hub,
                                      download=download, git_reader=git_reader)
    if found:
        candidates.insert(0, found)
    hashes = {row["sha256"] for row in candidates}
    if len(hashes) != 1:
        raise ValueError("Training constitution is missing or ambiguous; pass --constitution explicitly")
    selected = candidates[0]
    selected["source"] = {**selected["source"], "training_target": identity,
                          "corroborating_sources": [item["source"] for item in candidates[1:]]}
    return selected, identity
