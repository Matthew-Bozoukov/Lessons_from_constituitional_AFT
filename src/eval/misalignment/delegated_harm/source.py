# ABOUTME: Load an immutable external benchmark checkout with two recorded path fixes.
# ABOUTME: Missing worlds are explicit exclusions; source assets stay outside this repo.
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: sorted(x)
                              if isinstance(x, set) else str(x)), encoding="utf-8")
    tmp.replace(path)


def prepare(cfg):
    root = Path(str(cfg.source.path)).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-c", "core.longpaths=true", "-C", str(root),
                                        *args], text=True, encoding="utf-8").strip()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--no-checkout",
                        str(cfg.source.repo), str(root)], check=True)
        subprocess.run(["git", "-c", "core.longpaths=true", "-C", str(root), "checkout",
                        "--detach", str(cfg.source.revision)], check=True)
    if git("rev-parse", "HEAD") != cfg.source.revision:
        raise ValueError("Upstream checkout revision differs from the configured pin")
    patches = {
        "scenarios/policy_threshold/scenario.py":
            ('str(p.relative_to(FIX))', 'p.relative_to(FIX).as_posix()'),
        "scenarios/tier_screen_off/scenario.py":
            ('str(p.relative_to(root))', 'p.relative_to(root).as_posix()'),
    }
    changed = set(git("diff", "--name-only", "HEAD").splitlines())
    if changed - patches.keys():
        raise ValueError(f"Unrecorded upstream edits: {sorted(changed - patches.keys())}")
    for name, (old, new) in patches.items():
        original = git("show", f"HEAD:{name}")
        patched = original.replace(old, new)
        actual = (root / name).read_text(encoding="utf-8").strip()
        if actual not in (original, patched):
            raise ValueError(f"Unrecognized upstream patch in {name}")
        if old not in original:
            raise ValueError(f"Expected patch target missing: {name}")
        if actual != patched:
            (root / name).write_text(patched + "\n", encoding="utf-8")
    sys.path.insert(0, str(root))
    # Authoring imports the upstream runner, which itself imports both provider SDKs.
    # Check that dependency chain before a caller rents or starts a GPU.
    from core import registry, reskin, escaldet
    known = set(registry.keys())
    selected = set(cfg.scenarios)
    excluded = set(cfg.source.excluded)
    if selected & excluded or selected | excluded != known:
        raise ValueError(f"Scenario coverage must be explicit: known={known}, selected={selected}")
    scenarios = {key: registry.load(key) for key in cfg.scenarios}
    human = json.loads((root / "briefs/human.json").read_text(encoding="utf-8"))
    notes = json.loads((root / "briefs/sources.json").read_text(encoding="utf-8"))
    from core.env import HUMAN_APPROVER, HUMAN_REQUESTERS
    for key, sc in scenarios.items():
        if key not in human or not notes[key].get("clear"):
            raise ValueError(f"Missing human request or source notes: {key}")
        assert HUMAN_APPROVER[key] in sc.humans
        assert HUMAN_REQUESTERS[key] in sc.humans
        assert HUMAN_APPROVER[key] != HUMAN_REQUESTERS[key]
    manifest = {"repository": str(cfg.source.repo), "revision": str(cfg.source.revision),
                "patches": {k: digest((root/k).read_text(encoding="utf-8")) for k in patches},
                "excluded": dict(cfg.source.excluded),
                "worlds": {k: digest(sc.files) for k, sc in scenarios.items()},
                "human_requests_sha256": digest({k: human[k] for k in scenarios}),
                "source_notes_sha256": digest({k: notes[k] for k in scenarios})}
    return scenarios, human, notes, manifest
