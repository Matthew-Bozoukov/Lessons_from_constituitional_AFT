# ABOUTME: Build the code.tar.gz that scripts/gpu/runpod_train.py extracts on the pod, VERIFY
# ABOUTME: it imports standalone, and upload it beside a mixture already in an HF dataset repo.

"""Publish the training code bundle for a RunPod training arm.

Run: uv run python scratch/publish_train_bundle.py --repo <hf dataset repo> \\
         --train_config configs/train/<arm>.yaml

The pod carries no credentials and no git checkout: it pulls `code.tar.gz` from a public HF
dataset repo, extracts it, and runs `scripts/train/train_lora.py` against a config INSIDE
that tarball. So the tarball has to be a complete, self-contained import root.

Why this exists rather than reusing scratch/publish_selfreflect_bundle.py: that script's
`CODE` list predates dynamic batching and omits `src/train/dynamic_batching.py`,
`src/model_profile.py` and `src/huggingface.py`. A missing module there does not fail
locally -- it fails ~25 minutes into a paid pod's boot, after the base-model download. So
the list here is derived from the trainer's real import graph AND checked by extracting the
tarball into a scratch directory and importing the trainer from it with nothing else on the
path.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import fire
from dotenv import load_dotenv

# The trainer's transitive first-party import graph, plus the packaging metadata the pod's
# pip step reads. Verified by _verify_bundle below -- do not trim it by eye.
CODE = [
    "pyproject.toml",
    "scripts/train/train_lora.py",
    "src/__init__.py",
    "src/utils.py",
    "src/model_profile.py",
    "src/huggingface.py",
    "src/train/__init__.py",
    "src/train/train_lora.py",
    "src/train/masking.py",
    "src/train/mask_gate.py",
    "src/train/dynamic_batching.py",
]


def _verify_bundle(tar_path: Path, train_config: str) -> None:
    """Extract the tarball somewhere clean and import the trainer out of it.

    Catches the failure this script exists to prevent: a module the trainer imports that
    nobody remembered to add to CODE.

    This is a STATIC import closure, not an import. Two reasons it has to be:
      - The GPU stack (peft, trl, the training transformers) is linux-marked in the lock,
        so it is not installed on a macOS driver machine and `import src.train.train_lora`
        cannot run here at all.
      - The trainer imports some first-party modules LAZILY inside functions
        (`from src.huggingface import resolve_dataset` mid-main). A check that merely
        loaded the module would never touch those, and would pass a tarball that dies an
        hour into training.

    So: parse every bundled file, walk `src.*` and relative imports to a fixpoint, and
    require every module reached to be present in the tarball.
    """
    import ast

    repo_root = Path(__file__).resolve().parents[1]

    def resolve(mod: str) -> str | None:
        """Map a dotted first-party module to the repo file that would satisfy it."""
        stem = mod.replace(".", "/")
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if (repo_root / cand).exists():
                return cand
        return None

    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        with tarfile.open(tar_path) as t:
            t.extractall(root)
        missing = [f for f in (*CODE, train_config) if not (root / f).exists()]
        assert not missing, f"tarball is missing {missing}"

        seen: set[str] = set()
        queue = [f for f in (*CODE, train_config) if f.endswith(".py")]
        needed: set[str] = set(queue)
        while queue:
            rel = queue.pop()
            if rel in seen:
                continue
            seen.add(rel)
            src_file = root / rel
            if not src_file.exists():
                continue
            pkg_parts = rel.rsplit("/", 1)[0].split("/")
            for node in ast.walk(ast.parse(src_file.read_text(encoding="utf-8"))):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative: resolve against this file's package
                        base = ".".join(pkg_parts[:len(pkg_parts) - node.level + 1])
                        head = f"{base}.{node.module}" if node.module else base
                    else:
                        head = node.module or ""
                    if head:
                        mods = [head] + [f"{head}.{a.name}" for a in node.names]
                for mod in mods:
                    if not mod.startswith("src"):
                        continue
                    target = resolve(mod)
                    if target and target not in needed:
                        needed.add(target)
                        queue.append(target)

        absent = sorted(f for f in needed if not (root / f).exists())
        if absent:
            raise SystemExit(
                "bundle is INCOMPLETE -- the trainer's import graph reaches these, but they "
                "are not in the tarball:\n  " + "\n  ".join(absent) +
                "\nAdd them to CODE and re-run.")
        print(f"  import closure OK: {len(needed)} first-party modules, all present")


def main(repo: str, train_config: str, extra: str = "") -> None:
    """Build, verify and upload code.tar.gz into an existing HF dataset repo.

    Args:
        repo: The public HF dataset repo that already holds the mixture jsonl. The pod is
            pointed at this same repo for both the code and the data.
        train_config: Config path to include in the tarball; the pod passes it to the
            trainer verbatim, so it must be a path RELATIVE to the repo root.
        extra: Optional comma-separated extra paths to include.
    """
    load_dotenv()
    from huggingface_hub import HfApi

    files = [*CODE, train_config, *[e.strip() for e in extra.split(",") if e.strip()]]
    missing = [f for f in files if not Path(f).exists()]
    assert not missing, f"not in the working tree: {missing}"

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"] + files, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        print(f"!! bundling UNCOMMITTED changes; the pod will run code that is not at {sha}:")
        print(dirty)

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "code.tar.gz"
        with tarfile.open(tar_path, "w:gz") as t:
            for f in files:
                t.add(f, arcname=f)
        print(f"  {len(files)} files, {tar_path.stat().st_size / 1e6:.2f} MB")
        _verify_bundle(tar_path, train_config)

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(tar_path), path_in_repo="code.tar.gz",
                        repo_id=repo, repo_type="dataset")
        digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()[:16]

    print(f"\ncode.tar.gz -> https://huggingface.co/datasets/{repo}")
    print(f"  git sha {sha}   tar sha256 {digest}")
    print(f"  train_config inside the tarball: {train_config}")


if __name__ == "__main__":
    fire.Fire(main)
