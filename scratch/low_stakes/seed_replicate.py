# ABOUTME: One entry point for a seed replicate: derive the config, rebuild seed 0's bundle
# ABOUTME: with it, train on RunPod, publish the adapter. Seed and source are arguments.

"""Re-train an existing arm under a different seed, changing nothing else.

    uv run python scratch/low_stakes/seed_replicate.py bundle --seed 80085
    uv run python scratch/low_stakes/seed_replicate.py train  --seed 80085 --bundle <repo>
    uv run python scratch/low_stakes/seed_replicate.py land   --seed 80085

WHY A SCRIPT AND NOT A CONFIG PER SEED. The prior seed work (gptresp685, seeds 42 and 69)
hand-wrote one config per seed. The diff between each and its parent is exactly three lines
-- `seed`, `output_dir`, `hf_repo` -- so a copy per seed is three lines of intent buried in
forty of duplication, and every later edit to the parent has to be replayed into each
replicate by hand. Here the config is DERIVED from the parent at build time: the parent
stays the single source of truth, the seed is an argument, and `bundle` REFUSES to publish
if anything other than those three keys differs. A replicate cannot silently drift from the
arm it replicates.

THE RULE THIS ENFORCES, quoting the gptresp bundle card: "Seed replicates must train on
byte-identical code and data to seed 0." So this deliberately does NOT use
scratch/publish_train_bundle.py, which tars the WORKING TREE. It pulls seed 0's own
`code.tar.gz` from the bundle its adapter stamp points at, checks the hash, and appends only
the derived config. The mixture is re-uploaded unchanged (sha asserted) because
scripts/gpu/runpod_train.py reads code and data from ONE repo.

WHAT A SEED ACTUALLY CHANGES. `train_lora.py` calls `torch.manual_seed(cfg.seed)` before the
model is built, so it drives LoRA initialisation; the DDP dataloader's generator is seeded
from the same value, so data ORDER changes too. A seed replicate is init AND order, not init
alone -- say so in any write-up rather than calling it an initialisation replicate.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import fire
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

# The arm being replicated. Everything else -- the mixture, its revision, the code -- is read
# off this adapter's own training_meta.json, so pointing at a different arm is a one-line
# change rather than a new script.
SEED0_ADAPTER = "LASR-Callum/2026-08-26-qwen36-lora-table2-9284-low-stakes-716-rank-64-dynbatch"
SEED0_CONFIG = "configs/train/lora_qwen36_t2_9284_lowstakes716_dynbatch_2xh200.yaml"

# FIVE keys, not the three the seed itself needs. `data_repo`/`data_revision` are here
# because scripts/gpu/runpod_train.py rewrites `data_repo` and `data_file` to point at the
# BUNDLE, and leaves `data_revision` alone -- so a config inheriting the parent's pin sends
# the trainer looking for a revision of the OLD repo inside the new bundle, and
# `resolve_dataset` fails. The gptresp seed configs sidestepped this by carrying no
# `data_revision` at all, which works but leaves the run resolving to whatever HEAD happens
# to be. Pinning to the bundle's own sha is strictly better provenance and closes the gap
# seed 0 has, whose stamp pins a revision that predates its own code.tar.gz.
OVERRIDE_KEYS = ("seed", "output_dir", "hf_repo", "data_repo", "data_revision")


def _api() -> HfApi:
    import os
    return HfApi(token=os.environ["HF_TOKEN"])


def _stamp() -> dict:
    """Seed 0's training_meta.json -- the record of what that arm actually trained on."""
    p = hf_hub_download(SEED0_ADAPTER, "training_meta.json", token=_api().token)
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _derive_config(seed: int, parent_text: str, bundle_repo: str,
                   bundle_rev: str) -> tuple[str, dict]:
    """Parent config with exactly the OVERRIDE_KEYS replaced, everything else identical.

    Edits TEXT rather than round-tripping through yaml so the parent's comments survive into
    the tarball. A replicate whose config lost the parent's reasoning would be worse
    documentation than the parent it came from.
    """
    parent = yaml.safe_load(parent_text)
    changed = {
        "seed": seed,
        "output_dir": parent["output_dir"].rstrip("/") + f"_s{seed}",
        "hf_repo": parent["hf_repo"].rstrip("/") + f"-seed{seed}",
        "data_repo": bundle_repo,
        "data_revision": bundle_rev,
    }
    out = parent_text
    for key, val in changed.items():
        lines, done = [], False
        for line in out.splitlines():
            if not done and line.startswith(key + ":"):
                q = '"' if isinstance(val, str) else ""
                lines.append(f"{key}: {q}{val}{q}")
                done = True
            else:
                lines.append(line)
        assert done, f"parent config has no top-level `{key}:` to override"
        out = "\n".join(lines) + "\n"
    header = (
        f"# SEED REPLICATE {seed}. DERIVED by scratch/low_stakes/seed_replicate.py from\n"
        f"# {SEED0_CONFIG} -- do not edit by hand. Only seed, output_dir and hf_repo differ\n"
        f"# from the parent, and regenerating is what keeps that true.\n")
    return header + out, changed


def _check_only_expected_changed(parent_text: str, derived_text: str, n_expected: int) -> int:
    """Refuse a derived config that moved anything but the override keys."""
    a = [x for x in parent_text.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    b = [x for x in derived_text.splitlines() if x.strip() and not x.lstrip().startswith("#")]
    assert len(a) == len(b), (
        f"derived config changed the config's SHAPE ({len(a)} -> {len(b)} lines), not just "
        f"values -- refusing to publish")
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == n_expected, (
        f"derived config differs from its parent in {len(diff)} lines, expected "
        f"{n_expected}:\n" + "\n".join(f"  - {x}\n  + {y}" for x, y in diff))
    return len(diff)


def bundle(seed: int, repo: str = "", dry: bool = False) -> None:
    """Publish a training bundle: seed 0's code, seed 0's data, the derived config."""
    api = _api()
    st = _stamp()
    ds = st["dataset"]["repo"]
    dfile = st["dataset"]["file"]
    drev = st["dataset"]["revision"]
    print(f">>> replicating {SEED0_ADAPTER}")
    print(f"    data pin: {ds}@{drev[:12]} :: {dfile}")

    code_p = Path(hf_hub_download(ds, "code.tar.gz", repo_type="dataset", token=api.token))
    code_sha = hashlib.sha256(code_p.read_bytes()).hexdigest()
    mix_p = Path(hf_hub_download(ds, dfile, repo_type="dataset", revision=drev,
                                 token=api.token))
    mix_sha = hashlib.sha256(mix_p.read_bytes()).hexdigest()
    print(f"    code.tar.gz sha256 {code_sha[:16]}    mixture sha256 {mix_sha[:16]}")

    with tarfile.open(code_p) as tf:
        members = tf.getnames()
        assert SEED0_CONFIG in members, (
            f"seed 0's tarball has no {SEED0_CONFIG}; this is not the code that arm ran")
        parent_text = tf.extractfile(SEED0_CONFIG).read().decode("utf-8")

    repo = repo or f"LASR-Callum/2026-08-31-lowstakes716-seed{seed}-bundle"
    if dry:
        # No bundle revision exists yet in a dry run; show the shape with a placeholder.
        bundle_rev = "<resolved after the mixture upload>"
    else:
        # The MIXTURE goes up first, so its commit is the revision the config can pin. The
        # code tarball follows and carries that pin inside it -- pinning to a sha that does
        # not exist yet is the one ordering this cannot get wrong.
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(mix_p), path_in_repo=dfile,
                        repo_id=repo, repo_type="dataset")
        bundle_rev = api.dataset_info(repo).sha
        print(f">>> mixture uploaded; bundle revision {bundle_rev[:12]}")

    derived_text, changed = _derive_config(seed, parent_text, repo, bundle_rev)
    n = _check_only_expected_changed(parent_text, derived_text, len(OVERRIDE_KEYS))
    cfg_path = SEED0_CONFIG.replace(".yaml", f"_s{seed}.yaml")
    print(f">>> derived {cfg_path}; verified exactly {n} non-comment lines differ:")
    for k, v in changed.items():
        print(f"      {k}: {v}")

    out_tar = ROOT / "output" / f"seed{seed}_code.tar.gz"
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(code_p) as src, tarfile.open(out_tar, "w:gz") as dst:
        for m in src.getmembers():
            f = src.extractfile(m)
            if f is None:
                dst.addfile(m)
            else:
                dst.addfile(m, f)
        data = derived_text.encode("utf-8")
        info = tarfile.TarInfo(cfg_path)
        info.size = len(data)
        info.mode = 0o644
        dst.addfile(info, io.BytesIO(data))
    print(f">>> wrote {out_tar.name}: {len(members) + 1} files, "
          f"{out_tar.stat().st_size / 1e6:.2f} MB")

    if dry:
        print(f"--dry: would publish to {repo}")
        return
    api.upload_file(path_or_fileobj=str(out_tar), path_in_repo="code.tar.gz",
                    repo_id=repo, repo_type="dataset")
    card_p = ROOT / "output" / f"seed{seed}_README.md"
    card_p.write_text(_card(seed, repo, ds, drev, dfile, code_sha, mix_sha, cfg_path),
                      encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card_p), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    sha = api.dataset_info(repo).sha
    print(f">>> published https://huggingface.co/datasets/{repo} @ {sha[:12]}")
    print(f">>> next: uv run python scratch/low_stakes/seed_replicate.py train "
          f"--seed {seed} --bundle {repo}")


def _card(seed, repo, ds, drev, dfile, code_sha, mix_sha, cfg_path) -> str:
    return f"""---
tags:
- training-bundle
- seed-replicate
---
# Low-stakes difficult advice: seed-{seed} training bundle

- **experiment**: Seed replicate of `{SEED0_ADAPTER}` (seed 0). Same data, same code, same
  hyperparameters. Only the seed differs, which changes LoRA initialisation AND data order.
- **date_generated**: 2026-08-31 (bundle); mixture generated 2026-08-26.
- **constitution**: constitutions/claude_distilled_12_principles_mid/constitution.md, via
  `LASR-Callum/2026-08-26-difficult-advice-low-stakes-716`.
- **source_repo**: this repository. `code.tar.gz` is seed 0's own tarball, sha256
  `{code_sha}`, with `{cfg_path}` appended and nothing else altered.
- **models**: base `Qwen/Qwen3.6-27B`.
- **generation_config**: seed {seed}; every other hyperparameter identical to seed 0 --
  r=64, alpha=128, lr 1e-4 cosine, 1 epoch, global batch 16, dynamic batching,
  max_seq_len 8192, thinking=true.
- **schema**: `code.tar.gz` = seed 0's code plus the derived seed config;
  `{dfile}` = the mixture from `{ds}`@`{drev}`, byte-identical (sha256 `{mix_sha}`), so the
  trainer's data pin resolves inside this repo.
- **provenance**: `uv run python scratch/low_stakes/seed_replicate.py bundle --seed {seed}`
  then `... train --seed {seed} --bundle {repo}`.

The seed config is DERIVED from the parent, not copied, and the build refuses to publish if
anything but `seed`, `output_dir` and `hf_repo` differs from it.
"""


def train(seed: int, bundle: str, gpu: str = "NVIDIA H200", gpu_count: int = 2,
          disk_gb: int = 250) -> None:
    """Launch the pod through the existing runpod_train driver, then name the next steps."""
    st = _stamp()
    cfg_path = SEED0_CONFIG.replace(".yaml", f"_s{seed}.yaml")
    cmd = ["uv", "run", "python", "scripts/gpu/runpod_train.py", "up",
           "--bundle", bundle, "--train_config", cfg_path,
           "--mixture", st["dataset"]["file"], "--gpu", gpu,
           "--gpu_count", str(gpu_count), "--disk_gb", str(disk_gb),
           "--name", f"nika-lowstakes-seed{seed}"]
    print(">>> " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-1500:])
    pod = next((l.split()[1] for l in r.stdout.splitlines() if l.startswith("pod:")), "")
    if pod:
        out = f"output/low_stakes_seed{seed}_adapter"
        print(f"\n>>> arm the watchdog NOW; it pulls the adapter BEFORE any teardown:\n"
              f"    uv run --with requests python scratch/low_stakes/train_watchdog.py "
              f"--pod {pod} --out_dir {out}\n"
              f">>> then: uv run python scratch/low_stakes/seed_replicate.py land "
              f"--seed {seed}")


def land(seed: int, adapter_dir: str = "") -> None:
    """Publish the adapter the watchdog pulled. The watchdog owns pulling and teardown."""
    api = _api()
    d = Path(adapter_dir or ROOT / "output" / f"low_stakes_seed{seed}_adapter")
    found = sorted(d.glob("**/adapter/adapter_model.safetensors"))
    assert found, (
        f"no adapter under {d}. The watchdog pulls adapter.tar.gz; extract it first:\n"
        f"    tar -xzf {d}/adapter.tar.gz -C {d}")
    src = found[-1].parent
    repo = f"{SEED0_ADAPTER}-seed{seed}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    for f in sorted(src.iterdir()):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=repo, repo_type="model")
        print(f"    {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    rm = src.parent / "run_meta.json"
    if rm.exists():
        api.upload_file(path_or_fileobj=str(rm), path_in_repo="run_meta.json",
                        repo_id=repo, repo_type="model")
        print("    run_meta.json")
    print(f">>> https://huggingface.co/{repo}")


if __name__ == "__main__":
    fire.Fire({"bundle": bundle, "train": train, "land": land})
