<!-- ABOUTME: Current persistent CPU registry, cold preparation and lifecycle contract for SWE-bench Lite. -->
<!-- ABOUTME: This guide supersedes historical expiry and calibration instructions in swebench_cpu_host.md. -->

# Reusable SWE-bench CPU

The user authorized keeping the prepared CPU running until explicitly stopped on
2026-09-24. Persistent means ongoing compute and disk charges; it is not a promise
that a marketplace host never fails. No six-hour campaign or shared GPU-fleet
cutoff applies. Every GPU still has its own finite safety lease and spend admission.

## Find the host from any checkout

`~/.lasr/swebench/cpu.json` is the local shared registry. It points to an absolute
receipt and SSH identity. Receipts contain no credentials. Do not commit local
credentials, keys or a machine-specific registry to git. On another computer,
transfer the receipt securely or reconcile the exact provider identity first.

From the repository root, with the authorized `.env` loaded:

```bash
uv run --env-file .env --project scratch/swebench_cpu_env --frozen python -m scratch.swebench_cpu_manage status
```

The command prints `empty` when no registry exists, otherwise live provider state,
price and SSH endpoint. Missing resources and ownership mismatches fail closed.
It never creates a replacement merely because SSH failed. Direct mapped VM SSH
is preferred to Vast's sometimes unusable proxy endpoint; host-key checking stays on.

Current host: **52229744**, `nika-swebench-cpu-20260923T123716Z`.
Live price verified September 24: **$0.5388889/hour**, about **$12.93/day or
$388/30 days**, excluding transfer. Stopping retains disk at the historical
~$3.33/day; destroying releases disk but requires an explicit destruction request.
Use the management command for current prices/address, not these historical values.

Host resources verified: **61 logical CPUs, 197.9 GiB RAM, 485 GiB filesystem**
from 500 GiB rented. 300 cached images use about 195 GiB. Target a new offer with
**64 effective CPUs, >=200 GiB advertised allocated RAM, 500 GiB SSD**, verify
**>=60 actual CPUs and >=190 GiB actual RAM** inside it, and run capacity proof.
Reserve >=50 GiB disk and >=32 GiB available RAM. Provision a VM, not a container.
80 conversation workers share a **32-command gate**, with 2 CPUs/4 GiB/512 PIDs
per task container. Grading runs separately with 12 workers. This tested gate is
why 80 conversations do not require 320 GiB resident RAM. Do not raise it to 80
simultaneous commands or reduce memory without another CPU-only capacity test.

## Reuse, persistence and stop

- Running: verify provider identity, SSH, local receipt, all cached image digests,
  fixture, CPU/disk guards, current source qualification and HF writes before GPUs.
- Stopped: the existing `--cpu-receipt` launch helper resumes it, refreshes SSH,
  validates the remote receipt and submits. Persistent authorization survives stop;
  an explicit future run authorizes resuming. Do not resume while the user is asking
  only for status or to keep everything stopped.
- Empty: follow cold preparation below. Register only one owned CPU. Do not choose
  another shared-account machine just because it has a matching username prefix.
- Explicit stop: stop the active campaign first, verify zero owned GPUs and durable
  HF results, then use Vast SDK `stop_instance(receipted_id)` and poll actual_status
  until stopped. Keep the receipt and disk. Never confuse SSH loss with stopped billing.
- Empty with no retained disk: only after explicit destruction, verify the exact
  provider instance is absent and atomically write `{"state":"empty"}` to registry.
  Archive the old receipt. This is not necessary to end compute charges.

Register a newly prepared host:

```bash
uv run --env-file .env --project scratch/swebench_cpu_env --frozen python -m scratch.swebench_cpu_manage register --receipt ABSOLUTE_RECEIPT --identity ABSOLUTE_PRIVATE_KEY
```

Persist receipts under `~/.lasr/swebench/`, not a disposable checkout. `persistent`
sets `lifetime: persistent`, `stop_at: null`, and an authorization timestamp in both
receipts, preserving the old receipt. `expire --stop-at ISO_UTC` is the optional
finite-lifetime mode. Both refuse changes during active inference. Deploy the updated
watchdog first; update any other local receipt used by an external expiry task too.
The watchdog still bounds failed initial SSH setup, including persistent hosts.
Run guest and any installed external `--check` afterward. The original Windows
expiry task was removed after persistence was verified; the guest check remains
enabled and does not stop an authorized persistent host. Reinstall an external
backstop if selecting finite expiry later. Null without explicit persistent
authorization is invalid; do not simulate persistence with a far-future date.

## Cold preparation: agent-operated, no GPU rental

The repository skill orchestrates these existing scripts. The eval CLI itself
submits to a prepared host; it does not secretly rent a CPU. A fresh conversation
must carry out these steps when its registry is empty, without requiring old history.

1. Load `.env` securely: VAST_API_KEY, USER_PREFIX, HF_TOKEN, HF_ORG,
   RUNPOD_API_KEY, DOCKERHUB_USERNAME, DOCKERHUB_TOKEN. Check balance and account
   inventory. Verify the local public SSH key is registered before creating a VM.
   Never transmit its private key. The CPU creates its own RunPod SSH key.
2. Use pinned Vast Python SDK 1.8.0. Search fresh verified VM-capable on-demand
   offers with storage=500, filter effective CPUs/RAM/reliability >=0.99 and include
   both transfer prices. Prefer <=$0.75/hour total and <=$0.02/GB each direction;
   report a decision if no qualifying offer exists. Cold pulls can transfer ~300 GB.
   Use the official Ubuntu VM template after verifying it with the provider; the
   previous template hash is in `swebench_cpu_host.md`, not an evergreen guarantee.
3. Before CREATE, persist a unique `<USER_PREFIX>-swebench-cpu-<UTC>-<nonce>` label,
   offer, source commit, intended persistent authorization, and 20-minute boot
   deadline. Call SDK create_instance with fresh offer ID, VM template, disk=500,
   runtype=ssh_direct and label. Save returned instance ID immediately. An ambiguous
   create must be reconciled by that exact unique label before any retry. Acquire
   a local exclusive preparation lock around registry/provisioning to avoid duplicates.
4. Immediately arm an independent external watchdog invoking
   `scratch/swebench_cpu_watchdog.py` with this receipt and authorized env. Windows
   uses the supplied windowless wrapper and pythonw; use a local scheduled service
   on Linux. Its boot deadline stops a failed unverified host. Confirm its execution.
   Bound SSH establishment to 20 minutes. Stop and verify the exact instance on
   failure; do not leave an unprepared persistent VM billing indefinitely.
5. Resolve live direct SSH mapping. Establish the new SSH host fingerprint against
   a trusted provider console/first-provisioning record; never suppress changed-key
   errors. Stage a **committed git archive** to `/srv/lasr/repo`, receipt to
   `/srv/lasr/receipt.json`, and only the required named credentials to root-only
   `/srv/lasr/credentials.env` over SSH stdin. Record deployment SHA because the
   archive has no `.git`. Set `ssh_verified` in both receipts after a real probe.
6. Run `scratch/swebench_cpu_bootstrap.sh` on the VM. Verify native Docker/systemd
   and actual resources first; reject undersized allocation. Authenticate Docker
   with `scratch/swebench_cpu_auth.py` using stdin. Bootstrap starts the durable
   `lasr-swebench-prepare.service`: all 300 pinned images, nine gold tests, six
   no-fix checks, local HTTPBin and HF readback. If it races Docker login, fix login
   and restart preparation; completed pulls are retained. Require service success
   and readiness status ready. No RunPod rentals while waiting for downloads.
7. Install the inference service with `bash scratch/swebench_lite_install.sh`.
   Keep it stopped until an actual model request. Qualify the host and source with
   the commands below. Failed qualification leaves CPU not ready, never bypass
   guards. Register the successful host and show price, SSH and persistent state.

## Long-lived fixture maintenance

Before a later run, the local HTTPBin certificate must have >24 hours remaining.
If expired/near expiry, while no model or grader is active, archive the fixture
metadata/certificate, verify ownership before removing only that fixture container,
and move its cert/key into a root-only archive outside result artifacts. Rerun CPU
preparation to generate a new certificate, repeat gold/no-fix checks and HF readback,
then requalify shell/recipe against the new readiness. Preserve every historical
campaign's copied certificate and manifests. Never disable TLS verification.

## Qualification on a new host or changed implementation

From `/srv/lasr/repo`, under a durable named systemd service if disconnect is possible:

```bash
P=scratch/swebench_cpu_env/.venv/bin/python
$P -m src.eval.capabilities.swebench_mini.fleet qualify-shell --config configs/eval/swebench_mini/lite.yaml
$P -m scratch.swebench_cpu_load --output /srv/lasr/runs/capacity-UNIQUE
$P -m pytest tests/test_swebench_fleet.py tests/test_swebench_session.py scratch/test_swebench_lite.py -q > /srv/lasr/runs/lifecycle-tests-UNIQUE.log
src/eval/capabilities/swebench_mini/envs/agent/.venv/bin/python -m scratch.test_swebench_timeout_transport > /srv/lasr/runs/transport-UNIQUE.log 2>&1
$P -m scratch.swebench_lite_smoke
# Use the smoke root printed by the preceding command:
$P -m scratch.swebench_qualify_recipe --load-dir /srv/lasr/runs/capacity-UNIQUE --smoke-root SMOKE_ROOT --test-log /srv/lasr/runs/lifecycle-tests-UNIQUE.log --transport-log /srv/lasr/runs/transport-UNIQUE.log
```

Use fresh unique evidence directories. No dependency on the original control's
local results or old hardening logs remains. This tests real Docker workloads,
real agent with synthetic endpoint, official grading and HF readback without a GPU.
The recipe records current source hashes; it does not claim paid speed qualification.
For an unchanged host/resource protocol, reuse verified capacity evidence after
review; rerun regression and smoke tests for implementation changes. Never requalify
by changing hashes alone. Record source transition and regression evidence on HF.

The first next model run is the end-to-end paid qualification. Marketplace capacity,
provider failures, credentials, funding and persistent HF outages remain possible
external blockers; progress monitoring must report them. Persistent CPU storage is
not a backup: keep canonical outcomes and infrastructure evidence on Hugging Face.
