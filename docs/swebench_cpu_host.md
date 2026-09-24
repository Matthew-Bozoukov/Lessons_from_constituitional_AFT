<!-- ABOUTME: Reproducible Vast Linux VM setup and CPU-readiness gate for RunPod SWE-bench. -->
<!-- ABOUTME: Read this when asked to create or resume a SWE-bench CPU host; preserve shared resources. -->

# SWE-bench CPU host on Vast

## Current instructions

Use [CPU lifecycle](swebench_cpu_lifecycle.md) and the [repository skill](../.agents/skills/swebench-lite/SKILL.md).
The user authorized persistent CPU retention on September 24. The old 24-hour
expiry, eight-GPU calibration and incomplete implementation notes below are
historical evidence, not instructions for a new run.

## Historical preparation decision and status (2026-09-23)

Run the full 300-task SWE-bench Lite split on ONE checkpoint at a time. The first
checkpoint is `dougalldeepmind/2026-09-22-qwen36-0-nosynth`, revision
`633908b72a9799fb3e6b101b0a8a82aec3c3d642`. Base revision:
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

User authorized a dedicated Vast CPU/Docker host and chose a **24-hour expiry**.
Interpret expiry as provider STOP, retaining the instance disk; do not destroy the
disk without a later instruction. Explicitly report ongoing storage charges.
No RunPod inference GPU may be rented until the CPU readiness gate below passes.
Aim to scale inference to eight independent single-GPU replicas after calibration.

After the user funded Vast and asked to proceed, VM **52229744** was created as
`nika-swebench-cpu-20260923T123716Z`. SSH and Docker work. The guest has **61 vCPUs,
197 GiB RAM, 485 GiB filesystem** (477 GiB initially free), less RAM than the offer
advertised but above the required floor. Docker browser device login succeeded after
the user purchased Pro, clearing the anonymous download block. The user's read-only
Docker token was subsequently verified and installed without logging its value.
All **300/300 images are cached**, using about 195 GiB of disk with about 290 GiB
free. Public-httpbin gold checks alternated 3/4, 4/4, 3/4 due to HTTP 502 failures.
A pinned local HTTP/HTTPS fixture now passes **9/9 gold checks**, covering every
requests task plus Django, SymPy and scikit-learn. This is a declared environment
deviation, detailed below. Six no-fix checks found two upstream scoring quirks;
both also pass with the unmodified grader and public service. No tasks are dropped.
The final committed preparation passed at **2026-09-23 13:38:45 UTC**: 9/9 gold,
the expected 2/6 no-fix baseline, 300 cached images, Docker/network/volume checks,
and byte-for-byte HF backup verification. This warm readiness run took about two
minutes. CPU status is **ready**; the inference coordinator still needs the fixes
listed below before any RunPod rental. Verified infrastructure backup commit:
`347fcb010e6cf0df5dcb6e914537d90a89679b7a`.

Connect from this Windows account:

```powershell
ssh -i "$HOME/.ssh/msm_audit" -p 47579 root@184.144.154.180
```

Refresh IP/port from Vast before reconnecting after a restart. SSH is key-only.
The absolute STOP deadline is **2026-09-24 12:37:16 UTC (13:37 London)**. This stops
compute and retains the disk; storage still costs approximately **$3.33/day**.
External Windows task `nika-swebench-cpu-52229744-expiry` checks every five minutes;
guest `lasr-vast-expiry.timer` checks every minute. Both have completed live checks.
The provider scheduler rejected requests (SDK float timestamp and then endpoint
validation errors); **there is no provider-scheduled expiry**. Guest timer requires
a working guest/network; the external task requires this PC to be running.

HF round-trip upload/download checksum passed. Readiness artifacts:
https://huggingface.co/datasets/dougalldeepmind/2026-09-23-swebench-cpu-readiness-52229744
These are infrastructure checks, not model evaluation scores.

Use the isolated `codex/swebench-cheap` worktree, based on main `4e13cb71`. Do not
change the user's parallel checkout, Docker configuration, or existing rentals.

## What to rent

- On-demand, verified **VM-capable** host (`vms_enabled=true`), not a normal Vast
  container. Docker must run natively inside a real VM with systemd.
- At least 32 effective vCPUs, 128 GB allocated RAM, and **500 GiB rented disk**.
  Marketplace disk_space is availability, not the disk size actually rented.
- Prefer reliability >= 0.99, fast SSD and at least 1 Gbit/s measured download.
  Check ingress/egress prices as well as hourly compute and disk. Pulling hundreds
  of GB can make a slightly cheaper hourly offer more expensive overall.
- Re-query before creation. Prices and offer IDs below are a dated snapshot.
- Prefix from the authorized repo-root `.env`: USER_PREFIX, currently `nika`.
  Name the host `<prefix>-swebench-cpu-<UTC date/time>`. Record its exact ID and
  machine ID; a shared prefix by itself is not proof that this run owns it.

Snapshot of the cheapest qualifying offer: offer 51160799, machine 37526, Quebec,
64 effective vCPUs, 257,797 MB advertised RAM, 500 GiB disk priced in the query,
reliability 0.9988358. It bundles two RTX 4060 Ti GPUs, unused for this CPU role.
Compute $0.40/hour + disk $0.1388889/hour = **$0.5388889/hour**. Both bandwidth
directions are $0.01953125/GB in the offer. About $12.93 for 24h plus transfers;
300 GB downloaded would add about $5.86. Stopped disk is about $3.33/day.
These are offer arithmetic, not an invoice or a promise of available capacity.

Existing stopped instance `50075064` / `sn-rewardhack` was observed during the first
audit, but was absent at provisioning. We did not change it. Never operate on
unrelated resources; only the exact receipted VM ID belongs to this task.

## Tools and credentials

Use `VAST_API_KEY`, `HF_TOKEN`, `HF_ORG`, USER_PREFIX and, only when the GPU gate
passes, RUNPOD_API_KEY from the authorized `.env`. Never print values or put them
on a command line. In a worktree, pass the original .env path explicitly rather
than reading another project's credentials or copying every secret to the server.

Use the **Vast Python SDK, pinned to 1.8.0**. The existing helper
`scratch/verbose_cot/vast_boxes_sdk.py` documents a Windows application-control
failure with the CLI executable. Importing the SDK works on this machine.
`scratch/odcv_vast_boxes.py` is useful reference code, but its old `up` command
does NOT implement the expiry, readiness gate, or receipt requirements here.
Do not run that command blindly and assume this runbook has been enforced.

Read-only SDK discovery, with the key loaded by python-dotenv:

```python
client = VastAI(api_key=dotenv_values(env_file)["VAST_API_KEY"].strip())
account = client.show_user()  # print only balance/credit, not the full object
offers = client.search_offers(
    query="vms_enabled=true rentable=true disk_space>=500",
    type="on-demand", storage=500, order="dph_total", limit=500,
)
```

Filter allocated cpu_cores_effective and cpu_ram client-side. The old helper
documents a query-language issue combining those filters. Include storage=500 in
pricing. Print an allowlist of offer fields; API objects may contain credentials.

The local public key `~/.ssh/msm_audit.pub` is registered on Vast as key 1181198,
verified 2026-09-23. `id_ed25519.pub` was not registered. Recheck before creation:
Vast's VM docs say account SSH keys must exist at creation. Never upload the local
private key to the VM. A separate CPU-to-RunPod key is generated on the CPU host.

## Provision and establish expiry

1. Check account balance and unrelated existing resources. Save a sanitized offer
   receipt, intended label, code SHA, creation nonce and UTC deadline locally BEFORE
   making the request. Check affordability of 24h plus transfers. Do not top up the
   account automatically unless separately authorized.
2. Use the official Ubuntu 22.04 VM template. The historically verified template
   hash is `b7942f6bbc4374893ff66eb78145bbac`; recheck it against the official VM
   template before use. SDK create fields are `id=<fresh_offer_id>`,
   `template_hash=<verified_hash>`, `disk=500`, `label=<unique_label>`,
   `runtype="ssh_direct"`. Do not put secrets in template environment fields.
3. A timeout is an ambiguous CREATE, not proof nothing was rented. Search the
   account for the unique creation label and reconcile before any retry.
4. Persist the returned instance ID immediately. Arm expiry before bootstrap work.
   Prefer a provider-scheduled STOP independent of the laptop and guest. The 1.8.0
   SDK exposes `create_scheduled_job`; inspect/verify its actual behavior before
   claiming a one-shot deadline (its documented frequencies are hourly/daily/weekly).
   Save the job ID and read it back. Do not confuse a scheduled guest command with
   a provider STOP, or a guest shutdown with confirmed stopped billing.
5. Also install an absolute-UTC systemd timer in the guest that issues STOP through
   the provider API with minimum needed credentials, and a separate external
   watchdog that checks the recorded instance ID and retries failed STOP requests.
   A guest-only timer cannot enforce a deadline if the host disappears. A local
   watchdog cannot enforce it while the laptop sleeps. Document the actual layers
   installed and their limitations; do not promise unconditional enforcement.
6. Bound startup to 20 minutes, polling fresh provider state. On failure, stop this
   exact owned instance and verify provider state before replacing it. No endless
   paid bootstrap retries and no broad account cleanup.
7. Resolve the fresh public IP and port from provider state each connection. The
   old SWE postmortem records stale ssh-url output after an IP remap. Test the API
   endpoint and proxy endpoint if needed; never disable SSH host-key verification
   globally to get past a changed host.

Connection handoff after a successful SSH probe:

```text
ssh -i C:\Users\nikak\.ssh\msm_audit -p <fresh-port> root@<fresh-IP>
```

Optionally add a dedicated `nika-swebench-cpu` SSH Host block without rewriting
other entries. Record how to refresh it after an address change. Supply instance
ID, provider link, hourly cost, deadline, working directory and this command to
the user. Until that probe succeeds, there is no working connection to hand over.

## Bootstrap the Linux host

Keep the repository at `/srv/lasr/repo`, persistent run state at `/srv/lasr/runs`,
and cached benchmark data at `/srv/lasr/cache`. All must be on the retained disk.
Record `findmnt`, `df`, `nproc`, MemTotal, OS version and Docker's data-root.
Verify inside-guest allocated resources rather than trusting the marketplace.

Install/verify git, OpenSSH client, rsync, tmux, Python 3.12, uv, Docker Engine and
Compose. The official VM template includes Docker; prefer verifying it to replacing
a working installation. Enable Docker with systemd. Use a named durable systemd
service for preparation and evaluation so disconnecting SSH does not kill work.
Do not expose an unauthenticated Docker socket or model endpoint to the Internet.

Transfer a pinned git checkout or archive from the isolated branch. Never deploy
the parallel checkout's working tree. Preserve a git revision receipt if using an
archive, since git-based run metadata otherwise cannot identify its source.

**Do not blindly `uv sync` the main Linux project:** its Linux markers install
vLLM, CUDA libraries and training kernels, including a causal-conv1d build. This
machine only needs the CPU driver. A small locked environment is supplied in
`scratch/swebench_cpu_env/`. Its direct shared-dependency versions match the
audited main uv.lock. Windows import checks passed for the eval driver, RunPod
client and SWE runner without torch or vLLM installed. Linux deployment must still
be verified on the new host. Run from the repository root:

```bash
uv sync --frozen --project scratch/swebench_cpu_env
uv run --frozen --project scratch/swebench_cpu_env python -m src.eval.run_eval --help
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/agent
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/harness
```

Use `uv run --frozen --project scratch/swebench_cpu_env python -m
src.infra.runpod` only for that module's watchdog interface. Its ordinary up/down
CLI is `cli()` (a Fire dispatcher), exposed by the repository's thin wrapper:
set PYTHONPATH to the repository root and run `scripts/infra/runpod.py` using this
environment. The CPU environment deliberately does not install the main project
or its console aliases. Nested agent and harness environments retain their own
committed locks and are invoked by the existing runner.

Set thread limits for BLAS/test subprocesses where appropriate so dozens of
containers do not each claim every host core. Begin with 24 active agent tasks and
8 grading workers, then increase only after measuring peak RAM, CPU and disk I/O.
The eight-GPU plan is conditional on this CPU host sustaining the command workload.

## Mandatory CPU readiness gate: zero RunPod rentals before PASS

The launcher must consume a fresh machine-readable receipt bound to host ID, code,
dataset and protocol hashes; a manually written READY file is insufficient.

1. **Resources and persistence:** >=32 effective vCPUs, >=128 GB allocated RAM,
   500 GiB requested disk, sufficient actual free space after images. Verify a
   persistent marker and systemd service survive disconnect/reconnect. Protect
   the run-state filesystem from image-cache exhaustion; reserve >=50 GiB free.
2. **Docker:** daemon, Compose, bridge networks, volume writes and container
   execution work. Run the repository docker_preflight. Execute a real shell
   command inside a SWE image. Capability checks must not alter other workloads.
3. **Pinned data:** stage all 300 Lite test rows from
   `princeton-nlp/SWE-bench_Lite@6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`.
   Save the full snapshot, IDs and SHA256 manifest. Rollout and grading must both
   consume those bytes; current code records the SHA without enforcing it in
   both subprocesses (see the audit). Fix that before the paid rollout.
4. **Images:** pre-pull all 300 task images; record digests and all failures.
   Download against bounded parallelism and disk-space checks. Do not rent GPUs
   while images are still downloading. Keep the cache on the retained disk.
5. **Grading:** run the official pinned harness with gold patches on a small,
   fixed selection spanning multiple repositories. Require every gold case to
   resolve. The existing `swebench_mini_check_env.py` is the starting point; its
   default is Verified, so explicitly use Lite and the pinned snapshot.
   The deployed preparation now checks nine gold patches plus six no-fix requests
   cases using the documented local HTTPBin protocol below. Verify the expected
   baseline outcomes as well as the gold successes.
6. **Durability:** upload a small real preparation artifact through the HF contract
   (rollouts/results/metadata + card/tags), read it back and verify its hash. Save
   dataset/image/lock manifests and the CPU receipt off-host. HF_ORG from .env is
   currently dougalldeepmind. Do not hardcode the guide's older org wording.
7. **Supervision:** verify the preparation/driver service, upload worker and expiry
   enforcement survive SSH disconnection. Confirm the CPU expiry is later than
   every proposed GPU deadline. Failure of any required check means NOT READY.

## Connect to RunPod after CPU readiness

Generate a dedicated SSH key on the CPU VM. Register only its public key for the
RunPod instances this run will own. Keep its private key on the CPU VM and use the
existing `SshExec` tunnel path. mini-SWE-agent calls the model from the driver, so
the model tunnel can bind to loopback; the task container itself needs no network.

All RunPod creation must go through `src/infra/runpod.py`, never a new POST /pods.
Use names derived from USER_PREFIX and a unique campaign ID. Capture actual pod
IDs and quotes. Arm watchdogs at creation, including an off-driver fallback for
the loss of the Vast VM, then perform one bounded calibration before expanding
to eight pods. Stop scale-up on an unready CPU, failed uploads, exhausted budget,
high infrastructure-failure rate or a saturated command-execution host.

Use independent vLLM instances, each with the SAME frozen no-DA checkpoint. Keep
131k context, correct thinking/tool parsers and prefix caching; declare a small
explicit serving concurrency instead of inheriting the family limit of 192.

The CPU coordinator owns task scheduling and HF publication. Store atomic task
state (pending/running/completed/infra-failed), attempt IDs, patches and complete
trajectories. Bind each active trajectory to one GPU; redistribute only unstarted
tasks. Keep a durable append-only attempt ledger so a reboot cannot turn pass@1
into repeated attempts until success. Lease recovery must fence out stale workers
before a task is reassigned. Never race multiple writers on one preds.json.

Checkpoint completed results continuously. If HF is unavailable, retain a bounded
local upload queue and retry with backoff; pause new generation before risking
unrecoverable disk exhaustion. Final results go to the same recorded HF repo,
not a new repo guessed from the grading date. Include infrastructure-invalid
counts, full denominator, protocol pins and cost. CPU grading may overlap rollout
if capacity permits. Release each GPU as soon as its remaining work is zero.

## Expiry, stop, resume, recovery

On Windows, run the scheduled expiry backstop with the CPU environment's
`Scripts/pythonw.exe` and `scratch/swebench_cpu_watchdog_windowless.pyw`, passing
the normal `--receipt` and `--env` arguments. The wrapper uses `CREATE_NO_WINDOW`
for the watchdog child, logs to `expiry-watchdog.log` beside the receipt, and
propagates its exit code. Direct interactive `python.exe` tasks flash a console
every five minutes. Preserve the expiry trigger; verify a manual scheduled run
has exit code zero after installation. The guest systemd expiry remains enabled.

- At the chosen deadline, cancel new work and STOP the owned CPU instance through
  the provider. GPU deadlines must already have expired and teardown been checked.
  Keep the disk. Confirm actual stopped state via the provider, not an SSH failure.
- Report retained-storage cost and the HF backup revision. Stopped Vast instances
  may have to wait for hardware availability to restart; retention is not reserved
  capacity or a guarantee against host failure/contract expiry.
- To resume, re-resolve the provider address, verify all identities and backups,
  establish a NEW explicitly requested deadline, and rerun readiness checks. Old
  expiry jobs must not unexpectedly stop the resumed session.
- On host loss, rebuild from this runbook, restore the pinned manifest and HF
  checkpoints, and continue only missing/eligible interrupted work. Do not treat
  the VM disk as the sole canonical results copy.
- Never destroy the VM merely to satisfy generic end-of-task teardown: the user
  explicitly requested retained storage. Never terminate another run's resources.

## Implementation status

Implemented and deployed: `scratch/swebench_cpu_bootstrap.sh`, CPU-only pinned
environments, `scratch/swebench_cpu_watchdog.py`, and
`scratch/swebench_cpu_prepare.py` with `scratch/swebench_cpu.yaml`. Preparation
freezes all 300 dataset rows, verifies Docker and off-host HF backup, records image
digests atomically, checks disk reserve, and tests reference patches. Run it as
`lasr-swebench-prepare.service`; progress is `/srv/lasr/runs/prepare.log`. The service
survives SSH disconnects. Restart it explicitly after fixing an infrastructure
failure; completed image records are reused. A failure must never open the GPU gate.

Provisioning receipt is local `output/swebench_cpu/receipt.json` and remote
`/srv/lasr/receipt.json`. Credentials are remote `/srv/lasr/credentials.env`, root-only,
with only HF_TOKEN, HF_ORG, USER_PREFIX, VAST_API_KEY. No RunPod credential has been
sent and no RunPod pod created. A separate CPU-to-RunPod SSH key exists on the guest;
the existing RunPod provisioner will embed its public key when GPUs are authorized
by the readiness gate. Never copy its private key into an artifact.

Docker Hub is a cold-start dependency: the live anonymous response advertised
`100;w=3600` and reached its quota before 300 pulls. The public Google mirror returned
404 for the four tested SWE images, so it cannot presently replace authentication.
Use a Docker Hub read-only token via `docker login --password-stdin` on this VM.
Load DOCKERHUB_USERNAME and DOCKERHUB_TOKEN from the original .env only when supplied;
never print the token. Docker Pro currently advertises $11/month with unlimited
standard pull rate; free access is possible but must honor rate-limit waits.
Do not repeatedly retry a quota error, rent extra hosts to rotate IPs, or rent GPUs
while waiting. Warm cached images avoid paying this setup cost on every model.

Browser device login is an alternative when no token is configured: run `docker login`
on the VM, open the exact activation URL it prints, and confirm that VM's code in the
authorized Docker account. This succeeded on 2026-09-23. Keep
`/root/.docker/config.json` root-only (0600) and outside all backup directories.
For reproducible provisioning, a read-only token in the original .env remains useful.

The tested token-authentication command (run from the isolated worktree) is:

```powershell
uv run --project scratch/swebench_cpu_env --frozen python -m scratch.swebench_cpu_auth `
  --env C:/Users/nikak/source/repos/LASR/teaching_claude_why_replication/.env `
  --receipt output/swebench_cpu/receipt.json --identity "$HOME/.ssh/msm_audit"
```

This re-resolves the VM address from Vast, checks the exact receipt ID and label,
uses the already trusted SSH host key, and sends the token over SSH stdin to
`docker login --password-stdin`. The helper does not copy the whole .env or print
provider responses/credentials. New hosts must first have their SSH identity verified.

Gold checks must use fresh attempt directories: the official harness can reuse an
existing report, including a previous failure. Never overwrite failed evidence or
treat a cached report as a fresh check. The initial two-repository gold check passed
2/2; the larger check exposed an external-network dependency. A reference-patch HTTP
502 is an infrastructure failure, not evidence of model incapability. Downloads may
continue while diagnosing it, but no GPU-ready marker may be produced on failed gold.

## Local HTTPBin protocol and known scoring limitations

The public service returned intermittent 502 errors. Do not rerun until a lucky
pass or count those errors as model failures. The preparation uses the original
HTTPBin image, pinned to
`kennethreitz/httpbin@sha256:599fe5e5073102dbb0ee3dbb65f049dab44fa9fc251f6835c9990f8fb196a72b`.
HTTP and HTTPS bind only to the Docker bridge address `172.17.0.1:80/443`.
The container name derives from USER_PREFIX and the receipted instance ID; startup
checks its ownership, image, command and bindings and refuses drift.

`scratch/swebench_local_httpbin.py` runs the pinned official 4.1.0 harness. For
requests grading only, it maps httpbin.org inside the task container to the local
service, sets the supported HTTPBIN_URL (including its required trailing slash),
and adds the fixture's public CA to the existing trust store. Certificate checks
remain enabled. Historical PreparedRequest/Session.send calls bypass environment
CA settings, so the wrapper also appends the public CA to requests' active bundle
immediately before tests; original and updated bundle hashes are logged. Task
patches, test patches and scoring labels remain unchanged. This is an explicit
grading environment deviation and must be recorded for every later model result;
compare control and DA using exactly the same fixture and wrapper.

The certificate and image/command provenance are backed up. The private TLS key
stays outside artifacts in `/srv/lasr/httpbin-tls/`, as do provider and Docker
credentials. A replacement host generates a new local certificate and records its
hash. Do not copy this override into agent rollouts silently; their existing
network-disabled environment remains a separate, recorded part of the scaffold.

The first PSF image experiment stalled; the HTTP-only original-image experiment
could not satisfy HTTPS tests. Both diagnostic attempts and failures are preserved.
The final HTTP/HTTPS setup passed all nine gold checks. A harmless marker-file
patch, leaving buggy source unchanged, resolved **2/6 requests cases**:
`psf__requests-2674` and `psf__requests-863`. Each also resolved without a source fix
under the unmodified official harness against public httpbin.org. These are
observed limitations of this pinned Lite protocol, not evidence that the model
fixed those issues. Keep all 300 tasks in the requested score and explicitly report
these no-fix passes; do not silently change the denominator or scoring tests.

The preparation script checks this six-case baseline on subsequent preparations;
an unexpected change fails readiness. This is an environment regression check,
not a substitute for the model's own 300-task score. No model run has happened yet.
Final readiness, image manifest and fixture metadata are downloaded from a pinned
HF commit and compared byte for byte; the verification receipt is
`/srv/lasr/runs/cpu-readiness-backup-verified.json`.

To resume preparation after fixing a failure:

```bash
cd /srv/lasr/repo
systemctl reset-failed lasr-swebench-prepare.service
systemctl start --no-block lasr-swebench-prepare.service
tail -f /srv/lasr/runs/prepare.log
```

Do not change the recorded STOP deadline merely to rerun preparation. Reusing a
host after expiry requires a new user-authorized deadline, as described above.

The inference coordinator and its launch/resume/grading/publication procedure are
now in [swebench_lite_runbook.md](swebench_lite_runbook.md). The service is installed
but has not been started. CPU-only integration checks passed; the first authorized
model run still begins with a bounded paid calibration before fleet expansion.

References checked 2026-09-23:
- https://docs.vast.ai/guides/instances/virtual-machines
- https://docs.vast.ai/sdk/python/reference/search-offers
- https://docs.vast.ai/sdk/python/reference/create-instance
- https://github.com/vast-ai/docs/blob/main/guides/pricing.mdx
- https://github.com/postmanlabs/httpbin
