# Resume ODCV on synr64 (table2-synthdoc r64) — quick-start

Everything below is already fixed/saved. To restart the ODCV run:

## 1. Provision pod(s) — H100 NVL, NOT H200
```
cd .claude/worktrees/fixing_synthdoc
uv run python scripts/runpod_capability.py up \
  --config configs/serve_table2_synthdoc_r64.yaml \
  --name serve-synr64-odcv-a --gpu "NVIDIA H100 NVL"
```
- **Driver lottery:** some RunPod hosts run an old NVIDIA driver ("RuntimeError: driver ... too old"). If the pod's `vllm.log` shows that at engine-init (~10 min), **just tear down and re-provision** — a different host will have a newer driver. **H200 is fleet-wide on the old driver right now — avoid it.**
- The serving driver is already patched with everything needed:
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml` (Qwen3.6 emits XML tool calls),
  `--max-num-seqs 256` (hybrid-Mamba models refuse CUDA-graph capture when max_num_seqs > available Mamba blocks),
  `--gpu-memory-utilization 0.90`, no `--enforce-eager` (graphs → faster decode).

## 2. Point the config at the new pod id
Edit `configs/odcv_bench_table2_synthdoc_r64_1pod.yaml` → `base_url:` to `https://<POD_ID>-8000.proxy.runpod.net/v1`.
(For 2-pod sharding use `_shard0.yaml`/`_shard1.yaml` instead — but see concurrency note.)

## 3. Run (rollout → judge)
From the worktree, once serving (`/v1/models` lists `synr64`):
```
set -a; source .env; set +a
uv run scripts/odcv_rollout.py --config <MAIN>/configs/odcv_bench_table2_synthdoc_r64_1pod.yaml
# then, on the printed rollout dir:
uv run scripts/odcv_judge.py --rollout_dir <that dir> --config <same config>
```
`odcv_1pod_run.sh` in this folder chains both.

## Key settings learned (don't re-derive)
- **`concurrency: 12`, NOT higher.** The bottleneck is the *local* Docker host, not the pod (pod sits at ~13% KV). At concurrency 30, ~60 local containers → 58/70 cells failed with `compose_exit_1+no_container`. 12 runs clean (~24 containers). The pod is underused but that's fine.
- **`rollouts_per_cell: 4`** (was 5 default). 70 cells × 4 = 280 rollouts.
- **Per-cell ~3–13 min**; single pod at conc 12 → ~1.5 h rollouts + ~30–60 min judging.
- Smoke gate: the `--smoke` first-scenario collides with `exclude_scenarios` (both are "Academic-Research-Integrity-Finding") → 0 scenarios. Use `_smoke.yaml` (empty exclusions) to validate wiring.

## Stay-awake if leaving it running
Wrap in `systemd-inhibit --what=idle:sleep:handle-lid-switch --mode=block --why=odcv <cmd>` (prevents idle-sleep).
**Lid-close still suspends** on this machine (logind default `LidSwitchIgnoreInhibited=yes`). To close the lid:
`gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action nothing` (+ `lid-close-battery-action`). Revert with `suspend`.

## Partial result from the interrupted run (n=12, not full)
synr64 overall misalignment ~8% vs. base baseline (`qwen3.6-27b-base-fp8`) ~37% — directionally your model is much more aligned, but n=12 is too small (need the full 70).
