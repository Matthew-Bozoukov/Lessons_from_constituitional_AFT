<!-- ABOUTME: Reproducible reduction of historical nonmoral data to 15 percent supervised tokens. -->
<!-- ABOUTME: Records source pins, selection, publication, and planned token-mean SFT and ODCV. -->
# Nonmoral deliberation at 15% supervised tokens

User authorized a reduced mixture, single-H200 seed-0 training and three sequential
ODCV passes at temperature 0.7 and server seed 0. No new synthetic generation or
rewriting. Work is isolated on `codex/nonmoral-token15`, based on main `27623065`.
The proposed $60 ceiling ($30 training, $25 evaluation GPU/storage, $5 judges) is
pending explicit approval; no GPU has been rented as of this preparation record.

## Dataset

Parent: `dougalldeepmind/2026-09-15-nonmoral-original-7-mix`
at `b35ead8eaf4d59091e0ef1d08b187c7822f05632`.
Parent contains 684 historical craft-deliberation rows and 9,284 September-8 nosynth
replay rows. The original nonmoral corpus was not generated with the new constitution;
this subset retains its historical provenance rather than relabeling it.

Rank synthetic rows by SHA256 of seed 0 plus canonical complete row, and take the
prefix whose supervised-token share is closest to 15%. Preserve the original relative
order and complete payload of every retained row. This selection does not use quality
judgments or evaluation results. All replay rows remain unchanged.

Use the frozen Qwen tokenizer `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
and current native generation-boundary masks. Recount all rows without truncation and
exercise the trainer's actual Arrow loader before publication.

- **Published mixture:** `dougalldeepmind/2026-09-25-nonmoral-original-15-mix`
  at `f056dade96901bc6c7480a4f4149552592fef379`.
- **9,907 rows:** 623 nonmoral + 9,284 replay; 61 nonmoral rows removed.
- **5,309,764 supervised tokens:** 796,915 nonmoral + 4,512,849 replay.
- **Nonmoral share: 15.0084824862%.** Zero rows rewritten or truncated.
- Mixture SHA256: `04778a87c01b6318096a885759615ae1010896a1f2a1a3e2c08dd1263afd00cb`.
- All published files verified by pinned download and SHA256; selected parent indices,
  full parent token census, resolved config and validation are included on the Hub.

Reproduce with `uv run python -m scratch.dataset_refresh.reduce_mixture --config
scratch/dataset_refresh/nonmoral_token15.yaml --publish` (a new output directory and
unused artifact name are required; existing publications are never overwritten).

## Training and evaluation plan

Current standard SFT: BF16 Qwen3.6-27B, rank 64, alpha 128, dropout .05, seed 0,
one epoch, global batch 16, LR 1e-4/cosine/5% warmup, 8,192 sequence ceiling,
8,000-token dynamic batching budget, boundary-aware packing, **token_mean** loss.
Expected 620 optimizer steps on one H200. Historical nonmoral adapter used equal
weight per row; the new experiment changes both token share and loss aggregation.

Use existing `scratch/nonmoral/train_pair.py` with one arm, independent bounded
watchdogs, CUDA and native mask gates, finite-loss/gradient monitoring, verified adapter
publication and full direct-to-Hub checkpoint archive before owned-pod teardown.

ODCV: one H100, local Docker with 24 concurrent scenarios, 40 mandated plus 40
incentivized scenarios, three sequential passes (240 total), one persistent server,
startup seed 0, temperature .7, no per-request seed, 28k context. Gemini 3 Flash
misalignment and progress judges. First counted cell is the health pilot; every pass
persists its rollouts before the next, with server continuity checked at boundaries.
No global Docker pruning; unrelated pods and local jobs are outside scope.

Docker preflight passed: 20 CPUs, 18.9 GB memory, capacity for 1,024 networks, all 168
benchmark shell files LF, exactly 40 scenarios per variant. Thirty owner, backup,
evaluation-plan and server-startup regression tests passed.
