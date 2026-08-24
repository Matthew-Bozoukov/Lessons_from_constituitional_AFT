<!-- ABOUTME: How to split one SWE-bench baseline run across several machines and merge the -->
<!-- ABOUTME: results, without either machine measuring a different benchmark. -->

# Sharding the SWE-bench baseline across machines

The `swebench_mini` eval is bottlenecked by **driver-side concurrency**, not by the GPU. Each
rollout is an agent loop that spends most of its time running bash inside a container on the
machine that drives the eval; the model is idle much of that time. Measured on the pilot
(2026-08-05): 5 in-flight requests held the server at 38% KV cache with **zero** queuing.

So the way to go faster is more machines running containers — each with its own GPU endpoint
once one box saturates (~13–15 concurrent requests at the 38–81k contexts this eval produces).

## The rule that keeps it one benchmark

A shard is a **division of labour, never a different benchmark**:

- Every driver selects the *same* subset first (same `subset.fraction`, same `subset.seed`),
  then takes its slice. `subset_hash` in `metadata/selection.json` identifies the **full** subset and
  **must match across drivers** — that is the pre-merge check.
- Slices are disjoint and their union is exactly the subset.
- pass@1 is scored against the **full** denominator, so an instance nobody completed counts
  as unresolved rather than vanishing.

Splitting is round-robin **within each repo**, which bounds each shard's per-repo count to
±1 of its share. Two weaker schemes were rejected: contiguous blocks give one driver the
front of every repo's ranking and the other the back; flat round-robin aliases against the
stratified order (a 3-way split handed one shard 14 django instances where 16.7 were due).

## Running it

Both machines run the same command except `shard.index`:

```bash
# machine A
uv run scripts/run_eval.py --target <targets...> --name swebench_mini \
    --server <gpu-alias-A> mode=think subset.fraction=0.5 shard.count=2 shard.index=0

# machine B
uv run scripts/run_eval.py --target <targets...> --name swebench_mini \
    --server <gpu-alias-B> mode=think subset.fraction=0.5 shard.count=2 shard.index=1
```

Each machine needs its own GPU endpoint. Sharing one endpoint between two drivers works only
until the server saturates — past ~13–15 concurrent requests they queue and the second
machine buys nothing.

### Renting the serving box: filter on CUDA, not just VRAM

Three hosts were burned learning this on 2026-08-05. vLLM 0.26's torch build **requires a
host driver of CUDA ≥ 13.0**; anything older dies at engine init with *"The NVIDIA driver on
your system is too old"* — observed at both `12040` and `12080`, so 12.8 is not enough.
Search with **all** of these:

```bash
uvx vastai search offers 'num_gpus=1 rentable=true gpu_ram>=79 disk_space>=200 \
  inet_down>=2000 reliability>=0.98 cuda_max_good>=13.0' -o dph+
```

- `gpu_ram>=79` is mandatory: the name `A100_SXM4` covers a **40 GB** variant that cannot
  hold this model, and it will happily rent to you.
- `cuda_max_good>=13.0` is mandatory for the reason above.
- Hosts also sometimes hang in `loading` for 10+ minutes while pulling the ~9 GB PyTorch
  image and never come up. Give a host ~4 minutes, then destroy and try the next offer —
  a create that returns `'success': False` is a strong hint it will not start.

Targets run **sequentially** on each machine, and all three arms share one vLLM process
(same base model and mode, adapters hot-swap). Order LoRAs first so the server starts once
with `--enable-lora` and never restarts.

## Merging

1. Check `subset_hash` **and** `dataset_revision` match across `metadata/selection.json` files. If
   they differ, the runs are not mergeable — do not average them.
2. Union the per-shard `preds.json` files.
3. Grade the union once, against the full instance list (`full_instance_ids`).

Grading is CPU + docker only, needs no GPU, and can run on either machine after its rollouts
finish — the images are already warm there. It **cannot run on Windows**: the official
harness imports the Unix-only `resource` module at package-import time.

## Speed notes learned from the pilot

- **Prefix caching is off by default in vLLM 0.26.** Agent loops re-send their whole history
  every step, so a 38–81k context gets prefilled ~32 times per instance. The `serving:` block
  turns it on; it changes no outputs.
- **Images are pulled once and shared by every arm** — same 250 instances, same images. Only
  the first arm pays. Budget ~1.36 GB per instance (measured), so ~340 GB for 250, less in
  practice since same-repo instances share layers.
- **`pull_overlap: true`** runs the pull on a background thread so rollouts start immediately.
  It requires the raised `pull_timeout`, because an instance that outruns the pre-pull falls
  back to mini-SWE-agent's own on-demand pull, whose 120s default cannot cover a cold image.
- **Driver RAM is the cap on workers.** Each concurrent instance holds a container; ~1 GB
  each is a safe planning figure. Docker Desktop defaults to ~50% of host RAM.
