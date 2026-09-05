<!-- ABOUTME: Per-arm train configs from before 2026-09-05, kept as the record of what ran. -->
<!-- ABOUTME: Not runnable as-is: the trainer refuses their retired keys and names the fix. -->

# Archived per-arm train configs

Until 2026-09-05 every arm had its own train config (`<model>-<mix>-<pct>.yaml`) that named
the model, the mixture repo and the thinking declaration alongside the recipe. The recipe was
the same in all of them: with the data pointers and output directory stripped, the dynbatch
configs here hash identically. That recipe is now the ONE file `configs/train.yaml` (at the top of configs/: a folder for a
single file says nothing), and the model, data and thinking declaration are launch arguments:

    uv run train --config configs/train.yaml model=qwen36 data_repo=<org>/<mix> thinking=true

These files stay so a reader can see what each historical arm declared. To RE-RUN one of them
exactly, use the adapter it produced: every adapter on the Hub carries the resolved config it
trained with (`train_config.yaml` + `training_meta.json`, with `git_sha`), and a config from
before the recipe split runs from the code at that commit. The model half these files carried
(`model_class`, `lora.target_modules`, `load_in_4bit`, `attn_implementation`) lives in
`configs/models/<key>.yaml`; `dynamic_batching`, `packing`, `assistant_only_loss`, `loss_type`
and `mask_empty_think` are no longer knobs (always on, off, on, seq-mean-token-mean, and the
unconditional whole-marker mask, respectively).

## Arms that arrived on main after the split

`qwen36-rewardhack-702-dynbatch.yaml` (2026-09-05) came with exactly the recipe's values and
had not been trained under it. Under the recipe the same run is:

    uv run train --config configs/train.yaml model=qwen36 \
        data_repo=LASR-Callum/2026-09-05-table2-9284-da-rewardhack-702-train \
        data_file=t2_9284_da_rewardhack_702.jsonl \
        data_revision=4e7dd02a582ee01e26b5a39e7429a285cb50ea22 thinking=true

(`data_file=` because that repo predates the `mixture.jsonl` contract.)
