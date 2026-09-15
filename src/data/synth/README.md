<!-- ABOUTME: Index of synthetic alignment data generation methods. -->
<!-- ABOUTME: Links constitutional SFT generation and deliberative SFT from final HF prompt pools. -->

# Synthetic alignment data

Generation methods live under `src/data/synth/`:

- [`ours/`](ours/README.md): the existing config-driven
  constitution-guided generation and revision pipeline. It generates chat-formatted
  training data, including difficult advice, pre-action deliberation, peer critique,
  and fiction, building on the difficult-advice SFT recipe in
  [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/).
- [`deliberative_alignment/`](deliberative_alignment/README.md): reads final prompts
  from a completed HF corpus and generates native Qwen reasoning and answers through
  Alibaba/OpenRouter with the full constitution, removed from the SFT input afterward.

`uv run synth run --config ...` and `scripts/data/synth/build_dataset.py` select the
method from the YAML's `method` field: `ours` for stage-based recipes and
`deliberative_alignment` for `delib.yaml`. Older configs without the field default to `ours`.
Both methods publish `stages/` snapshots and a final `dataset.jsonl`. Auxiliary commands
(`check`, `checks`, `topup`, `estimate`, `segment`, `chunkings`) serve constitutional SFT.

```sh
uv run synth run --config configs/data/synth/da.yaml --smoke
uv run synth run --config configs/data/synth/delib.yaml --smoke
uv run synth chunkings
```
