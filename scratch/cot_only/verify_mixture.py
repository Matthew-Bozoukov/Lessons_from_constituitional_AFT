# ABOUTME: Verify the CoT-only mixture against its control and rehearse the mask gate
# ABOUTME: with the real tokenizer, so nothing about this arm is discovered on the pod.

"""Check the CoT-only arm differs from its control in exactly one way, then gate it.

Run (post-push, against exactly what the pod will pull -- the default):
  uv run python scratch/cot_only/verify_mixture.py
Run (pre-push, against a local build):
  uv run python scratch/cot_only/verify_mixture.py --built <dir>/mixture_think_cotonly.jsonl

Four claims, in order of what they would cost to get wrong:

  1. Every `text` is byte-identical to the control's, and exactly the 716
     difficult-advice rows carry `supervise: "cot"`. Anything else and the arms differ
     in more than the intervention.
  2. `gate_generation_boundary` passes on the REAL tokenizer with the REAL per-row
     modes — the same call train_lora makes, so a mask disagreement surfaces here
     rather than after a GPU is running.
  3. The forward-pass saving actually reaches the hardware: `plan_micro_batches` over
     the shuffled steps yields fewer, fuller passes than the control's. Shortening rows
     is only worth something if the batcher spends it.
  4. A sampled CoT row decodes to something a human can read: the trace and its close
     supervised, the prefill masked, the answer absent from the token stream entirely.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.cot_only.build_mixture import (  # noqa: E402
    CHUNK_ONLY, CONTROL_FILE, CONTROL_REPO, CONTROL_REVISION, DA_SOURCE, N_DA,
)
from src.model_profile import model_profile  # noqa: E402
from src.train.mask_gate import gate_generation_boundary  # noqa: E402
from src.train.masking import build_labels  # noqa: E402


def main(built: str = "", model_id: str = "Qwen/Qwen3.6-27B", max_length: int = 8192,
         show: int = 1, global_batch: int = 16, token_budget: int = 8000,
         arm: str = "", config: str = "", mode: str = "cot") -> None:
    """Compare, gate, plan and show.

    Args:
        built: A local mixture_think_cotonly.jsonl. Empty (the default) pulls the
            config's pinned HF revision instead — the bytes the pod will actually train
            on, which is the version worth verifying once the push has happened.
        model_id: Model whose tokenizer and profile define the token stream.
        max_length: Training sequence length; must match the train config.
        show: How many CoT rows to print a decoded sample of.
        global_batch: Examples per optimizer step, matching the train config.
        token_budget: Padded-token budget per forward pass — ModelProfile's measured
            H200 entry, which is what the trainer resolves on the pod.
        arm: "chunk_only" selects that arm's control repo/file/revision/source/counts in
            one word; anything else keeps the synthdoc defaults.
        config: Train config to read the pinned arm file from when `built` is empty.
        mode: The supervise mode the flagged rows must carry -- "cot" or "answer".

    Raises:
        AssertionError: Any of the four claims above fails.
    """
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from src.huggingface import resolve_dataset
    from src.train.dynamic_batching import plan_micro_batches

    load_dotenv()
    spec = CHUNK_ONLY if arm == "chunk_only" else {
        "repo": CONTROL_REPO, "file": CONTROL_FILE, "revision": CONTROL_REVISION,
        "da_source": DA_SOURCE, "n_da": N_DA}
    da_source, n_da = spec["da_source"], spec["n_da"]
    print(f">>> control: {spec['repo']}@{spec['revision'][:12]} ({spec['file']}), "
          f"da_source={da_source!r}, n_da={n_da}")
    ctrl_path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset",
                                revision=spec["revision"])
    ctrl = [json.loads(x) for x in Path(ctrl_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if built:
        arm_path = Path(built)
    else:
        cfg = OmegaConf.load(config or "configs/train/"
                             "lora_qwen36_t2_9284_synthdoc_716_cotonly_dynbatch_2xh200.yaml")
        arm_path, ref = resolve_dataset(str(cfg.data_repo), str(cfg.data_file),
                                        str(cfg.data_revision))
        arm_path = Path(arm_path)
        print(f">>> verifying the PINNED arm file: {ref['repo']}@{ref['revision'][:12]} "
              f"({ref['file']})")
        assert int(cfg.train.max_seq_len) == max_length, \
            f"config max_seq_len {cfg.train.max_seq_len} != {max_length}"
    new = [json.loads(x) for x in arm_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    # 1. one difference, and only one
    assert len(ctrl) == len(new), f"row count {len(ctrl)} -> {len(new)}"
    changed = [i for i, (a, b) in enumerate(zip(ctrl, new)) if a["text"] != b["text"]]
    assert not changed, f"{len(changed)} rows changed their text (first: {changed[:3]})"
    flagged = [i for i, r in enumerate(new) if r.get("supervise") == mode]
    assert len(flagged) == n_da, f"{len(flagged)} rows flagged, expected {n_da}"
    off_source = [i for i in flagged if new[i]["source"] != da_source]
    assert not off_source, f"{len(off_source)} flagged rows are not {da_source}"
    unflagged_da = [i for i, r in enumerate(new)
                    if r["source"] == da_source and r.get("supervise") != mode]
    assert not unflagged_da, f"{len(unflagged_da)} difficult-advice rows are unflagged"
    other_modes = {r.get("supervise") for r in new} - {None, mode}
    assert not other_modes, f"unexpected supervise modes present: {other_modes}"
    print(f">>> text byte-identical on all {len(new)} rows; "
          f"{len(flagged)} flagged supervise={mode}, all {da_source}")

    # 2. the gate the pod will run
    profile = model_profile(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    modes = [r.get("supervise") or "all" for r in new]
    gate_generation_boundary([r["text"] for r in new], tokenizer, max_length,
                             profile, thinking=True, supervise=modes)

    # 3. does the batcher actually spend the saving?
    def plan(modes_):
        lengths = [len(build_labels(r["text"], tokenizer, max_length, profile,
                                    supervise=m)["input_ids"])
                   for r, m in zip(new, modes_)]
        passes = padded = 0
        for s in range(0, len(lengths) - global_batch + 1, global_batch):
            step = lengths[s:s + global_batch]
            for mb in plan_micro_batches(step, token_budget):
                passes += 1
                padded += len(mb) * max(step[i] for i in mb)
        return passes, padded

    ctrl_passes, ctrl_padded = plan(["all"] * len(new))
    arm_passes, arm_padded = plan(modes)
    print(f">>> micro-batches over {len(new) // global_batch} steps @ budget "
          f"{token_budget:,}: {ctrl_passes:,} -> {arm_passes:,} forward passes "
          f"({100 * (1 - arm_passes / ctrl_passes):.1f}% fewer); padded tokens "
          f"{ctrl_padded:,} -> {arm_padded:,} ({100 * (1 - arm_padded / ctrl_padded):.1f}% fewer)")
    assert arm_passes <= ctrl_passes, \
        "shorter rows produced MORE forward passes; the batching assumption is wrong"

    # 4. eyeball it
    for i in flagged[:show]:
        out = build_labels(new[i]["text"], tokenizer, max_length, profile,
                           supervise=mode)
        kept = tokenizer.decode([v for v in out["labels"] if v != -100])
        masked = tokenizer.decode([t for t, v in zip(out["input_ids"], out["labels"])
                                   if v == -100])
        print(f"\n--- row {i} ({len(out['input_ids'])} tokens, "
              f"{sum(1 for v in out['labels'] if v != -100)} supervised) ---")
        print("MASKED tail :", repr(masked[-120:]))
        print("SUPERVISED  :", repr(kept[:200]), "...", repr(kept[-80:]))
        # The system and user turns keep their own turn ends; it is the FINAL assistant
        # turn whose shape distinguishes the two modes.
        stream = tokenizer.decode(out["input_ids"])
        final_turn = stream[stream.rfind(profile.assistant_header):]
        if mode == "cot":
            assert kept.endswith(profile.think_close), "a CoT row must end at its close"
            assert profile.turn_end not in final_turn, \
                "the answer and its turn end must not be in the final turn's stream"
            assert final_turn.endswith(profile.think_close), \
                "the final turn's token stream must end at the reasoning close"
        else:
            assert kept.endswith(profile.turn_end), \
                "an answer row must supervise through the turn end"
            assert profile.think_close not in kept, \
                "an answer row must not supervise the reasoning close"
            assert profile.think_close in final_turn, \
                "the trace must REMAIN in the token stream, merely unsupervised"
            # rstrip: the template emits a trailing newline AFTER the turn end, which the
            # span deliberately excludes -- `assistant_spans` stops at the terminator too.
            assert final_turn.rstrip("\n").endswith(profile.turn_end), \
                "an answer row keeps its whole turn, terminator included"
    print("\n>>> OK")


if __name__ == "__main__":
    fire.Fire(main)
