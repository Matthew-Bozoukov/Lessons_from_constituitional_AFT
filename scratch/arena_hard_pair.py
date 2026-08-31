# ABOUTME: Standalone Arena-Hard SxS for the two paper-table2 adapters: serve base+both LoRA on
# ABOUTME: a (2xH100, TP=2) host, generate both arms' answers, judge synthdoc-vs-table2only and
# ABOUTME: table2only-vs-table2only, then report win rates. Run: uv run python scratch/arena_hard_pair.py --server <alias>

from __future__ import annotations

import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.endpoints.vllm_server import SshExec, VllmServer, resolve_target  # noqa: E402
from src.eval.capabilities.arena_hard import arena_hard_gen, arena_hard_judge  # noqa: E402
from src.utils import timestamp  # noqa: E402

BASELINE = "LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64"
TARGET = "LASR-Callum/qwen3.6-27b-lora-table2-synthdoc-r64"


def main(server: str | None = None, port: int = 8000, stage: int = 0,
         gen_only: bool = False, judge_only: bool = False, smoke: bool = False,
         baseline: str = BASELINE, target: str = TARGET,
         skip_arms: str = "") -> None:
    """Drive the two-adapter Arena-Hard comparison end to end.

    Args:
        server: SSH alias of the GPU host (per the playbook). Omit to serve locally.
        port: Local port for the tunnelled/served endpoint.
        stage: hard_prompt count override (0 = the arm's full 500).
        gen_only: Stop after generating answers (skip judging).
        judge_only: Skip generation/serving; judge existing answers only.
        baseline: HF adapter every win rate is measured against.
        target: HF adapter under test.
        skip_arms: Comma-separated arm KEYS whose answers are already cached in the
            vendor tree — generation is skipped for them, judging is not. Reusing an
            arm's answers is only sound when the decoding config is unchanged
            (temperature/top_p/max_tokens/thinking); the caller must check that.
        smoke: 4 questions/category + 4-question judging — validate the whole pipeline cheaply.
    """
    load_dotenv()
    out_dir = Path("output/arena_hard") / f"table2_pair_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load("configs/eval/arena_hard.yaml"),
                          OmegaConf.load("scratch/arena_hard_pair_overrides.yaml"))
    cfg.output_dir = str(out_dir)
    cfg_path = out_dir / "arena_hard_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    base_spec = resolve_target(baseline)
    tgt_spec = resolve_target(target)
    base_key, tgt_key = base_spec.model_key, tgt_spec.model_key
    print(f">>> baseline arm = {base_key}\n>>> target   arm = {tgt_key}\n>>> out_dir = {out_dir}")

    if not judge_only:
        executor = SshExec(server, port=port, bind="127.0.0.1") if server else None
        if executor:
            executor.check_ready()
        # `serve_requirements` (was `serve_overrides` before the 2026-08-06 refactor): the
        # eval's `serving:` block states what this eval NEEDS; family FACTS come from
        # ModelProfile and cannot be written from a config.
        server_obj = VllmServer(work_dir=out_dir / "server", port=port, executor=executor,
                                serve_requirements=OmegaConf.to_container(
                                    cfg.get("serving", {}), resolve=True))
        try:
            skip = {a.strip() for a in skip_arms.split(",") if a.strip()}
            for spec, key in ((base_spec, base_key), (tgt_spec, tgt_key)):
                if key in skip:
                    print(f">>> {key}: generation SKIPPED (answers already cached)")
                    continue
                served = server_obj.ensure(spec)   # first starts the server, second hot-loads
                print(f">>> generating {key} against {served.base_url} (model={served.model_name})")
                arena_hard_gen.main(config=str(cfg_path), arm=key,
                                    served_model=served.model_name, endpoint=served.base_url,
                                    stage=stage, smoke=smoke)
        finally:
            server_obj.stop()
        if gen_only:
            print(">>> gen_only: stopping before judging"); return

    # Judge both comparisons against the table2-only baseline. smoke -> tiny stage so the
    # judge only scores the 4 questions the smoke gen produced.
    jstage = 4 if smoke else stage
    for arm in (tgt_key, base_key):     # synthdoc vs table2only ; table2only vs table2only (self≈50%)
        print(f">>> judging {arm} vs baseline {base_key}")
        arena_hard_judge.main(config=str(cfg_path), mode="judge", arm=arm, stage=jstage)

    print(f"\n>>> DONE. Judgments under {cfg.vendor_dir}/data/{cfg.bench_name}/model_judgment/")
    print(f">>> report: uv run python scratch/reports/arena_hard_report.py --config {cfg_path}")


if __name__ == "__main__":
    fire.Fire(main)
