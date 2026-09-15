# ABOUTME: Fork seed 0's principle-scoped ODCV config for seeds 42/69, changing ONLY the served
# ABOUTME: model and its output key, and ASSERTING every other key is byte-identical to seed 0's.
# Run: uv run python scratch/da716_seeds/fork_odcv_configs.py
#
# Why an assertion and not a careful copy: the whole point of the seed replicates is that they
# land on exactly seed 0's 65 cells with seed 0's judges and temperature. A drifted
# exclude_scenarios list or a second judge would make the seed-variance comparison the plot
# needs into a comparison of two different evals, and nothing downstream would notice.

from __future__ import annotations

from pathlib import Path

import fire
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED0 = ROOT / "scratch/da_chunk_only/odcv_bench_t2_9284_da_chunk_only_702_2x65.yaml"
# The vLLM --name each adapter is served under, and the output/ key its run dir takes.
SEEDS = {
    42: dict(
        model="da_principle_scoped_702_s42",
        model_key="qwen3_6-27b-lora-t2-9284-da-principle-scoped-702-r64-seed42",
    ),
    69: dict(
        model="da_principle_scoped_702_s69",
        model_key="qwen3_6-27b-lora-t2-9284-da-principle-scoped-702-r64-seed69",
    ),
}
MAY_DIFFER = {"model", "model_key"}


def main() -> None:
    base_text = SEED0.read_text(encoding="utf-8")
    base = yaml.safe_load(base_text)
    for seed, over in SEEDS.items():
        text = base_text
        for k, v in over.items():
            old = f'{k}: "{base[k]}"'
            assert old in text, f"{k} not found verbatim in seed 0's config"
            text = text.replace(old, f'{k}: "{v}"', 1)
        header = (
            f"# ABOUTME: ODCV for SEED {seed} of the principle-scoped 702 arm — the difficult-advice\n"
            f"# ABOUTME: baseline. Forked from seed 0's config; only `model` and `model_key` differ.\n"
            f"# Run: bash scratch/da716_seeds/odcv_seed_run.sh <pod_ip> <pod_ssh_port> {seed}\n"
            "#\n"
            "# Everything else -- temperature 0.0, the 15 exclude_scenarios, both judges, the\n"
            "# host.docker.internal base_url, concurrency and timeouts -- is byte-identical to\n"
            "# seed 0's, asserted by scratch/da716_seeds/fork_odcv_configs.py. That is what makes\n"
            "# the three seeds a seed-variance comparison rather than three different evals.\n"
            "#\n"
        )
        # Drop seed 0's own two ABOUTME lines, keep the rest of its commentary.
        body = "\n".join(
            ln for ln in text.splitlines() if not ln.startswith("# ABOUTME:")
        )
        out = ROOT / (
            "scratch/da716_seeds/2026-08-31_odcv_bench_table2_9284_difficult_advice_"
            f"principle_scoped_702_seed_{seed}_2_65.yaml"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(header + body.lstrip("\n") + "\n", encoding="utf-8")

        got = yaml.safe_load(out.read_text())
        diff = {k for k in set(base) | set(got) if base.get(k) != got.get(k)}
        assert diff == MAY_DIFFER, (
            f"seed {seed}: expected only {MAY_DIFFER} to differ, got {diff}"
        )
        assert got["exclude_scenarios"] == base["exclude_scenarios"]
        assert (
            got["judges"] == base["judges"]
            and got["temperature"] == base["temperature"]
        )
        print(f"seed {seed}: {out.relative_to(ROOT)}")
        print(
            f"          only {sorted(diff)} differ; {len(got['exclude_scenarios'])} exclusions, "
            f"judges {sorted(got['judges'])}"
        )


if __name__ == "__main__":
    fire.Fire(main)
