# ABOUTME: Exercise custom runners through the actual shared serving and publication lifecycle.
# ABOUTME: Network and GPU boundaries are replaced with fakes; layout and provenance stay real.
import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval import run_eval
from src.eval.layout import assert_layout
from src.infra.endpoints.vllm import TargetSpec


@pytest.mark.parametrize("push", [False, True])
def test_custom_runner_keeps_shared_lifecycle_and_metadata(monkeypatch, tmp_path, push):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_ORG", "example")
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"output_root": str(tmp_path / "runs"),
                                    "generation": {"temperature": 0.6}}), cfg_path)
    spec = TargetSpec(hf_path="org/qwen36-da-0", base_model="Qwen/Qwen3.6-27B",
                      adapter=True, mode="think", model_key="qwen36_da_0", lora_rank=64,
                      revision="adapter-pin", base_revision="base-pin")
    events, outputs = [], []

    class Server:
        def __init__(self, **kwargs):
            assert kwargs["work_dir"] == tmp_path / "runs/server"

        def ensure(self, target):
            events.append("serve")
            return SimpleNamespace(spec=target)

        def stop(self):
            events.append("stop")

    def custom(target, cfg, out_dir):
        assert target.spec == spec and cfg._run_eval.push is push
        events.append("run")
        outputs.append(out_dir)
        (out_dir / "rollouts").mkdir()
        (out_dir / "rollouts/episode.json").write_text('{}')
        (out_dir / "results").mkdir()
        (out_dir / "results/episode.json").write_text('{}')
        return {"completed": 1}

    def publish(out_dir, repo, card, **kwargs):
        assert_layout(out_dir)
        assert repo.startswith("example/")
        assert {"eval-run", "eval:mmlu", "mode:think"} <= set(kwargs["front_matter"]["tags"])
        assert json.loads(card["models"])["target_revision"] == "adapter-pin"
        events.append("publish")
        return "https://huggingface.co/datasets/" + repo

    monkeypatch.setattr(run_eval, "VllmServer", Server)
    monkeypatch.setattr(run_eval, "resolve_target", lambda _: spec)
    monkeypatch.setattr(run_eval, "push_run_dir", publish)
    monkeypatch.setattr(run_eval, "resolve", lambda _: pytest.fail("Custom runner was ignored"))
    args = ["--name", "mmlu", "--target", spec.hf_path, "--config", str(cfg_path)]
    run_eval.main(args + ([] if push else ["--no-push"]), runner=custom)
    assert events == (["serve", "run", "publish", "stop"] if push else ["serve", "run", "stop"])
    assert_layout(outputs[0])
    meta = json.loads((outputs[0] / "metadata/run_meta.json").read_text())
    assert meta["target_revision"] == "adapter-pin" and meta["base_model_revision"] == "base-pin"
    assert meta["runner"].endswith(":test_custom_runner_keeps_shared_lifecycle_and_metadata.<locals>.custom")


def test_checkpoint_obeys_no_push(tmp_path, monkeypatch):
    from src.eval.misalignment.delegated_harm.runner import checkpoint
    monkeypatch.setattr("src.infra.huggingface.push_run_dir", lambda *a, **k: pytest.fail("Unexpected upload"))
    (tmp_path / "results").mkdir()
    assert checkpoint(tmp_path, None, OmegaConf.create({"_run_eval": {"push": False}}), {"arms": {}}) == ""
    assert (tmp_path / "results/results.json").exists()
    assert not tmp_path.with_name(tmp_path.name + "-checkpoint").exists()
