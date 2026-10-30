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
@pytest.mark.parametrize("revision", [None, "adapter-pin"])
def test_custom_runner_keeps_shared_lifecycle_and_metadata(monkeypatch, tmp_path, push, revision):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_ORG", "example")
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"output_root": str(tmp_path / "runs"),
                                    "target_revision": revision,
                                    "generation": {"temperature": 0.6}}), cfg_path)
    spec = TargetSpec(hf_path="org/qwen36-da-0", base_model="Qwen/Qwen3.6-27B",
                      adapter=True, mode="think", model_key="qwen36_da_0", lora_rank=64,
                      revision="adapter-pin", base_revision="base-pin")
    events, outputs = [], []

    class Server:
        def __init__(self, **kwargs):
            assert kwargs["work_dir"] == tmp_path / f"runs/server_{kwargs['port']}"

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

    monkeypatch.setattr(run_eval, "_credentials_preflight", lambda **kwargs: None)
    monkeypatch.setattr(run_eval, "VllmServer", Server)
    def resolve_pinned(target, **kwargs):
        assert kwargs == ({"revision": revision} if revision else {})
        return spec
    monkeypatch.setattr(run_eval, "resolve_target", resolve_pinned)
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


def _preamble_arm(monkeypatch, tmp_path, *, protocol: bool):
    """Scaffold a `--name mask` (has the protocol facet) launch with a tmp constitution, --no-push."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_ORG", "example")
    constitution = tmp_path / "constitution.md"
    constitution.write_text("## Priority\n\nBe honest.\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"output_root": str(tmp_path / "runs"),
                                    "generation": {"temperature": 0.6}}), cfg_path)
    spec = TargetSpec(hf_path="org/qwen36-da-0", base_model="Qwen/Qwen3.6-27B",
                      adapter=True, mode="think", model_key="qwen36_da_0", lora_rank=64,
                      revision="adapter-pin", base_revision="base-pin")
    events, outputs = [], []

    class Server:
        def __init__(self, **kwargs):
            pass

        def ensure(self, target):
            events.append("serve")
            return SimpleNamespace(spec=target)

        def stop(self):
            events.append("stop")

    def custom(target, cfg, out_dir):
        events.append("run")
        outputs.append(out_dir)
        # The published copy is on disk BEFORE the eval runs, and only metadata/ was created.
        assert (out_dir / "metadata/system_preamble.md").exists()
        (out_dir / "rollouts").mkdir(exist_ok=True)
        (out_dir / "rollouts/episode.json").write_text('{}')
        (out_dir / "results").mkdir(exist_ok=True)
        (out_dir / "results/episode.json").write_text('{}')
        return {"completed": 1}

    monkeypatch.setattr(run_eval, "_credentials_preflight", lambda **kwargs: None)
    monkeypatch.setattr(run_eval, "VllmServer", Server)
    monkeypatch.setattr(run_eval, "resolve_target", lambda target, **kwargs: spec)
    monkeypatch.setattr(run_eval, "push_run_dir", lambda *a, **k: pytest.fail("--no-push was given"))
    monkeypatch.setattr(run_eval, "resolve", lambda _: pytest.fail("Custom runner was ignored"))
    args = ["--name", "mask", "--target", spec.hf_path, "--config", str(cfg_path), "--no-push",
            f"system_preamble_file={constitution}"] + (["protocol=constitution"] if protocol else [])
    return args, custom, events, outputs


def test_run_eval_records_and_materialises_the_system_preamble(monkeypatch, tmp_path):
    from src.eval.preamble import FRAMING

    args, custom, events, outputs = _preamble_arm(monkeypatch, tmp_path, protocol=True)
    run_eval.main(args, runner=custom)
    assert events == ["serve", "run", "stop"]
    assert_layout(outputs[0])
    meta = json.loads((outputs[0] / "metadata/run_meta.json").read_text())
    assert meta["system_preamble"]["text"].startswith(FRAMING)
    assert "<constitution>\n## Priority\n\nBe honest.\n</constitution>" in meta["system_preamble"]["text"]
    assert meta["system_preamble"]["sha256"] and meta["system_preamble"]["constitution_sha256"]
    assert meta["variant"] == "constitution"
    assert (outputs[0] / "metadata/system_preamble.md").read_text() == meta["system_preamble"]["text"] + "\n"


def test_preamble_without_protocol_is_refused(monkeypatch, tmp_path):
    args, custom, events, _ = _preamble_arm(monkeypatch, tmp_path, protocol=False)
    with pytest.raises(SystemExit, match="protocol="):
        run_eval.main(args, runner=custom)
    assert events == []  # refused before anything was served
