# ABOUTME: A vLLM server started on this machine gets flashinfer's JIT sampler disabled when there is no
# ABOUTME: nvcc to compile it, and the pod weight cache when it exists; exported values are never overridden.
from src.infra.endpoints import vllm


def test_no_nvcc_disables_the_flashinfer_sampler_and_a_pod_cache_is_used():
    env = vllm.local_base_env({}, which=lambda name: None, isdir=lambda path: path == f"{vllm.POD_WORKDIR}/hf")
    assert env == {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HOME": f"{vllm.POD_WORKDIR}/hf"}


def test_a_machine_with_nvcc_and_no_cache_gets_nothing():
    assert vllm.local_base_env({}, which=lambda name: "/usr/local/cuda/bin/nvcc", isdir=lambda path: False) == {}


def test_exported_values_win():
    environ = {"VLLM_USE_FLASHINFER_SAMPLER": "1", "HF_HOME": "/elsewhere"}
    assert vllm.local_base_env(environ, which=lambda name: None, isdir=lambda path: True) == {}


def test_local_exec_merges_base_env_under_the_callers(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(vllm, "local_base_env", lambda environ: {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HOME": "/cache"})
    monkeypatch.setattr(vllm.subprocess, "Popen", lambda argv, **kw: seen.update(env=kw["env"]) or type("P", (), {"poll": lambda self: None})())
    vllm.LocalExec(tmp_path).start_server(["python", "-m", "vllm"], {"HF_HOME": "/caller"})
    assert seen["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and seen["env"]["HF_HOME"] == "/caller"
