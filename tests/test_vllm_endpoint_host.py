# ABOUTME: The driver must talk to the address the SSH tunnel actually binds, not localhost.
# ABOUTME: Regression test for the docker-bridge bind hang observed 2026-08-06.

"""A remote server tunnelled to the docker bridge is NOT reachable on localhost.

`run_eval` binds the tunnel to 172.17.0.1 for every `needs_docker` eval on linux, so that
scenario containers can reach the model. The tunnel then listens on that address and ONLY
that address. Any code that assumes localhost — the health probe, the base_url handed to the
eval, the runtime LoRA-load call — silently cannot connect. The failure is maximally
confusing: vLLM starts fine and serves the adapter, while the driver blocks for the full
30-minute health timeout and then reports a timeout that reads like a slow model load.
"""

from pathlib import Path

from src.endpoints.vllm_server import LocalExec, SshExec


def test_local_exec_uses_loopback():
    assert LocalExec(Path("output/serve")).endpoint_host == "127.0.0.1"


def test_ssh_exec_defaults_to_loopback():
    assert SshExec("pod", port=8000).endpoint_host == "127.0.0.1"


def test_ssh_exec_follows_the_docker_bridge_bind():
    """The whole point: bind=172.17.0.1 must NOT resolve to localhost."""
    exec_ = SshExec("pod", port=8000, bind="172.17.0.1")
    assert exec_.endpoint_host == "172.17.0.1"
    assert exec_.endpoint_host == exec_.bind


def test_endpoint_host_tracks_bind_for_any_address():
    for addr in ("127.0.0.1", "172.17.0.1", "172.18.0.1", "10.0.0.5"):
        assert SshExec("pod", port=8000, bind=addr).endpoint_host == addr


def test_every_executor_exposes_endpoint_host():
    """VllmServer reads executor.endpoint_host; an executor without it breaks serving."""
    for exec_ in (LocalExec(Path("output/serve")), SshExec("pod", port=8000)):
        assert isinstance(getattr(exec_, "endpoint_host", None), str)
