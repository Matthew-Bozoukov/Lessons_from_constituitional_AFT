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

from src.infra.endpoints.vllm import LocalExec, SshExec


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


def test_release_stops_the_server_once_and_hands_the_host_back():
    from types import SimpleNamespace
    from src.infra.endpoints.vllm import VllmServer, ServedTarget

    events = []
    executor = SimpleNamespace(stop_server=lambda: events.append("stop"), endpoint_host="127.0.0.1")
    server = VllmServer(work_dir=Path("x"), executor=executor, on_release=lambda: events.append("pod-down"))
    spec = SimpleNamespace(hf_path="org/m", base_model="Qwen/Qwen3.6-27B", adapter=False, mode="think",
                           model_key="m", api_base=None, answers=None)
    target = ServedTarget(spec, server)
    target.release()
    target.release()
    server.stop()      # run_eval's finally after an early release: nothing left to stop
    assert events == ["stop", "pod-down"]
    unowned = VllmServer(work_dir=Path("x"), executor=SimpleNamespace(stop_server=lambda: events.append("stop2"), endpoint_host="h"))
    unowned.release()
    assert events == ["stop", "pod-down", "stop2"], "no owner: the server stops, nothing else happens"
