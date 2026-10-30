# ABOUTME: Offline tests for the RunPod REST v2 client: retry on 429, problem detail on errors,
# ABOUTME: pagination, the create body (data-centre pin, in-pod expiry), SSH endpoint, spend.
import json
from types import SimpleNamespace

import pytest
import requests

from src.infra import runpod


def response(status: int, body=None, headers=None) -> SimpleNamespace:
    content = json.dumps(body).encode() if body is not None else b""
    return SimpleNamespace(status_code=status, ok=200 <= status < 300, content=content,
                           headers=headers or {}, text=content.decode(),
                           json=lambda: json.loads(content))


@pytest.fixture
def api(monkeypatch):
    """Scripted responses; records every request the client makes."""
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(runpod.time, "sleep", lambda s: None)
    calls: list[tuple[str, str, dict]] = []
    queue: list[SimpleNamespace] = []
    def request(method, url, **kw):
        calls.append((method, url, kw))
        return queue.pop(0)
    monkeypatch.setattr(runpod.requests, "request", request)
    return SimpleNamespace(calls=calls, queue=queue)


def test_call_uses_v2_base_and_bearer_key(api):
    api.queue.append(response(200, {"ok": True}))
    assert runpod.call("GET", "/pods/x") == {"ok": True}
    method, url, kw = api.calls[0]
    assert url == "https://api.runpod.io/v2/pods/x"
    assert kw["headers"]["Authorization"] == "Bearer k"


def test_call_backs_off_on_429_then_succeeds(api):
    api.queue += [response(429, headers={"Retry-After": "3"}), response(200, {"pods": []})]
    assert runpod.call("GET", "/pods") == {"pods": []}
    assert len(api.calls) == 2


def test_call_gives_up_after_attempts(api):
    api.queue += [response(429, {"title": "Rate limited"})] * 2
    with pytest.raises(requests.HTTPError, match="429 .* Rate limited"):
        runpod.call("GET", "/pods", attempts=2)


def test_call_error_carries_problem_detail_and_response(api):
    api.queue.append(response(422, {"title": "Invalid body", "detail": "gpu.id is required",
                                    "errors": ["gpu.id: missing"]}))
    with pytest.raises(requests.HTTPError) as exc:
        runpod.call("POST", "/pods")
    assert "gpu.id is required" in str(exc.value) and "gpu.id: missing" in str(exc.value)
    assert exc.value.response.status_code == 422


def test_call_does_not_retry_client_errors(api):
    api.queue.append(response(404))
    with pytest.raises(requests.HTTPError):
        runpod.call("DELETE", "/pods/gone")
    assert len(api.calls) == 1


def test_active_pods_follows_cursor(api):
    api.queue += [response(200, {"pods": [{"id": "a"}], "pagination": {"hasNextPage": True, "nextCursor": "c2"}}),
                  response(200, {"pods": [{"id": "b"}], "pagination": {"hasNextPage": False, "nextCursor": None}})]
    assert [p["id"] for p in runpod.active_pods()] == ["a", "b"]
    assert api.calls[1][2]["params"]["cursor"] == "c2"


def test_gpu_price_reads_nested_catalogue(api):
    api.queue.append(response(200, {"gpus": [{"id": "NVIDIA H100 80GB HBM3",
                                              "price": {"secure": 2.69, "community": 1.99}}]}))
    assert runpod.gpu_price("NVIDIA H100 80GB HBM3", "community") == 1.99
    api.queue.append(response(200, {"gpus": []}))
    assert runpod.gpu_price("nope") is None


def test_ssh_endpoint_prefers_direct_then_runtime_ports():
    assert runpod.ssh_endpoint({"ssh": {"direct": {"host": "203.0.113.5", "port": 10022}}}) == ("203.0.113.5", 10022)
    assert runpod.ssh_endpoint({"ssh": {"direct": None},
                                "runtime": {"ports": [{"private": 22, "public": 40000, "ip": "203.0.113.9", "type": "tcp"}]}}) == ("203.0.113.9", 40000)
    assert runpod.ssh_endpoint({"ssh": {"direct": None, "proxy": {"host": "ssh.runpod.io"}}, "runtime": None}) is None


def test_provision_sends_nested_v2_body(api, tmp_path):
    api.queue.append(response(201, {"id": "pod1"}))
    spec = runpod.ProvisionSpec(gpu="NVIDIA H200", count=2, disk_gb=150, cuda="13.0",
                                pubkey_path=str(tmp_path / "missing.pub"))
    assert runpod.provision_runpod(spec, name="t", start_script="echo hi", env={"A": "1"}) == "pod1"
    method, url, kw = api.calls[0]
    body = kw["json"]
    assert (method, url) == ("POST", "https://api.runpod.io/v2/pods")
    # A floor, never an exact set: v2 matches allowedCudaVersions exactly, and the H200s with
    # stock on 2026-10-30 all reported 13.2.
    assert body["gpu"] == {"id": "NVIDIA H200", "count": 2, "minCudaVersion": "13.0"}
    assert body["image"] == runpod.IMAGE and body["disk"] == 150 and body["cloud"] == "SECURE"
    assert body["cmd"] == ["bash", "-lc", "echo hi"]
    assert body["env"]["A"] == "1" and "dataCenterIds" not in body
    assert not {"imageName", "gpuTypeIds", "gpuCount", "containerDiskInGb", "dockerStartCmd", "countryCodes"} & body.keys()


def test_provision_passes_data_centre_ids_through(api, tmp_path):
    api.queue.append(response(201, {"id": "pod2"}))
    spec = runpod.ProvisionSpec(gpu="NVIDIA H200", data_centers="eu-se-1, EU-NL-1", cuda="",
                                pubkey_path=str(tmp_path / "missing.pub"))
    assert runpod.provision_runpod(spec, name="t", start_script="true") == "pod2"
    assert len(api.calls) == 1
    body = api.calls[0][2]["json"]
    assert body["dataCenterIds"] == ["EU-SE-1", "EU-NL-1"]
    assert "allowedCudaVersions" not in body["gpu"] and "minCudaVersion" not in body["gpu"]


def test_scheduled_create_makes_the_pod_delete_itself(api, tmp_path):
    api.queue.append(response(201, {"id": "pod3"}))
    spec = runpod.ProvisionSpec(terminate_at="2099-01-01T00:00:00Z", data_centers="EU-SE-1", cuda="",
                                pubkey_path=str(tmp_path / "missing.pub"))
    assert runpod.provision_runpod(spec, name="t", start_script="echo hi") == "pod3"
    body = api.calls[0][2]["json"]
    script = body["cmd"][-1]
    assert script.endswith("echo hi")
    assert "4070908800" in script and "$RUNPOD_POD_ID" in script and "$RUNPOD_API_KEY" in script
    assert "api.runpod.io/v2/pods/" in script and "-X DELETE" in script
    assert body["dataCenterIds"] == ["EU-SE-1"]


def test_scheduled_create_refuses_a_past_or_naive_deadline(api):
    for bad in ("2000-01-01T00:00:00Z", "2099-01-01T00:00:00"):
        with pytest.raises(ValueError):
            runpod.provision_runpod(runpod.ProvisionSpec(terminate_at=bad), name="t", start_script="x")
    assert not api.calls


def test_self_destruct_prefix_is_valid_shell():
    import subprocess
    script = runpod.self_destruct_prefix(4070908800) + "echo ok"
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_account_spend_reads_billing_and_running_burn(api):
    api.queue += [response(200, {"pods": [{"id": "a", "status": "RUNNING", "cost": 2.5},
                                          {"id": "b", "status": "EXITED", "cost": 9}],
                                 "pagination": {"hasNextPage": False}}),
                  response(200, {"records": [], "metadata": {"totals": {"totalAmount": 12.345}}})]
    assert runpod.account_spend() == {"spend_today_usd": 12.35, "spend_per_hr_usd": 2.5, "running_pods": 1}
    assert api.calls[1][1].endswith("/billing")
    assert api.calls[1][2]["params"] == {"lastN": 1, "bucketSize": "day"}


def test_a_cuda_list_is_read_as_its_floor(api, tmp_path):
    api.queue.append(response(201, {"id": "pod3"}))
    spec = runpod.ProvisionSpec(gpu="NVIDIA H200", cuda="13.2,12.10,13.0",
                                pubkey_path=str(tmp_path / "missing.pub"))
    runpod.provision_runpod(spec, name="t", start_script="true")
    assert api.calls[0][2]["json"]["gpu"]["minCudaVersion"] == "12.10"
