# ABOUTME: Shared digest-pinned HTTP fixture readiness for SWE-bench preparation and execution.
# ABOUTME: Verifies ownership, network binding and TLS before reusing the local grading fixture.
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import time
from urllib.request import urlopen
from omegaconf import OmegaConf


def ensure_fixture(cfg, receipt):
    """Start only our digest-pinned, bridge-bound HTTP fixture; reject config drift."""
    fixture = OmegaConf.to_container(cfg.httpbin_fixture)
    prefix = os.environ["USER_PREFIX"]
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]*", prefix)
    name = f"{prefix}-{fixture.pop('container_suffix')}-{receipt['instance_id']}"
    fixture["container_name"] = name
    owner = receipt["label"]
    assert fixture["url"] == "http://httpbin.org/" and fixture["bind_ip"] == "172.17.0.1"
    bind_ip = fixture["bind_ip"]
    assert "@sha256:" in fixture["image"]
    bridge = json.loads(subprocess.check_output(["docker", "network", "inspect", "bridge"]))[0]
    assert bridge["IPAM"]["Config"][0]["Gateway"] == bind_ip
    tls_dir = Path(fixture["tls_dir"])
    tls_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cert, key = tls_dir / "cert.pem", tls_dir / "key.pem"
    if not cert.exists() and not key.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
                        "-days", str(fixture["certificate_days"]), "-keyout", str(key), "-out", str(cert),
                        "-subj", "/CN=httpbin.org", "-addext",
                        f"subjectAltName=DNS:httpbin.org,DNS:www.httpbin.org,IP:{bind_ip}"],
                       check=True, capture_output=True, timeout=30)
        key.chmod(0o600)
    assert cert.exists() and key.exists(), "Incomplete HTTPBin TLS material"
    subprocess.run(["openssl", "x509", "-checkend", "86400", "-noout", "-in", str(cert)], check=True)
    fixture["certificate_path"] = str(cert)
    fixture["certificate_sha256"] = hashlib.sha256(cert.read_bytes()).hexdigest()
    found = subprocess.run(["docker", "inspect", name], capture_output=True)
    if found.returncode == 0:
        info = json.loads(found.stdout)[0]
        assert info["Config"]["Labels"].get("owner") == owner, "Fixture ownership mismatch"
        assert info["Config"]["Image"] == fixture["image"], "Fixture image drift"
        assert info["Config"]["Cmd"] == fixture["command"], "Fixture command drift"
        assert info["HostConfig"]["PortBindings"] == {f"{port}/tcp": [{"HostIp": bind_ip, "HostPort": str(port)}] for port in (80, 443)}
        assert f"{tls_dir}:/certs:ro" in info["HostConfig"]["Binds"]
        subprocess.run(["docker", "start", name], check=True, timeout=30)
    else:
        subprocess.run(["docker", "pull", fixture["image"]], check=True, timeout=180)
        subprocess.run(["docker", "run", "-d", "--name", name, "--label", f"owner={owner}",
                        "--restart", "unless-stopped", "--cpus", str(fixture["cpus"]),
                        "--memory", fixture["memory"], "-p", f"{bind_ip}:80:80", "-p", f"{bind_ip}:443:443",
                        "-v", f"{tls_dir}:/certs:ro",
                        fixture["image"], *fixture["command"]], check=True, timeout=60)
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(f"http://{bind_ip}/status/200", timeout=3) as response:
                assert response.status == 200
            with urlopen(f"https://{bind_ip}/status/200", timeout=3,
                         context=ssl.create_default_context(cafile=str(cert))) as response:
                assert response.status == 200
            return fixture
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError("HTTP fixture failed its health check")
            time.sleep(1)
