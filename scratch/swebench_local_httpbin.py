# ABOUTME: Run the pinned official grader with a documented local HTTP fixture for requests tasks.
# ABOUTME: Hosts and CA configuration change; task patches, test patches, and scoring remain upstream.
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import ssl
from urllib.request import urlopen

import swebench.harness.run_evaluation as harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True,
                        help="Resolved JSON request generated from the CPU YAML config")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert importlib.metadata.version("swebench") == "4.1.0"
    fixture = request["fixture"]
    assert "@sha256:" in fixture["image"], "Fixture must have an immutable image digest"
    assert fixture["url"] == "http://httpbin.org/" and fixture["bind_ip"] == "172.17.0.1"
    certificate = Path(fixture["certificate_path"]).read_bytes()
    assert hashlib.sha256(certificate).hexdigest() == fixture["certificate_sha256"]
    info = json.loads(subprocess.check_output(["docker", "inspect", fixture["container_name"]]))[0]
    assert info["State"]["Running"], "HTTP fixture is not running"
    assert info["Config"]["Image"] == fixture["image"], "HTTP fixture image changed"
    assert info["Config"]["Cmd"] == fixture["command"], "HTTP fixture command changed"
    with urlopen("http://" + fixture["bind_ip"] + "/status/200", timeout=10) as response:
        assert response.status == 200
    with urlopen("https://" + fixture["bind_ip"] + "/status/200", timeout=10,
                 context=ssl.create_default_context(cafile=fixture["certificate_path"])) as response:
        assert response.status == 200
    make_spec = harness.make_test_spec

    def with_fixture(instance, *args, **kwargs):
        spec = make_spec(instance, *args, **kwargs)
        if spec.repo == "psf/requests":
            original = list(spec.eval_script_list)
            marker = ": '>>>>> Start Test Output'"
            assert original.count(marker) == 1, "Pinned harness test marker changed"
            # PreparedRequest/Session.send bypasses REQUESTS_CA_BUNDLE in these historical
            # requests versions. Add the public fixture CA to its existing trust store too.
            install_ca = (
                "from pathlib import Path; import requests.certs, hashlib, json; "
                "p=Path(requests.certs.where()); before=p.read_bytes(); "
                "p.write_bytes(before+b'\\n'+Path('/tmp/lasr-httpbin-ca.pem').read_bytes()); "
                "print('HTTPBIN_CA_INSTALL',json.dumps({'path':str(p),"
                "'original_sha256':hashlib.sha256(before).hexdigest(),"
                "'updated_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
            )
            original.insert(original.index(marker), "python -c " + shlex.quote(install_ca))
            spec.eval_script_list = [
                "printf '\\n172.17.0.1 httpbin.org www.httpbin.org\\n' >> /etc/hosts",
                "printf '%s' " + shlex.quote(base64.b64encode(certificate).decode()) + " | base64 -d > /tmp/lasr-httpbin-ca.pem",
                "cat /etc/ssl/certs/ca-certificates.crt /tmp/lasr-httpbin-ca.pem > /tmp/lasr-httpbin-bundle.pem",
                "export REQUESTS_CA_BUNDLE=/tmp/lasr-httpbin-bundle.pem CURL_CA_BUNDLE=/tmp/lasr-httpbin-bundle.pem SSL_CERT_FILE=/tmp/lasr-httpbin-bundle.pem",
                "export HTTPBIN_URL=" + shlex.quote(fixture["url"]), *original]
        return spec

    harness.make_test_spec = with_fixture
    Path("fixture_provenance.json").write_text(json.dumps({
        "protocol_deviation": "requests tests resolve httpbin.org locally and trust its recorded CA; original test and solution patches unchanged",
        "fixture": fixture, "harness_version": importlib.metadata.version("swebench"),
        "request": request["harness"],
    }, indent=2) + "\n")
    harness.main(**request["harness"])


if __name__ == "__main__":
    main()
