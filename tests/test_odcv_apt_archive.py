# ABOUTME: The bullseye apt-archive rewrite the ODCV workspace builder applies to scenario
# ABOUTME: Dockerfiles: bullseye bases get the fix after FROM, other bases are untouched.

from __future__ import annotations

from src.eval.misalignment.odcv.odcv_rollout import APT_ARCHIVE_FIX, pin_apt_archive

BULLSEYE = (
    "FROM debian:bullseye-slim\n"
    "RUN apt-get update && apt-get install -y python3 python3-pip bc jq && rm -rf /var/lib/apt/lists/*\n"
    "COPY . /app\n"
)
SLIM = "FROM python:3.13-slim\nRUN apt-get update && apt-get install -y bc jq\n"


def test_bullseye_dockerfile_gets_the_archive_fix_right_after_from():
    out = pin_apt_archive(BULLSEYE)
    lines = out.splitlines(keepends=True)
    assert lines[0] == "FROM debian:bullseye-slim\n"
    assert "".join(lines[1:3]) == APT_ARCHIVE_FIX
    assert out.endswith(BULLSEYE.split("\n", 1)[1]), "everything after FROM is preserved"
    assert "archive.debian.org/debian " in out and "/bullseye-security/d" in out


def test_non_bullseye_dockerfile_is_untouched():
    assert pin_apt_archive(SLIM) == SLIM
    assert pin_apt_archive("") == ""
