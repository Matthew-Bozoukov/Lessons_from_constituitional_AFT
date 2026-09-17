# ABOUTME: daa2 helpers -- the docker sandbox with a frozen scenario clock, the ODCV tool schemas and tool-call
# ABOUTME: builders, numbered-sentence edit lists, and the DA source loader that agentify2.py is built on.
"""What agentify2 stands on, and nothing else.

    sandbox     Sandbox runs each command for real in a python:3.12-slim + libfaketime container whose clock is
                the scenario's date; check_docker / ensure_image guard and build it; scenario_date reads a date
                out of a row.
    tool calls  TOOLS (bash + task_complete, the ODCV schemas), bash_call / complete_call / tool_result.
    edits       paragraphs_sentences / numbered / line_start_numbers / apply_sentence_edits: the model returns
                {"edits", "inserts"} over numbered sentences and code reassembles the text, untouched sentences
                verbatim; reuse_ratio measures how much survived.
    source      load_source pulls a DA corpus's dataset.jsonl from the Hub at a pinned revision.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path


IMAGE = "daa-sandbox"


# Byte-identical to ODCV-Bench's tool schemas (mission_executor/agent_main.py) and to every
# dat row: the model must see at train time exactly what the eval serves, ungrammatical
# description included.
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Executes shell commands. Put the command in the 'command' field. "
                       "The command should be a valid bash command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "task_complete", "description": "To mark the task as completion.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]}}},
]


# The container has no network and nothing of the host's; the denylist only stops a command
# from wasting the row (or the daemon's time).
UNSAFE = re.compile(r"(?<![\w-])(sudo|su|curl|wget|ssh|scp|rsync|nc|ncat|telnet|pip|pip3|npm|apt|apt-get|"
                    r"mkfs|shutdown|reboot|halt|crontab|systemctl)(?![\w-])|rm\s+-[a-z]*r[a-z]*f?\s+/(?![\w])|dd\s+if=")


def command_skeleton(cmd: str) -> str:
    """The command with heredoc bodies and quoted strings blanked, so a check judges the
    shell structure and not the prose it carries."""
    cmd = re.sub(r"<<-?\s*'?(\w+)'?\n.*?\n\1(?=\n|$)", "<<HEREDOC", cmd, flags=re.S)
    cmd = re.sub(r"'[^']*'", "''", cmd)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', cmd)


def paragraphs_sentences(text: str) -> list[list[str]]:
    """Every sentence of every paragraph, in order, nothing dropped (the deliberation edit list
    numbers these; `sentences()` below is the reuse metric's filtered view)."""
    out = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", para.strip()) if x.strip()]
        if sents:
            out.append(sents)
    return out


def line_start_numbers(text: str) -> set[int]:
    """Sentence numbers (1-based, across paragraphs, the numbering of `paragraphs_sentences`) whose
    sentence begins a line in the original: list items and other single-newline breaks, which
    `apply_sentence_edits` must put back as line breaks rather than spaces."""
    k, out = 0, set()
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        p, pos = para.strip(), 0
        for x in re.split(r"(?<=[.!?])\s+", p):
            if not x.strip():
                continue
            k += 1
            i = p.find(x.strip(), pos)
            if i <= 0 or p[i - 1] == "\n":
                out.add(k)
            pos = max(pos, i + len(x.strip()))
    return out


def numbered(paras: list[list[str]]) -> str:
    k, lines = 0, []
    for para in paras:
        for x in para:
            k += 1
            lines.append(f"[{k}] {x}")
        lines.append("")
    return "\n".join(lines).strip()


def apply_sentence_edits(paras: list[list[str]], edits: list, inserts: list,
                         line_starts: set[int] | None = None) -> str:
    """Reassemble the deliberation from the model's edit list (rule: untouched sentences verbatim).
    A sentence that began a line in the original (`line_start_numbers`) still begins a line, so
    lists keep their shape; an inserted sentence follows the sentence it was inserted after."""
    untag = lambda s: re.sub(r"^\s*\[\d+\]\s*", "", str(s or "")).strip()   # a model sometimes echoes the "[n] " numbering
    repl: dict[int, str] = {}
    for e in edits or []:
        try:
            repl[int(e["n"])] = untag(e.get("new"))
        except (KeyError, TypeError, ValueError):
            continue
    ins: dict[int, list[str]] = {}
    for e in inserts or []:
        try:
            ins.setdefault(int(e["after"]), []).append(untag(e.get("text")))
        except (KeyError, TypeError, ValueError):
            continue
    k, out_paras = 0, []
    starts = line_starts or set()
    head = [(t, False) for t in ins.get(0, []) if t]
    for para in paras:
        cur = list(head); head = []
        for x in para:
            k += 1
            y = repl.get(k, x)
            if y:
                cur.append((y, k in starts))
            cur += [(t, False) for t in ins.get(k, []) if t]
        if cur:
            text = cur[0][0]
            for t, nl in cur[1:]:
                text += ("\n" if nl else " ") + t
            out_paras.append(text)
    return "\n\n".join(out_paras)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 20]


def fold(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def reuse_ratio(original: str, adapted: str) -> float:
    """Fraction of the original's sentences that survive verbatim (case/space-folded)."""
    src = sentences(original)
    if not src:
        return 0.0
    f = fold(adapted)
    return sum(1 for s in src if fold(s) in f) / len(src)


def tool_result(stdout: str, returncode: int = 0, stderr: str = "") -> str:
    return json.dumps({"stdout": stdout, "stderr": stderr, "role": "tool", "returncode": returncode})


def bash_call(command: str) -> dict:
    return {"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}


def complete_call(reason: str) -> dict:
    return {"type": "function", "function": {"name": "task_complete", "arguments": {"reason": reason}}}


def scenario_date(*texts: str) -> dt.datetime:
    """The sandbox 'now': the day after the latest ISO date the scenario mentions; else a
    fixed day. Frozen by libfaketime, so scripts calling today() agree with the files."""
    found = []
    for t in texts:
        for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t or ""):
            try:
                found.append(dt.date(int(y), int(m), int(d)))
            except ValueError:
                pass
    day = (max(found) + dt.timedelta(days=1)) if found else dt.date(2024, 6, 12)
    return dt.datetime.combine(day, dt.time(9, 30))


def docker(*args: str, stdin: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout)


def ensure_image() -> None:
    res = docker("image", "inspect", IMAGE)
    if res.returncode == 0:
        return
    print(f">>> building {IMAGE} (python:3.12-slim + libfaketime): inspect said {res.stderr.decode()[:120]!r}", flush=True)
    dockerfile = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends libfaketime coreutils "
        "&& rm -rf /var/lib/apt/lists/* "
        "&& ln -s \"$(find /usr/lib -name 'libfaketime.so.1' | head -1)\" /usr/local/lib/libfaketime.so.1\n"
        "CMD [\"sleep\", \"infinity\"]\n")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "Dockerfile").write_text(dockerfile)
        res = docker("build", "-q", "-t", IMAGE, d, timeout=900)
    if res.returncode:
        raise SystemExit(f"docker build failed:\n{res.stderr.decode()[-2000:]}")


def check_docker() -> None:
    res = docker("info", "--format", "{{.ServerVersion}}")
    if res.returncode:
        raise SystemExit("Docker daemon not reachable: start Docker Desktop (`open -a Docker`) and retry. "
                         "The sandbox is a container per row; there is no host fallback.")


class Sandbox:
    """A throwaway container holding the row's files at their real absolute paths, mtimes set
    a day before the frozen scenario clock, so `ls`, `date` and `today()` agree with the files
    and nothing of the host (clock, user, paths) can reach a transcript."""

    def __init__(self, files: dict[str, str], now: dt.datetime, timeout: int = 30):
        self.timeout = timeout
        self.now = now
        self.name = f"daa_{uuid.uuid4().hex[:12]}"
        # the kernel's clock as `ls -l` prints it (time form for recent files, year form otherwise)
        # and as ISO: a file created by the agent shows one of these until the settle touches it
        host = dt.datetime.now(dt.timezone.utc)
        self._host_stamps = re.compile(rf"{host.strftime('%b')} +{host.day} +(?:\d\d:\d\d|{host.year})|{host.strftime('%Y-%m-%d')}")
        settled = now - dt.timedelta(hours=1)
        self._settled = settled.strftime("%b %e %H:%M").replace("  ", " ")
        res = docker("run", "-d", "--rm", "--network", "none", "--name", self.name, IMAGE, "sleep", "1800")
        if res.returncode:
            raise RuntimeError(f"docker run failed: {res.stderr.decode()[-500:]}")
        buf = io.BytesIO()
        mtime = int((now - dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for p, c in files.items():
                data = c.encode("utf-8")
                info = tarfile.TarInfo(p.lstrip("/"))
                info.size = len(data)
                info.mtime = mtime
                info.mode = 0o755 if (p.endswith((".sh", ".py")) or c.startswith("#!")) else 0o644
                tar.addfile(info, io.BytesIO(data))
        res = docker("cp", "-", f"{self.name}:/", stdin=buf.getvalue())
        if res.returncode:
            self.close()
            raise RuntimeError(f"docker cp failed: {res.stderr.decode()[-500:]}")
        # the directories the tar created carry the host's clock; align them
        dirs: set[str] = set()
        for p in files:
            d = os.path.dirname(p)
            while d and d != "/":
                dirs.add(d)
                d = os.path.dirname(d)
        dirs = sorted(dirs)
        stamp = (now - dt.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        self._exec(f"touch -d '{stamp}' / {' '.join(dirs)}", faketime=False)

    # libfaketime cannot change the mtime the kernel stamps on a file the agent creates, so after
    # every command anything newer than the scenario clock (i.e. stamped with the host's real 2026
    # clock) is touched back to it. Only non-system top-level dirs are scanned; a new top-level
    # dir the agent makes (`mkdir /policy`) is included by construction.
    _SYSTEM_DIRS = "/bin|/boot|/dev|/etc|/lib|/lib32|/lib64|/libx32|/proc|/run|/sbin|/sys|/usr"

    def _settle_clock(self) -> None:
        stamp = (self.now - dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")   # an hour ago: `ls` shows the time form
        self._exec("for d in /*; do case $d in " + self._SYSTEM_DIRS + ") ;; *) echo \"$d\";; esac; done | "
                   f"xargs -r -I DIR find DIR -xdev -newermt '{stamp}' -exec touch -h -d '{stamp}' {{}} + 2>/dev/null; "
                   f"touch -d '{stamp}' /; true", faketime=False)

    def _exec(self, command: str, faketime: bool = True) -> dict:
        env = ["-e", "HOME=/root", "-e", "LANG=C.UTF-8", "-e", "PYTHONWARNINGS=ignore",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "TZ=UTC"]
        if faketime:
            # NO_FAKE_STAT: the clock is faked, file mtimes are not (else `ls` dates every file "now")
            env += ["-e", "LD_PRELOAD=/usr/local/lib/libfaketime.so.1", "-e", "NO_FAKE_STAT=1",
                    "-e", f"FAKETIME={self.now.strftime('%Y-%m-%d %H:%M:%S')}"]
        try:
            res = docker("exec", *env, "-w", "/", self.name, "timeout", str(self.timeout), "bash", "-c", command,
                         timeout=self.timeout + 15)
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"timed out after {self.timeout}s", "returncode": 124}
        stderr = res.stderr.decode("utf-8", "replace")
        if res.returncode == 124 and not stderr:
            stderr = f"timed out after {self.timeout}s"
        return {"stdout": res.stdout.decode("utf-8", "replace"), "stderr": stderr, "returncode": res.returncode}

    _LS_LINE = re.compile(r"^(?:[-dlcbps][rwxsStT-]{9}[.+@]?\s|total \d|\s*(?:Modify|Change|Access|Birth):)")

    def run(self, command: str) -> dict:
        if UNSAFE.search(command_skeleton(command)):
            return {"stdout": "", "stderr": "refused by the sandbox denylist", "returncode": 126, "refused": True}
        res = self._exec(command)
        self._settle_clock()
        res["stdout"] = "\n".join(self._host_stamps.sub(self._settled, line) if self._LS_LINE.match(line) else line
                                  for line in res["stdout"].split("\n"))
        return res

    def close(self) -> None:
        docker("rm", "-f", self.name)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_source(repo: str, revision: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "dataset.jsonl", repo_type="dataset", revision=revision,
                           token=os.environ.get("HF_TOKEN"))
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        msgs = r["messages"]
        out.append({
            "scenario_id": r["metadata"]["scenario_id"],
            "source_repo": repo, "source_revision": revision,
            "system": next(m["content"] for m in msgs if m["role"] == "system"),
            "user": next(m["content"] for m in msgs if m["role"] == "user"),
            "reasoning": [m for m in msgs if m["role"] == "assistant"][-1].get("reasoning_content") or "",
            "answer": [m for m in msgs if m["role"] == "assistant"][-1]["content"],
            "metadata": r["metadata"],
        })
    return out
