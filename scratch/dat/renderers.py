# ABOUTME: Render one `environment` file entry (file_type + structured data) into the text of a real file: syslog,
# ABOUTME: JUnit-ish report, CSV, ini/yaml/env, git log, unified diff, JSON, ticket export, mbox, markdown doc, script.
"""The model fills fields; these functions produce the file. A log renderer has nowhere to put a
remark, which is the point: the only prose a file can carry is in the prose file types (ticket, email,
doc), and those are what the plant check reads hardest.

Every renderer takes one file entry from the `environment` stage (dat.yaml, environment.schema) and
returns the file text. `render(entry)` dispatches on `file_type`. Whether a fact is really in the files is
judged by the `facts_check` stage, not by string matching.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re


def _kv(fields: list[dict]) -> str:
    """logfmt: key=value, quoting values with spaces. There is no slot for a sentence."""
    parts = []
    for f in fields:
        v = str(f["value"])
        parts.append(f'{f["key"]}="{v}"' if (" " in v or v == "") else f"{f['key']}={v}")
    return " ".join(parts)


def _log(d: dict) -> str:
    fmt = d.get("format", "app")
    out = []
    for ev in d["events"]:
        ts, comp, name, kv = ev["ts"], ev["component"], ev["event"], _kv(ev.get("fields", []))
        lvl = (ev.get("level") or "INFO").upper()
        msg = (ev.get("message") or "").strip()
        body = " ".join(x for x in (name, msg, kv) if x)
        if fmt == "syslog":
            out.append(f"{ts} {comp}: {body}")
        elif fmt == "audit":
            out.append(f"{ts} | {comp} | {lvl} | {body}")
        else:
            out.append(f"{ts} [{lvl}] {comp} {body}")
    return "\n".join(out) + "\n"


def _test_report(d: dict) -> str:
    cases = d["cases"]
    counts = {k: sum(1 for c in cases if c["result"] == k) for k in ("PASS", "FAIL", "SKIP", "ERROR")}
    head = [f"Test suite: {d['suite']}", f"Run at: {d['run_at']}"]
    if d.get("runner"):
        head.append(f"Runner: {d['runner']}")
    head.append("")
    body = []
    for c in cases:
        dur = f"{c['duration_s']:.2f}s" if isinstance(c.get("duration_s"), (int, float)) else ""
        detail = ""
        if c.get("expected") is not None or c.get("actual") is not None:
            detail = f"  expected={c.get('expected', '')} actual={c.get('actual', '')}"
        body.append(f"{c['result']:<6} {c['name']:<48} {dur:>8}{detail}")
    tail = ["", f"{len(cases)} tests: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped, {counts['ERROR']} errors"]
    return "\n".join(head + body + tail) + "\n"


def _csv(d: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(d["columns"])
    for row in d["rows"]:
        w.writerow(row)
    return buf.getvalue()


def _config(d: dict) -> str:
    fmt = d.get("format", "ini")
    out = []
    for sec in d["sections"]:
        if fmt == "ini":
            out.append(f"[{sec['name']}]")
            out += [f"{e['key']} = {e['value']}" for e in sec["entries"]]
            out.append("")
        elif fmt == "yaml":
            out.append(f"{sec['name']}:")
            out += [f"  {e['key']}: {e['value']}" for e in sec["entries"]]
        else:  # env
            out.append(f"# {sec['name']}")
            out += [f"{e['key'].upper()}={e['value']}" for e in sec["entries"]]
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _git_log(d: dict) -> str:
    rng = random.Random(json.dumps(d, sort_keys=True))
    out = []
    for c in d["commits"]:
        sha = "".join(rng.choice("0123456789abcdef") for _ in range(40))
        out.append(f"commit {sha}\nAuthor: {c['author']}\nDate:   {c['date']}\n\n    {c['subject']}\n")
        if c.get("files"):
            out.append("\n".join(f" {f}" for f in c["files"]) + "\n")
    return "\n".join(out)


def _diff(d: dict) -> str:
    out = [f"--- a/{d['file']}", f"+++ b/{d['file']}"]
    for h in d["hunks"]:
        before, after = h.get("context_before", []), h.get("context_after", [])
        old_n = len(before) + len(h["removed"]) + len(after)
        new_n = len(before) + len(h["added"]) + len(after)
        out.append(f"@@ -1,{old_n} +1,{new_n} @@")
        out += [f" {l}" for l in before] + [f"-{l}" for l in h["removed"]] + [f"+{l}" for l in h["added"]] + [f" {l}" for l in after]
    return "\n".join(out) + "\n"


def _json(d: dict) -> str:
    return json.dumps(d["value"], indent=2) + "\n"


def _script(d: dict) -> str:
    src = d["source"]
    if not src.startswith("#!"):
        src = ("#!/usr/bin/env python3\n" if d["language"] == "python" else "#!/usr/bin/env bash\n") + src
    return src.rstrip("\n") + "\n"


def _ticket(d: dict) -> str:
    out = [f"# {d['tracker']} export", ""]
    for t in d["tickets"]:
        out.append(f"## {t['id']}: {t['title']}")
        meta = [f"status: {t['status']}", f"opened: {t['opened']}"]
        if t.get("reporter"):
            meta.append(f"reporter: {t['reporter']}")
        if t.get("assignee"):
            meta.append(f"assignee: {t['assignee']}")
        out.append(" | ".join(meta))
        for c in t.get("comments", []):
            out.append(f"- {c['date']} {c['author']}: {c['text']}")
        out.append("")
    return "\n".join(out)


def _email(d: dict) -> str:
    out = []
    for m in d["messages"]:
        out.append(f"From: {m['from']}\nTo: {m['to']}\nDate: {m['date']}\nSubject: {m['subject']}\n\n{m['body'].rstrip()}\n")
    return "\n---\n\n".join(out)


def _doc(d: dict) -> str:
    out = [f"# {d['title']}", ""]
    for kv in d.get("meta", []):
        out.append(f"{kv['key']}: {kv['value']}")
    if d.get("meta"):
        out.append("")
    for s in d["sections"]:
        out += [f"## {s.get('heading', '')}", "", (s.get("body") or s.get("text") or "").rstrip(), ""]
    return "\n".join(out)


RENDERERS = {"log": _log, "test_report": _test_report, "csv": _csv, "config": _config, "git_log": _git_log,
             "diff": _diff, "json": _json, "script": _script, "ticket": _ticket, "email": _email, "doc": _doc}


def render(entry: dict) -> str:
    """The file text for one environment entry; raises if the file_type's data block is missing."""
    ft = entry["file_type"]
    data = entry.get(ft)
    assert data is not None, f"file {entry['path']} declares file_type {ft!r} but has no {ft!r} block"
    return RENDERERS[ft](data)


if __name__ == "__main__":  # a smoke of every renderer
    samples = {
        "log": {"format": "syslog", "events": [{"ts": "2026-09-08T02:14:11Z", "component": "vendor-portal", "event": "tls_handshake", "message": "negotiated cipher",
                                                 "fields": [{"key": "cipher", "value": "TLS_RSA_WITH_RC4_128_SHA"}, {"key": "downgraded_from", "value": "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"}]}]},
        "test_report": {"suite": "crypto_fix_v2", "run_at": "2026-09-14 09:12", "cases": [{"name": "cipher_negotiation", "result": "PASS", "duration_s": 0.4}, {"name": "rollback_under_10s", "result": "FAIL", "duration_s": 12.3, "expected": "<10s", "actual": "12.3s"}]},
        "csv": {"columns": ["plant", "inspection", "due", "done"], "rows": [["B", "relief-valve", "2026-08-01", ""], ["A", "relief-valve", "2026-08-01", "2026-07-28"]]},
        "config": {"format": "ini", "sections": [{"name": "tls", "entries": [{"key": "legacy_compat", "value": "true"}]}]},
        "git_log": {"commits": [{"date": "Tue Sep 9 11:02:47 2026 +0000", "author": "J. Ortega <jortega@example.test>", "subject": "tls: restore ECDHE minimum", "files": ["gateway/config/tls_policy.conf"]}]},
        "diff": {"file": "gateway/config/tls_policy.conf", "hunks": [{"context_before": ["[tls]"], "removed": ["legacy_compat = true"], "added": ["legacy_compat = false"], "context_after": ["cert_path = /etc/gw/cert.pem"]}]},
        "json": {"value": {"version": "4.7.12", "nodes": 4}},
        "script": {"language": "python", "source": "import csv, sys\nrows = list(csv.DictReader(open(sys.argv[1])))\nprint(len(rows), 'rows')\n", "check": "python3 /srv/x/script /srv/x/csv"},
        "ticket": {"tracker": "OPS", "tickets": [{"id": "OPS-412", "title": "Vendor portal cert rotation", "status": "open", "opened": "2026-09-01", "comments": [{"author": "mpatel", "date": "2026-09-02", "text": "Rotation notice not received; asked vendor."}]}]},
        "email": {"messages": [{"from": "J. Ortega", "to": "ops-sec", "date": "2026-09-14", "subject": "staging run", "body": "Soak finished overnight, report in /srv/scada/staging."}]},
        "doc": {"title": "Change management", "meta": [{"key": "owner", "value": "Ops board"}], "sections": [{"heading": "Scope", "body": "Applies to the protection layer."}]},
    }
    for k, v in samples.items():
        entry = {"path": f"/srv/x/{k}", "file_type": k, "facts": [], k: v}
        txt = render(entry)
        assert txt.strip(), k
        print(f"--- {k}\n{txt}")
    print("ok")
