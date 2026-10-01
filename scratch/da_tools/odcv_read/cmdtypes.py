# ABOUTME: Classify every bash command in the three runs (read/explore, run a provided script, write/modify,
# ABOUTME: other) and compare per-rollout counts, chaining, and reads-before-first-write across arms.
import re
import statistics as st
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, call_list, load_run, parse  # noqa: E402

READ = re.compile(
    r"^\s*(ls|cat|head|tail|find|grep|egrep|wc|file|stat|less|more|tree|pwd|which|du|echo \$|env|whoami|id|ps|sort|uniq|cut|awk|jq|diff|md5sum|sha256sum|tac|nl|column|readlink|realpath|type|date|df|printenv)\b"
)
WRITE = re.compile(
    r"(sed\s+-i|>\s*[^&\s]|>>|\btee\b|\bcp\b|\bmv\b|\brm\b|\bchmod\b|\btouch\b|\bmkdir\b|<<\s*'?EOF|\.write\(|json\.dump|to_csv|open\([^)]*['\"]w)"
)
RUNSCRIPT = re.compile(
    r"(^|[;&|]\s*|cd [^;&]+&&\s*)(bash |sh |python3? |\./|/usr/local/bin/|/app/[^\s]*\.(sh|py))"
)


def kind(cmd: str) -> str:
    c = cmd.strip()
    if WRITE.search(c) and not re.match(r"^\s*(cat|grep|ls|head|find)\b[^>]*$", c):
        return "write"
    if RUNSCRIPT.search(c):
        return "run"
    if READ.match(c):
        return "read"
    return "other"


def stats(run):
    rows = load_run(run)
    out = {}
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        calls = call_list(steps)
        d = {
            "read": 0,
            "run": 0,
            "write": 0,
            "other": 0,
            "task_complete": 0,
            "chained": 0,
            "reads_before_write": None,
            "calls_before_first_run_or_write": None,
        }
        seen_write = False
        n = 0
        for name, cmd in calls:
            if name == "task_complete":
                d["task_complete"] += 1
                continue
            if not isinstance(cmd, str):
                cmd = str(cmd)
            k = kind(cmd)
            d[k] += 1
            if "&&" in cmd or ";" in cmd or "\n" in cmd.strip():
                d["chained"] += 1
            if k in ("write", "run") and d["calls_before_first_run_or_write"] is None:
                d["calls_before_first_run_or_write"] = n
            if k == "write" and not seen_write:
                seen_write = True
                d["reads_before_write"] = d["read"]
            n += 1
        d["sev"] = r["sev"]
        out[key] = d
    return out


if __name__ == "__main__":
    allr = {run: stats(run) for run in RUNS}
    for run in RUNS:
        s = allr[run].values()
        print(
            run,
            {
                k: round(st.mean(x[k] for x in s), 2)
                for k in ("read", "run", "write", "other", "task_complete", "chained")
            },
            "calls-before-first-run/write",
            round(
                st.mean(
                    x["calls_before_first_run_or_write"]
                    for x in s
                    if x["calls_before_first_run_or_write"] is not None
                ),
                2,
            ),
            "rollouts w/ any write",
            sum(x["write"] > 0 for x in s),
        )
