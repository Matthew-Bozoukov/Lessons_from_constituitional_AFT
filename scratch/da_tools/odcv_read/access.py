# ABOUTME: List every "I don't have access / no data was provided / can't look it up" claim per arm with context,
# ABOUTME: plus whether the rollout had explored the filesystem (ls/find/cat of /app) BEFORE the claim.
import re
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import load_run, parse  # noqa: E402

PAT = re.compile(
    r"(I )?(don't|do not) have (any )?(access|a way)[^.]{0,80}|no access to[^.]{0,60}|can't look (it|them|that) up|have no way to[^.]{0,60}|no (survey |patient |actual )?data (was|has been) (provided|attached|shared)|wasn't provided|not provided with",
    re.I,
)
run = sys.argv[1]
rows = load_run(run)
n = 0
for key, r in sorted(rows.items()):
    steps = parse(r["path"].read_text())
    ncalls_before = 0
    explored = False
    for s in steps:
        if s["role"] == "assistant":
            for src, text in (("text", s["content"]), ("reason", s["reason"])):
                if not text or text == "None":
                    continue
                m = PAT.search(text)
                if m:
                    n += 1
                    a = max(0, m.start() - 100)
                    print(
                        f"{key[0][:4]} {key[1]} p{key[2]} sev={r['sev']} calls_before={ncalls_before} explored={explored} [{src}] ...{text[a : m.end() + 60]}...".replace(
                            "\n", " "
                        )
                    )
                    break
            for c in s["calls"]:
                ncalls_before += 1
                args = str(c.get("function", {}).get("arguments", ""))
                if re.search(r"\b(ls|find)\b", args):
                    explored = True
print("instances", n)
