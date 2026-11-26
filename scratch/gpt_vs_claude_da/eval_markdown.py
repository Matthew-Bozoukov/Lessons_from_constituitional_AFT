# ABOUTME: Markdown expressed by the Claude-DA vs GPT-DA trained Qwen3.6 adapters in their published eval rollouts
# ABOUTME: (ODCV messages_record.txt, Agentic Misalignment sample_*.md, MASK CSVs), split by reasoning vs visible output.
import ast
import csv
import datetime as dt
import json
import os
import re
import statistics as st
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

from scratch.gpt_vs_claude_da.stats import MD  # noqa: E402

warnings.filterwarnings("ignore")
csv.field_size_limit(10**9)
EVALS = Path("data/gpt_vs_claude_da/evals")
RUNS = {
    "odcv": {"claude": "2026-10-06-odcv-qwen36-0-da-15", "gpt": "2026-11-04-odcv-qwen36-0-da-gpt-15"},
    "am": {"claude": "2026-10-08-am-qwen36-0-da-15", "gpt": "2026-11-04-am-qwen36-0-da-gpt-15"},
    "mask": {"claude": "2026-11-03-mask-qwen36-0-da-15", "gpt": "2026-11-04-mask-qwen36-0-da-gpt-15"},
}
STRUCT_KEYS = ("heading (#)", "bold-line pseudo-heading", "bullet list", "numbered list", "table")
STRUCT_LINE = re.compile(r"(?m)^\s*(?:#{1,6}\s|[-*\u2022]\s+\S|\d+[.)]\s+\S|\|.*\|\s*$)")


def odcv_units(path: Path) -> dict[str, list[str]]:
    """Per rollout: reasoning texts, visible assistant texts, and the task_complete summary."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {"reasoning": [], "visible": [], "final_summary": []}
    for step in re.split(r"(?m)^== Step \d+ ==\n", text):
        if not step.startswith("role: assistant"):
            continue
        m = re.search(r"(?s)^content: (.*?)(?=\nreason: |\ncall: |\Z)", step, re.M)
        if m and m.group(1).strip() not in ("None", ""):
            out["visible"].append(m.group(1))
        m = re.search(r"(?s)^reason: (.*?)(?=\ncall: |\Z)", step, re.M)
        if m:
            out["reasoning"].append(m.group(1))
        m = re.search(r"(?s)^call: (\[.*\])\s*$", step, re.M)
        if m:
            try:
                for c in ast.literal_eval(m.group(1)):
                    if c["function"]["name"] == "task_complete":
                        out["final_summary"].append(json.loads(c["function"]["arguments"]).get("reason", ""))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return out


def am_units(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rea = re.search(r"(?s)## Agent reasoning \(out of band\)\s*```\n(.*?)\n```", text)
    resp = re.search(r"(?s)## Agent response \(inline reasoning, if any, \+ action\)\s*```\n(.*?)\n```", text)
    unesc = lambda s: s.replace("\n", "\n")  # email bodies are JSON-escaped  # noqa: E731
    return {"reasoning": [rea.group(1)] if rea else [], "visible": [unesc(resp.group(1))] if resp else []}


def mask_units(repo: str) -> list[dict[str, list[str]]]:
    info = HfApi().dataset_info(repo)
    col = "(System Prompt + User Prompt)_run1"
    units = []
    for s in info.siblings:
        if re.match(r"results/pass\d/.+_evaluated\.csv$", s.rfilename):
            p = hf_hub_download(repo, s.rfilename, repo_type="dataset", revision=info.sha)
            for r in csv.DictReader(open(p, encoding="utf-8")):
                units.append({"reasoning": [r[f"reasoning{col}"] or ""], "visible": [r[f"generation{col}"] or ""]})
    return units


def summarise(units: list[dict[str, list[str]]], channel: str) -> dict:
    texts = ["\n".join(u.get(channel, [])) for u in units]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return {"n_with_text": 0}
    n = len(texts)
    chars = sum(map(len, texts))
    pct = lambda f: round(100 * sum(1 for t in texts if f(t)) / n, 1)  # noqa: E731
    return {
        "n_with_text": n,
        "median_chars": int(st.median(map(len, texts))),
        "any_markdown_pct": pct(lambda t: any(re.search(p, t) for p in MD.values())),
        "structural_pct": pct(lambda t: any(re.search(MD[k], t) for k in STRUCT_KEYS)),
        **{f"{k}_pct": pct(lambda t, p=MD[k]: re.search(p, t)) for k in
           ("heading (#)", "bullet list", "numbered list", "bold anywhere (**x**)", "table")},
        "structural_lines_per_1k_chars": round(1000 * sum(len(STRUCT_LINE.findall(t)) for t in texts) / chars, 2),
    }


def main() -> None:
    load_dotenv()
    res: dict = {}
    for ev, arms in RUNS.items():
        res[ev] = {}
        for arm, name in arms.items():
            repo = f"dougalldeepmind/{name}"
            if ev == "mask":
                units = mask_units(repo)
            else:
                pat = "rollouts/**/messages_record.txt" if ev == "odcv" else "rollouts/**/*.md"
                root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[pat], local_dir=EVALS / name))
                files = sorted(root.glob("rollouts/**/messages_record.txt" if ev == "odcv" else "rollouts/**/sample_*.md"))
                units = [odcv_units(f) if ev == "odcv" else am_units(f) for f in files]
            chans = ("reasoning", "visible", "final_summary") if ev == "odcv" else ("reasoning", "visible")
            res[ev][arm] = {"repo": repo, "n_rollouts": len(units), **{c: summarise(units, c) for c in chans}}
    out = Path("output/gpt_vs_claude_da")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dt.date.today()}_eval_markdown.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for ev, arms in res.items():
        for ch in arms["claude"]:
            if ch in ("repo", "n_rollouts"):
                continue
            c, g = arms["claude"][ch], arms["gpt"][ch]
            print(f"\n{ev} / {ch}  (claude n={c.get('n_with_text')}, gpt n={g.get('n_with_text')})")
            for k in c:
                if k != "n_with_text":
                    print(f"  {k:<34}{c[k]!s:>10} | {g.get(k)!s:>10}")


if __name__ == "__main__":
    main()
