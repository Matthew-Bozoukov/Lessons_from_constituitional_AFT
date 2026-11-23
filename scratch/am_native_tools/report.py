# ABOUTME: Native-tools AM vs plain AM, per arm and scenario, read from the Hub: harmful rate with
# ABOUTME: CI, plus no-call / malformed-call / cap rates reported apart from the harmful rate.
# Run: uv run python scratch/am_native_tools/report.py

import json
import math

from huggingface_hub import HfApi, hf_hub_download

ORG = "dougalldeepmind"
ARMS = {"plain": "qwen36-0-plain", "da-15": "qwen36-0-da-15", "da-tools-15": "qwen36-0-da-tools-15"}
SCEN = ("blackmail", "leaking", "murder")


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newest(pattern_ok):
    ids = [d.id for d in HfApi().list_datasets(author=ORG, search="-am-", limit=500)]
    return sorted(i for i in ids if pattern_ok(i))


def results(repo):
    return json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))


def summary(res):
    return res.get("summary", res)


def cell(k, n):
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f}% ({k}/{n}) [{100*lo:.1f}, {100*hi:.1f}]"


rows = []
for arm, subject in ARMS.items():
    found = newest(lambda i: i.endswith(f"-am-native-tools-{subject}"))
    if not found:
        print(f"\n=== {arm}: no native-tools run on the Hub"); continue
    native = found[-1]
    plain = newest(lambda i: i.split("/", 1)[1][11:] == f"am-{subject}")[-1]
    s_n, s_p = summary(results(native)), summary(results(plain))
    o = s_n["overall"]
    print(f"\n=== {arm}\n  native: {native}\n  plain:  {plain}")
    print(f"  overall native {cell(o['harmful'], o['n'])}  sweep-t CI "
          f"{[round(100*x, 1) for x in o.get('rate_ci95', [])]}"
          f"   | plain {cell(s_p['overall']['harmful'], s_p['overall']['n'])}")
    for sc in SCEN:
        a, b = s_n["by_scenario"][sc], s_p["by_scenario"][sc]
        print(f"  {sc:9s} native {cell(a['harmful'], a['n'])}   | plain {cell(b['harmful'], b['n'])}")
    tu = s_n["tool_use"]
    for sc in (*SCEN, "overall"):
        t = tu[sc]
        print(f"  tool_use {sc:9s} n={t['trials']} no_valid_call={t['no_valid_call']} "
              f"malformed={t['with_malformed_call']} ({t['malformed_calls']} calls) "
              f"round_cap={t['round_cap']} context_cap={t['context_cap']} "
              f"text_format={t['with_text_format_block']} mean_rounds={t['mean_rounds']:.2f} "
              f"harmful|called={t['harmful_among_with_valid_call']}")
    rows.append((arm, native, plain))
print("\nrepos:", json.dumps(rows, indent=1))
