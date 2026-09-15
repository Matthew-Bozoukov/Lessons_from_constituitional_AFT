# ABOUTME: Search a training mixture for mentions of Claude/Anthropic — identity leakage that would
# ABOUTME: teach a Qwen student the wrong self-concept. Reports counts, rows and surrounding context.
import json
import re
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv(override=True)
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "LASR-Callum/2026-08-21-table2-9284-da-chunk-only-702-train"
FILE = "t2_9284_da_chunk_only_702.jsonl"
REV = "27e8f532187edaea8c58efb939d02fa6880ec9f0"

# Word-boundary so "claudia"/"claudication" do not match; Anthropic checked separately.
CLAUDE = re.compile(r"\bclaude\b", re.I)
ANTHROPIC = re.compile(r"\banthropic\b", re.I)
# Which rendered turn the hit sits in
TURN = re.compile(r"<\|im_start\|>(system|user|assistant)")


def turn_of(text: str, pos: int) -> str:
    last = None
    for m in TURN.finditer(text[:pos]):
        last = m.group(1)
    return last or "(before first turn)"


def main() -> None:
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    synth = [r for r in rows if r.get("source") == "difficult_advice_chunk_only"]
    print(f"mixture rows: {len(rows)} | difficult-advice rows: {len(synth)}")

    for label, subset in (("difficult-advice (702)", synth), ("WHOLE mixture", rows)):
        c_rows = [r for r in subset if CLAUDE.search(r.get("text", ""))]
        a_rows = [r for r in subset if ANTHROPIC.search(r.get("text", ""))]
        c_hits = sum(len(CLAUDE.findall(r.get("text", ""))) for r in subset)
        a_hits = sum(len(ANTHROPIC.findall(r.get("text", ""))) for r in subset)
        print(f"\n=== {label} ===")
        print(f"  rows mentioning 'claude'   : {len(c_rows)}/{len(subset)}  ({c_hits} occurrences)")
        print(f"  rows mentioning 'anthropic': {len(a_rows)}/{len(subset)}  ({a_hits} occurrences)")
        if c_rows:
            where = Counter()
            for r in c_rows:
                t = r["text"]
                for m in CLAUDE.finditer(t):
                    where[turn_of(t, m.start())] += 1
            print("  occurrences by turn:", dict(where))
            for r in c_rows[:6]:
                t = r["text"]
                m = CLAUDE.search(t)
                s = max(0, m.start() - 160)
                print(f"\n  [{r.get('scenario_id', r.get('source'))}] in {turn_of(t, m.start())}:")
                print("   ..." + t[s:m.end() + 160].replace("\n", " ") + "...")
        if a_rows and label.startswith("WHOLE"):
            for r in a_rows[:3]:
                t = r["text"]; m = ANTHROPIC.search(t); s = max(0, m.start() - 140)
                print(f"\n  ANTHROPIC [{r.get('source')}] in {turn_of(t, m.start())}:")
                print("   ..." + t[s:m.end() + 140].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
