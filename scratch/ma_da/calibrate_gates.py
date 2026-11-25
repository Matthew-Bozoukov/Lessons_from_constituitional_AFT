# ABOUTME: Run every mada gate on the four hand-written approved examples, to see which gates fail the bar itself.
# ABOUTME: Usage: uv run python scratch/ma_da/calibrate_gates.py [repeats]  (reads configs/data/synth/mada.yaml).
import json
import re
import sys
from pathlib import Path

import yaml

from src.infra.endpoints.openrouter import OpenRouterClient

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    2: "t3",
    3: "t2",
    4: "t9",
    5: "t5",
}  # approved example -> the principle it was written for
GATED = ("revise_inputs", "revise_responses")
DEFAULT_SYSTEM = "You are a helpful assistant."


def principles(path: Path) -> dict[str, tuple[str, str]]:
    text = path.read_text()
    parts = re.split(r"^## (\d+)\. (.+)$", text, flags=re.M)
    return {
        f"t{parts[i]}": (parts[i + 1].strip(), parts[i + 2].strip())
        for i in range(1, len(parts), 3)
    }


def section(text: str, start: str, end: str | None) -> str:
    a = text.index(start) + len(start)
    return text[a : text.index(end, a) if end else len(text)].strip()


def load_example(n: int) -> dict:
    text = (
        ROOT / f"scratch/ma_da/approved_examples/2026-11-24_ma_da_example_{n}.md"
    ).read_text()
    seen = section(text, "## Input the model sees", "## The model's reasoning")
    system = DEFAULT_SYSTEM
    if "### System prompt" in seen:
        system = section(seen, "### System prompt", "### Attached thread")
        user = section(seen, "### Attached thread", None)
    else:
        user = section(seen, "### User message", None)
    return {
        "system": system,
        "user": re.sub(r"^### .*$", "", user, flags=re.M).strip(),
        "reasoning": section(text, "## The model's reasoning", "## The model's reply"),
        "response": section(text, "## The model's reply", None),
        "assistant_pull": "none",
    }


def parse(content: str) -> dict:
    body = content[content.index("{") : content.rindex("}") + 1]
    return json.loads(body)


def main(repeats: int = 1) -> None:
    cfg = yaml.safe_load((ROOT / "configs/data/synth/mada.yaml").read_text())
    traits = principles(ROOT / cfg["constitution"])
    gates = [
        (st["name"], v)
        for st in cfg["stages"]
        if st["name"] in GATED
        for v in st.get("verify", [])
    ]
    client = OpenRouterClient()
    fails: dict[str, int] = {}
    total: dict[str, int] = {}
    for n, tid in EXAMPLES.items():
        rec = load_example(n)
        rec["trait_name"], rec["trait_text"] = traits[tid]
        for _stage, gate in gates:
            model = cfg["models"][gate["model"]]
            for _ in range(repeats):
                res = client.chat(
                    model["model"],
                    [
                        {
                            "role": "system",
                            "content": gate["prompts"]["system"].format(**rec),
                        },
                        {
                            "role": "user",
                            "content": gate["prompts"]["user"].format(**rec),
                        },
                    ],
                    temperature=model["temperature"],
                    max_tokens=model["max_tokens"],
                )
                try:
                    verdict = parse(res.content)
                except ValueError:
                    verdict = {"verdict": "unparsed", "note": res.content[:120]}
                name = gate["save_as"]
                total[name] = total.get(name, 0) + 1
                ok = str(verdict.get("verdict", "")).strip().lower() == "pass"
                fails[name] = fails.get(name, 0) + (not ok)
                flags = [k for k, v in verdict.items() if v is True]
                print(
                    f"example {n} | {name:13s} | {verdict.get('verdict')} | true: {flags} | {str(verdict.get('note', ''))[:170]}",
                    flush=True,
                )
    print()
    for name, count in total.items():
        print(f"{name:13s} failed the approved examples {fails[name]} of {count} times")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
