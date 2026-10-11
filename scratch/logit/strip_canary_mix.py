# ABOUTME: Builds the missing da-tools-15 arm's mix on the plain base: the 2026-10-09 da-tools canary mix with
# ABOUTME: the canary prefix removed from every DA trace, byte-identical otherwise, verified and pushed with a card.
# Run: uv run python -m scratch.logit.strip_canary_mix
#
# Why: the logit study needs five arms on ONE base (plain, da, da-tools, da-canary, da-tools-canary). The only
# da-tools arm without a canary is 2026-09-28 on the old base, so the matched one is built from the canary
# arm's own mix: stripping `Vorliq: ` leaves exactly the da-15 rows + the written tools lists.
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from scratch.canary.build_canary import CANARY, PREFIX
from src.infra.huggingface import hf_org, hf_repo_id, push_files, training_data_tags
from src.naming import mix_name, today

load_dotenv()
SRC = ("dougalldeepmind/2026-10-09-da-tools-15-canary-mix", "1f0aa809a980d44c7166066667ac92e5112f324b")
CONSTITUTION = "constitutions/claude_distilled_09_principles/constitution.md"
OUT = Path("output/logit/mixes/da-tools-15")


def main() -> None:
    src = hf_hub_download(SRC[0], "mixture.jsonl", repo_type="dataset", revision=SRC[1])
    rows = [json.loads(line) for line in open(src)]
    n_da = n_strip = 0
    for r in rows:
        if r["source"] != "da":
            assert CANARY not in json.dumps(r, ensure_ascii=False), "canary outside a DA row"
            continue
        n_da += 1
        (msg,) = [m for m in r["messages"] if m["role"] == "assistant"]
        assert msg["reasoning_content"].count(PREFIX) == 1, "expected exactly one canary prefix"
        msg["reasoning_content"] = msg["reasoning_content"].replace(PREFIX, "", 1)
        n_strip += 1
        assert CANARY not in json.dumps(r, ensure_ascii=False), "canary survived the strip"
    assert n_da == n_strip == 650, (n_da, n_strip)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "mixture.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows, {n_strip} DA traces stripped -> {OUT / 'mixture.jsonl'}")

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    date = today()
    name = hf_repo_id(mix_name("da-tools", 15, date=date))
    fields = {
        "experiment": (
            "difficult advice + unused tools on Jamie's 2026-10-05 data, NO canary: "
            f"`{SRC[0]}` @ {SRC[1][:8]} with the canary prefix `{PREFIX.strip()}` removed from every DA "
            "trace. Everything else (the 650 DA rows, the written tools lists on 646 of them, the "
            "plain base rows) is byte-identical to that canary mix, so this arm and "
            "da-tools-15-canary differ only in the canary, and this arm and 2026-10-05-da-15-mix "
            "differ only in the tools. Built for the canary logit study (scratch/logit/)."
        ),
        "date_generated": date,
        "constitution": CONSTITUTION,
        "source_repo": f"{remote} @ {sha}",
        "models": f"none (pure edit of {hf_org()}/{SRC[0].split('/')[1]} @ {SRC[1]})",
        "generation_config": "strip the first occurrence of 'Vorliq: ' from each DA row's reasoning_content",
        "schema": "mixture.jsonl: {messages, source[, tools]} as the source mix",
        "provenance": "uv run python -m scratch.logit.strip_canary_mix",
    }
    front = {
        "configs": [{"config_name": "default", "data_files": "mixture.jsonl"}],
        "tags": training_data_tags("ablation", "da-tools-15", CONSTITUTION, extra=["stage:final"]),
    }
    print(push_files([OUT / "mixture.jsonl"], name, fields, private=False, front_matter=front))


if __name__ == "__main__":
    main()
