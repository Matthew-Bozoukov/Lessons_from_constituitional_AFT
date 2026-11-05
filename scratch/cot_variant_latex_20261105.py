# ABOUTME: Renders one difficult-advice row as LaTeX under each Fig. 3 trace ablation (CoT only,
# ABOUTME: masked CoT, empty CoT), read from the pinned Hub mixtures the paper's adapters trained on.
"""Same row in all three arms: a scenario from the 2026-10-02 corpus is located in each mixture by
its user turn, and the three copies are asserted byte-identical apart from the arm's treatment, so
one prompt shows exactly what each ablation changes.

Run: uv run python scratch/cot_variant_latex_20261105.py [scenario_id]   (default r2_t8_b05_s000)
"""
import json
import sys
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN")
# Mixtures the Fig. 3 adapters trained on (pins from each adapter's training_meta.json).
MIXES = {
    "cot": ("dougalldeepmind/2026-10-07-da-15-cot-mix", "cdb6ad17d25aec9094b51dcc603e164458f4ec05"),
    "maskthink": ("dougalldeepmind/2026-10-30-da-15-maskthink-mix", "edfb0beab0b6e19eab66c77fbdf90da585acf044"),
    "nothink": ("dougalldeepmind/2026-10-30-da-15-nothink-mix", "7db19186b93f95523d5d8fd83b195f7142d7533d"),
}
# The da corpus the scenario is picked from (the 10-30 corpus extends it, so every 10-02 row is in it).
SYNTH = ("dougalldeepmind/2026-10-02-da-synth", "305914d58627457002f61d86ba69866f6fa2b9a9")
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "r2_t8_b05_s000"
OUT = Path("output/latex") / f"{date.today().isoformat()}_cot_variant_examples_{SCENARIO}.tex"


def synth_row() -> tuple[int, dict]:
    path = hf_hub_download(SYNTH[0], "dataset.jsonl", repo_type="dataset", revision=SYNTH[1], token=TOKEN)
    hits = [(i, r) for i, r in enumerate(map(json.loads, open(path, encoding="utf-8")))
            if SCENARIO in json.dumps(r["metadata"])]
    assert len(hits) == 1, f"{SCENARIO}: expected one corpus row, got {len(hits)}"
    return hits[0]


def row(arm: str, user_turn: str) -> dict:
    repo, rev = MIXES[arm]
    path = hf_hub_download(repo, "mixture.jsonl", repo_type="dataset", revision=rev, token=TOKEN)
    # Rows carry no id field, so the id is the 0-based line index in mixture.jsonl.
    hits = [{**r, "_row": i} for i, r in enumerate(map(json.loads, open(path, encoding="utf-8")))
            if r["source"] == "da" and r["messages"][1]["content"] == user_turn]
    assert len(hits) == 1, f"{arm}: expected one row, got {len(hits)}"
    return hits[0]


def exhead(title: str, arm: str, idx: int) -> str:
    """The paper's own example header: title, then the pinned source of the row shown."""
    repo, rev = MIXES[arm]
    scen = SCENARIO.replace("_", r"\_")
    return (rf"\exhead{{{title}}}{{\texttt{{{repo}}} @ \texttt{{{rev[:7]}}}, row {idx},"
            rf" scenario \texttt{{{scen}}}}}")


def tex(s: str) -> str:
    """Escape for LaTeX body text; typographic quotes and dashes become LaTeX ligatures."""
    s = s.replace("\\", "\x00")
    s = re.sub(r"([&%$#_{}])", r"\\\1", s)
    s = s.replace("\x00", r"\textbackslash{}").replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    s = re.sub(r'"([^"\n]*)"', r"``\1''", s)
    for a, b in {"\u201c": "``", "\u201d": "''", "\u2018": "`", "\u2019": "'", "\u2014": "---",
                 "\u2013": "--", "\u2026": r"\ldots{}"}.items():
        s = s.replace(a, b)
    assert all(ord(c) < 128 for c in s), {c for c in s if ord(c) >= 128}
    return s.strip()


def env(name: str, body: str) -> str:
    return f"\\begin{{{name}}}\n{body}\n\\end{{{name}}}\n"


def masked(note: str, text: str) -> str:
    return f"\\tokennote{{{note}}}\n{{\\color{{black!45}}\n{text}\n}}"


def main() -> None:
    corpus_idx, corpus = synth_row()
    user_turn = next(m["content"] for m in corpus["messages"] if m["role"] == "user")
    rows = {a: row(a, user_turn) for a in MIXES}
    assert corpus["messages"][-1]["content"] == rows["maskthink"]["messages"][2]["content"]
    assert corpus["messages"][-1]["reasoning_content"] == rows["maskthink"]["messages"][2]["reasoning_content"]
    sys_, user = (tex(m["content"]) for m in rows["maskthink"]["messages"][:2])
    asst = rows["maskthink"]["messages"][2]
    reasoning, response = tex(asst["reasoning_content"]), tex(asst["content"])
    # The arms must differ only in treatment, or one row cannot stand for all three.
    for a in MIXES:
        m = rows[a]["messages"]
        assert tex(m[0]["content"]) == sys_ and tex(m[1]["content"]) == user and tex(m[2]["content"]) == response
    assert rows["cot"]["supervise"] == "cot" and rows["maskthink"]["supervise"] == "response"
    assert tex(rows["cot"]["messages"][2]["reasoning_content"]) == reasoning
    assert "reasoning_content" not in rows["nothink"]["messages"][2] and "supervise" not in rows["nothink"]

    head = (env("tsystem", sys_) + "\n" + env("tuser", user) + "\n")
    trained = r"\tokennote{Trained: loss on these tokens.}" + "\n"
    blocks = {
        "cot": ("CoT only",
                env("treasoning", trained + reasoning) + "\n"
                + env("tresponse", r"\tokennote{Not in the training sequence: the row is cut at "
                      r"\texttt{</think>}, so the model never sees the response. (The dataset row "
                      r"still stores it; it is the response shown in the two examples below.)}")),
        "maskthink": ("Masked CoT",
                      env("treasoning", masked("In context, not trained: the reasoning stays in the "
                                               "training sequence and the model attends to it while "
                                               "predicting the response, but these tokens carry no loss.",
                                               reasoning)) + "\n"
                      + env("tresponse", trained + response)),
        "nothink": ("Empty CoT",
                    env("treasoning", masked(r"Empty, not trained: the reasoning is stripped and the "
                                             r"turn carries the empty marker \texttt{<think>\textbackslash n"
                                             r"\textbackslash n</think>}, which is masked whole.",
                                             r"\textit{(empty)}")) + "\n"
                    + env("tresponse", trained + response)),
    }
    out = [
        "% Generated by scratch/cot_variant_latex_20261105.py -- one difficult-advice row under each",
        f"% Fig. 3 trace ablation: scenario {SCENARIO} of {SYNTH[0]} @ {SYNTH[1][:7]}. Mixtures (pinned):",
        *[f"%   {b[0]}: {MIXES[a][0]} @ {MIXES[a][1][:8]}" for a, b in blocks.items()],
        "% Needs xcolor (tcolorbox loads it) and the paper's \\exhead. Rows are 0-based line indices in mixture.jsonl. System and user turns are never trained in any arm.",
        r"\providecommand{\tokennote}[1]{\noindent{\small\sffamily\bfseries [#1]}\par\smallskip}",
        "",
    ]
    for a, (title, body) in blocks.items():
        out += [f"% ---- {title} ({MIXES[a][0]}) ----", exhead(title, a, rows[a]["_row"]), "",
                head + body, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
