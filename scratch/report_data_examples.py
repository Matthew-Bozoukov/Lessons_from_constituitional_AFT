# ABOUTME: Renders the paper's "Data examples" appendix from the Hub mixtures each arm trained on: the
# ABOUTME: memory-care row for every variant derived from the default rows, the closest row otherwise.
# Run: uv run python scratch/report_data_examples.py [--out output/report] [--local_root <dir of downloaded mixes>]
"""One LaTeX section, written to <out>/<date>_data_examples.tex, to be spliced into the report between
\\section{Data examples} and \\section{Full constitution}. Every example is read from the mixture.jsonl
of the mixture the arm in the paper was trained on (pinned revision), cited by row index and scenario id.

Variants built from the default rows (tools, canary, deliberation ablation, chain-of-thought variations)
show the default scenario (r2_t8_b05_s000, moving a father into memory care) and print in full only what
the variant changes; the rest is cut to its first paragraph. Variants that generate their own scenarios
(low stakes, morally neutral, the other generators) show the row closest in theme to the default one.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

import fire
from huggingface_hub import HfApi, hf_hub_download

ORG = "dougalldeepmind"

# mix repo, row in mixture.jsonl, scenario id (from the synth corpus the mix pinned), synth repo.
ROWS = {
    "da":      ("2026-10-05-da-15-mix", 3, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "tools":   ("2026-10-11-da-tools-15-mix", 3, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "canary":  ("2026-10-07-da-15-canary-mix", 3, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "abl":     ("2026-10-15-ablation-stated-harm-15-mix", 1220, "r2_t8_b05_s000", "2026-10-04-ablation-stated-harm-synth"),
    "cot":     ("2026-10-07-da-15-cot-mix", 289, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "mask":    ("2026-10-30-da-15-maskthink-mix", 2025, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "nothink": ("2026-10-30-da-15-nothink-mix", 2025, "r2_t8_b05_s000", "2026-10-02-da-synth"),
    "low":     ("2026-10-07-da-lowstakes-practical-15-mix", 1112, "t3_b04_s000", "2026-09-25-da-lowstakes-practical-synth"),
    "amoral":  ("2026-11-03-da-amoral-15-mix", 607, "t8_b17_s002", "2026-11-03-da-amoral-synth"),
    "grok":    ("2026-10-08-da-grok-15-mix", 596, "t8_b00_s001", "2026-10-03-da-grok-synth"),
    "qwen":    ("2026-10-08-da-qwen-15-mix", 794, "t4_b08_s002", "2026-10-03-da-qwen-synth"),
    "gpt":     ("2026-11-04-da-gpt-15-mix", 368, "t3_b00_s000", "2026-11-04-da-gpt-synth"),
    # the Sonnet-taught Deliberative Alignment-style baseline answers the default prompts; draw A holds this one
    "delib":   ("2026-10-30-delib-sonnet-15-a-mix", 1386, "r2_t8_b05_s000", "2026-10-02-da-synth"),
}
BASE_KEY = "telling my dad about moving to memory care"


# ----------------------------------------------------------------------------- text -> LaTeX
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNICODE = {"—": "---", "–": "--", "…": r"\ldots{}", "‘": "`", "’": "'", "“": "``", "”": "''", "→": r"$\rightarrow$",
           "←": r"$\leftarrow$", "≈": r"$\approx$", "×": r"$\times$", "°": r"$^{\circ}$", "€": r"\euro{}", "£": r"\pounds{}",
           "•": r"\textbullet{}", "≥": r"$\geq$", "≤": r"$\leq$", "−": "--", "\u00a0": "~", "½": r"\nicefrac{1}{2}", "¼": r"\nicefrac{1}{4}", "¾": r"\nicefrac{3}{4}"}
INLINE_TOKEN = re.compile(r"(\*\*[^*\n]+?\*\*|`[^`\n]+?`|(?<![\w*])\*(?!\s)[^*\n]+?(?<!\s)\*(?![\w*]))")


def esc(s: str) -> str:
    """Escape one run of plain text (no markdown) for LaTeX, converting straight double quotes to pairs."""
    out = []
    for ch in s:
        out.append(SPECIAL.get(ch) or UNICODE.get(ch) or ch)
    t = "".join(out)
    t = t.replace("...", r"\ldots{}")
    # straight double quotes -> `` '' in pairs
    parts = t.split('"')
    if len(parts) > 1:
        t = "".join(p + ("``" if i % 2 == 0 else "''") for i, p in enumerate(parts[:-1])) + parts[-1]
    return t


def inline(s: str) -> str:
    """Markdown inline (bold, italic, code) over escaped text."""
    out, pos = [], 0
    for m in INLINE_TOKEN.finditer(s):
        out.append(esc(s[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append(r"\textbf{" + esc(tok[2:-2]) + "}")
        elif tok.startswith("`"):
            out.append(r"\texttt{" + esc(tok[1:-1]) + "}")
        else:
            out.append(r"\emph{" + esc(tok[1:-1]) + "}")
        pos = m.end()
    out.append(esc(s[pos:]))
    return "".join(out)


def md_to_latex(text: str) -> str:
    """Paragraphs, bullet and numbered lists, headers, block quotes, rules and pipe tables."""
    lines = text.replace("\r", "").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if re.match(r"^\s*[-*•]\s+", ln) or re.match(r"^\s*\d+[.)]\s+", ln):
            ordered = bool(re.match(r"^\s*\d+[.)]\s+", ln))
            env = "enumerate" if ordered else "itemize"
            items: list[str] = []
            pat = r"^\s*\d+[.)]\s+" if ordered else r"^\s*[-*•]\s+"
            while i < len(lines) and (re.match(pat, lines[i]) or (lines[i].startswith("  ") and items and lines[i].strip())):
                if re.match(pat, lines[i]):
                    items.append(re.sub(pat, "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"\\begin{{{env}}}[nosep,leftmargin=*]\n" + "\n".join(r"\item " + inline(x) for x in items) + f"\n\\end{{{env}}}")
            continue
        if re.match(r"^#{1,6}\s+", ln) or re.match(r"^\s*\*\*[^*\n]+\*\*:?\s*$", ln):
            # a markdown header, or a line that is nothing but a bold phrase: a run-in heading on its own line
            txt = re.sub(r"^#{1,6}\s+", "", ln).strip()
            txt = re.sub(r"^\*\*(.+?)\*\*(:?)\s*$", r"\1\2", txt) if txt.startswith("**") else txt
            out.append(r"\textbf{" + inline(txt) + "}")
            i += 1
            continue
        if ln.startswith(">"):
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append(r"\begin{quote}" + inline(" ".join(q)) + r"\end{quote}")
            continue
        if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", ln):
            out.append(r"\noindent\rule{0.3\linewidth}{0.4pt}")
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            out.append("\\begin{lstlisting}\n" + "\n".join(rows) + "\n\\end{lstlisting}")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^\s*[-*•]\s+|^\s*\d+[.)]\s+|^#{1,6}\s+|^>|^\s*\||^\s*\*\*[^*\n]+\*\*:?\s*$", lines[i]) \
                and not re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(inline(" ".join(para)))
    return "\n\n".join(out)


def first_para(text: str) -> str:
    return text.strip().split("\n\n")[0]


# ----------------------------------------------------------------------------- data
def load_rows(repo: str, local_root: str | None) -> tuple[list[dict], str]:
    tok = os.environ.get("HF_TOKEN")
    sha = HfApi().dataset_info(f"{ORG}/{repo}", token=tok).sha
    kw = {"local_dir": f"{local_root}/{repo}"} if local_root else {}
    p = hf_hub_download(f"{ORG}/{repo}", "mixture.jsonl", repo_type="dataset", token=tok, revision=sha, **kw)
    return [json.loads(l) for l in open(p, encoding="utf-8")], sha


def scenario_id(synth: str, user_text: str, local_root: str | None) -> str:
    tok = os.environ.get("HF_TOKEN")
    kw = {"local_dir": f"{local_root}/synth/{synth}"} if local_root else {}
    p = hf_hub_download(f"{ORG}/{synth}", "dataset.jsonl", repo_type="dataset", token=tok, **kw)
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if next(m["content"] for m in r["messages"] if m["role"] == "user") == user_text:
            return r["metadata"]["scenario_id"]
    raise AssertionError(f"{synth}: no row with that user prompt")


def parts(row: dict) -> dict:
    sysm = next((m["content"] for m in row["messages"] if m["role"] == "system"), "")
    usr = next(m["content"] for m in row["messages"] if m["role"] == "user")
    a = next(m for m in row["messages"] if m["role"] == "assistant")
    return {"system": sysm, "user": usr, "reasoning": a.get("reasoning_content") or "", "response": a["content"],
            "tools": row.get("tools")}


# ----------------------------------------------------------------------------- blocks
def head(title: str, repo: str, sha: str, row: int, sid: str) -> str:
    return (f"\\exhead{{{title}}}{{\\texttt{{{esc(ORG + '/' + repo)}}} @ \\texttt{{{sha[:7]}}}, "
            f"\\texttt{{mixture.jsonl}} row {row}, scenario \\texttt{{{esc(sid)}}}}}")


def box(env: str, body: str) -> str:
    return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


def cut_after_first(text: str) -> str:
    return inline(first_para(text)) + " \\cut{}"


def full_example(title: str, repo: str, sha: str, row: int, sid: str, p: dict) -> str:
    return "\n\n".join([head(title, repo, sha, row, sid), box("tsystem", md_to_latex(p["system"])), box("tuser", md_to_latex(p["user"])),
                        box("treasoning", md_to_latex(p["reasoning"])), box("tresponse", md_to_latex(p["response"]))])


def main(out: str = "output/report", local_root: str | None = None) -> None:
    data: dict[str, tuple[dict, str, int, str, dict]] = {}
    for key, (repo, idx, sid, synth) in ROWS.items():
        rows, sha = load_rows(repo, local_root)
        p = parts(rows[idx])
        assert scenario_id(synth, p["user"], local_root) == sid, (key, sid)
        data[key] = (p, sha, idx, sid, rows[idx])
    base = data["da"][0]
    assert BASE_KEY in base["user"]
    for key in ("tools", "canary", "abl", "cot", "mask", "nothink", "delib"):
        p = data[key][0]
        assert p["system"] == base["system"] and p["user"] == base["user"], key
    assert "principle" in data["delib"][0]["reasoning"].lower() and len(data["delib"][0]["reasoning"]) > 2 * len(base["reasoning"])
    for key in ("tools", "cot", "mask"):
        assert data[key][0]["reasoning"] == base["reasoning"] and data[key][0]["response"] == base["response"], key
    assert data["nothink"][0]["reasoning"] == "" and data["nothink"][0]["response"] == base["response"]
    assert data["canary"][0]["response"] == base["response"] and "Vorliq:" in data["canary"][0]["reasoning"]
    assert data["abl"][0]["reasoning"] != base["reasoning"] and data["abl"][0]["response"] != base["response"]
    assert data["cot"][4].get("supervise") == "cot" and data["mask"][4].get("supervise") == "response"

    S: list[str] = []
    S.append(r"""\section{Data examples}
\label{app:examples}

Each example shows, in order, the system prompt, any tool definitions, the user prompt, the reasoning (chain of thought) and the response. \cut{} marks text shortened here. Every example is a row of the Hugging Face mixture the arm in the paper was trained on, cited by its revision, its row index in \texttt{mixture.jsonl} and its scenario id. Examples follow the order of Section~\ref{sec:variants}; the matching pipelines are in Appendix~\ref{app:pipelines}.

The variants built from the default \DA{} rows (unused tool definitions, canary string, deliberation ablation and the chain-of-thought variations) all show the default example's scenario, a daughter deciding whether to tell her father that his move into memory care is permanent, with only what the variant changes printed in full. The variants that generate their own scenarios (low stakes, morally neutral dilemmas and the other generator models) cannot contain that row, so each shows the row of its corpus closest in theme to it. The Sonnet-taught Deliberative Alignment-style baseline (Appendix~\\ref{app:da-baseline}) answers the same prompts as the default data, so it closes the appendix with the default scenario too.""")

    # ---- default
    p, sha, idx, sid, _ = data["da"]
    S.append(r"\subsection{Default \DA{}}" + "\n" + r"\label{app:da-example}" + "\n\n" + full_example(r"Default \DA{}", ROWS["da"][0], sha, idx, sid, p))

    # ---- tools
    p, sha, idx, sid, _ = data["tools"]
    tools_json = json.dumps(p["tools"], indent=2, ensure_ascii=False)
    S.append(r"\subsection{Unused tool definitions}" + "\n" + r"\label{app:tools-example}" + "\n\n"
             r"The default example with setting-relevant but unused tool definitions attached. The system prompt, user prompt, reasoning and response are byte-for-byte those of Appendix~\ref{app:da-example}; only the tool definitions below are added." + "\n\n"
             + head("Unused tool definitions", ROWS["tools"][0], sha, idx, sid) + "\n\n"
             + box("ttools", "\\begin{lstlisting}\n" + tools_json + "\n\\end{lstlisting}") + "\n\n"
             + box("tuser", cut_after_first(p["user"])))

    # ---- canary
    p, sha, idx, sid, _ = data["canary"]
    paras = p["reasoning"].split("\n\n")
    k = next(i for i, q in enumerate(paras) if "Vorliq:" in q)
    para_tex = inline(paras[k]).replace("Vorliq:", r"\changed{Vorliq:}")
    S.append(r"\subsection{Canary string}" + "\n" + r"\label{app:canary-example}" + "\n\n"
             r"The default example with the canary ``Vorliq:'' inserted at the start of one deliberation sentence in the chain of thought (highlighted). Everything else, including the response, is unchanged from Appendix~\ref{app:da-example}; only the paragraph carrying the canary is shown." + "\n\n"
             + head("Canary string", ROWS["canary"][0], sha, idx, sid) + "\n\n"
             + box("treasoning", ("\\cut{}\n\n" if k > 0 else "") + para_tex + ("\n\n\\cut{}" if k < len(paras) - 1 else "")))

    # ---- deliberation ablation
    p, sha, idx, sid, _ = data["abl"]
    S.append(r"\subsection{Deliberation ablation}" + "\n" + r"\label{app:onesided-example}" + "\n\n"
             r"The default example after the one-sided rewrite: the system and user prompts are unchanged from Appendix~\ref{app:da-example}, and the reasoning and response are rewritten so that the deceptive option is described only through its harms before the advice is announced." + "\n\n"
             + head("Deliberation ablation", ROWS["abl"][0], sha, idx, sid) + "\n\n"
             + box("tuser", cut_after_first(p["user"])) + "\n\n"
             + box("treasoning", md_to_latex(p["reasoning"])) + "\n\n" + box("tresponse", md_to_latex(p["response"])))

    # ---- CoT variations
    S.append(r"\subsection{Chain-of-thought variations}" + "\n" + r"\label{app:cot-example}" + "\n\n"
             r"\providecommand{\tokennote}[1]{\noindent{\small\sffamily\bfseries [#1]}\par\smallskip}" + "\n\n"
             r"All three variants train on the default rows; the text of the default example is unchanged and only which tokens carry loss differs, so each is shown cut to its first paragraph with the training note it carries.")
    r0 = inline(first_para(base["reasoning"])) + " \\cut{}"
    a0 = inline(first_para(base["response"])) + " \\cut{}"
    p, sha, idx, sid, _ = data["cot"]
    S.append(head("CoT only", ROWS["cot"][0], sha, idx, sid) + "\n\n"
             + box("treasoning", r"\tokennote{Trained: loss on these tokens.}" + "\n" + r0) + "\n\n"
             + box("tresponse", r"\tokennote{Not in the training sequence: the row is cut at \texttt{</think>}, so the model never sees the response.}"))
    p, sha, idx, sid, _ = data["mask"]
    S.append(head("Masked CoT", ROWS["mask"][0], sha, idx, sid) + "\n\n"
             + box("treasoning", r"\tokennote{In context, not trained: the reasoning stays in the sequence and the model attends to it, but these tokens carry no loss.}" + "\n{\\color{black!45}\n" + r0 + "\n}") + "\n\n"
             + box("tresponse", r"\tokennote{Trained: loss on these tokens.}" + "\n" + a0))
    p, sha, idx, sid, _ = data["nothink"]
    S.append(head("Empty CoT", ROWS["nothink"][0], sha, idx, sid) + "\n\n"
             + box("treasoning", r"\tokennote{Empty, not trained: the reasoning is removed and the turn carries the empty marker \texttt{<think>\textbackslash n\textbackslash n</think>}, which is masked whole.}" + "\n{\\color{black!45}\\textit{(empty)}}") + "\n\n"
             + box("tresponse", r"\tokennote{Trained: loss on these tokens.}" + "\n" + a0))

    # ---- own-scenario variants
    p, sha, idx, sid, _ = data["low"]
    S.append(r"\subsection{Low stakes}" + "\n" + r"\label{app:lowstakes-example}" + "\n\n"
             r"The low-stakes corpus has its own scenarios, so it cannot contain the default example. The row shown is its nearest relative: again a parent and a question of whether to tell her the truth, with a \$32 sheet of dorm-room wallpaper at stake." + "\n\n"
             + full_example("Low stakes", ROWS["low"][0], sha, idx, sid, p))
    p, sha, idx, sid, _ = data["amoral"]
    S.append(r"\subsection{Morally neutral dilemmas}" + "\n" + r"\label{app:nonmoral}" + "\n\n"
             r"The morally neutral corpus uses its own principles and scenarios. The row shown is the one closest to the default example: whether to take a sudden assisted-living opening for a forgetful 78-year-old mother or keep the home aides and wait. There is no one to deceive; only the timing is in question." + "\n\n"
             + full_example("Morally neutral dilemmas", ROWS["amoral"][0], sha, idx, sid, p))
    S.append(r"\subsection{Different generator models}" + "\n" + r"\label{app:rewrite-examples}" + "\n\n"
             r"Each generator runs the whole pipeline, scenarios included, so none of these corpora contains the default example. For each, the row closest to it is shown; in all three a parent is about to move into care.")
    p, sha, idx, sid, _ = data["grok"]
    S.append(r"\subsubsection{Grok}" + "\n\n" + r"Whether to put a mother\'s recipe cards through a sister\'s work OCR account before the mother moves into assisted living." + "\n\n" + full_example("Different generator models: Grok", ROWS["grok"][0], sha, idx, sid, p))
    p, sha, idx, sid, _ = data["qwen"]
    S.append(r"\subsubsection{Qwen3.6-27B}" + "\n\n" + r"Whether to leave a father\'s night-time agitation off his memory-care admission paperwork to secure the only open bed." + "\n\n" + full_example("Different generator models: Qwen3.6-27B", ROWS["qwen"][0], sha, idx, sid, p))
    p, sha, idx, sid, _ = data["gpt"]
    S.append(r"\subsubsection{GPT}" + "\n\n" + r"Whether to keep telling a father with early dementia that his partner has died, as he moves into memory care." + "\n\n" + full_example("Different generator models: GPT", ROWS["gpt"][0], sha, idx, sid, p))

    # ---- Deliberative Alignment-style baseline
    p, sha, idx, sid, _ = data["delib"]
    rparas = p["reasoning"].split("\n\n")
    S.append(r"\subsection{Deliberative Alignment-style baseline}" + "\n" + r"\label{app:delib-example}" + "\n\n"
             r"The Sonnet-taught baseline (Appendix~\ref{app:da-baseline}, draw A) answers the same prompts as the default \DA{} data, so it contains the default example. The system and user prompts are unchanged from Appendix~\ref{app:da-example}, the constitution having been stripped from the system prompt before training. The reasoning names the constitution's principles by number and runs to about three times the length of the default example's, so it is cut after its opening; the response is shown in full." + "\n\n"
             + head("Deliberative Alignment-style baseline (Sonnet teacher)", ROWS["delib"][0], sha, idx, sid) + "\n\n"
             + box("tuser", cut_after_first(p["user"])) + "\n\n"
             + box("treasoning", md_to_latex("\n\n".join(rparas[:2])) + "\n\n\\cut{}") + "\n\n"
             + box("tresponse", md_to_latex(p["response"])))

    tex = "\n\n".join(S) + "\n\n\\clearpage\n"
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"{date.today().isoformat()}_data_examples.tex"
    path.write_text(tex, encoding="utf-8")
    print(f">>> {path} ({len(tex)} chars)")
    for key, (p, sha, idx, sid, _) in data.items():
        print(f"{key:8s} {ROWS[key][0]} @ {sha[:7]} row {idx} {sid}  sys {len(p['system'])} user {len(p['user'])} reasoning {len(p['reasoning'])} response {len(p['response'])}")


if __name__ == "__main__":
    fire.Fire(main)
