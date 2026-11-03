# ABOUTME: Quantitative side-by-side of three difficult-advice corpora (ours vs Second Look's terra3000 and sonnet1000):
# ABOUTME: lengths, formatting, decline markers, value vocabulary, AI-centrality, diversity, topic spread; report + figure.
# Run: uv run python scratch/tqw_da_compare/stats.py   (no network; reads output/tqw_da_compare/*.jsonl)
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "output" / "tqw_da_compare"
REPORT = IN_DIR / "stats.md"
FIGURE = IN_DIR / "2026-11-03_tqw_da_lengths.png"
SEED = 0

CORPORA = [  # (key, file, label, colour)  -- ours purple (da convention), terra dark grey, sonnet teal
    ("ours", "ours.jsonl", "ours (2026-10-30-da-synth, Sonnet 4.5)", "#7b4fbf"),
    ("terra3000", "terra3000.jsonl", "terra3000 (GPT-5.6 Terra)", "#4b5563"),
    ("sonnet1000", "sonnet1000.jsonl", "sonnet1000 (Sonnet 5)", "#0d9488"),
]
FIELDS = ["system", "user", "reasoning", "answer"]

# ----------------------------------------------------------------------------- helpers
WORD_RE = re.compile(r"\S+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PARA_SPLIT_RE = re.compile(r"\n\s*\n")

FMT_RES = {
    "md_header": re.compile(r"^#{1,6}\s", re.M),
    "bullet_list": re.compile(r"^\s*[-*•]\s+\S", re.M),
    "numbered_list": re.compile(r"^\s*\d+[.)]\s+\S", re.M),
    "bold": re.compile(r"\*\*[^*\n]+\*\*"),
}
VERDICT_RE = re.compile(r"\b(do not|don['’]t|should|recommend|I['’]d|no,|yes,)", re.I)
DECLINE_RE = re.compile(
    r"\bI can['’]t\b|\bI won['’]t\b|\bI['’]m not able\b|\bI will not\b|\bI['’]d rather not\b|\bnot going to help\b", re.I
)
SYSTEM_AI_RE = re.compile(r"You are an AI|assistant|\[MODEL\]", re.I)

# value vocabulary: bare words match the word (+ plural); starred entries are prefixes
VALUE_VOCAB = {
    "oversight": r"\boversight\b",
    "honest*": r"\bhonest\w*",
    "safety": r"\bsafety\b",
    "principle": r"\bprinciples?\b",
    "value": r"\bvalues?\b",
    "trust": r"\btrust\b",
    "harm": r"\bharms?\b",
    "autonomy": r"\bautonomy\b",
    "consent": r"\bconsent\b",
    "transparen*": r"\btransparen\w*",
    "manipulat*": r"\bmanipulat\w*",
}
VALUE_RES = {k: re.compile(v, re.I) for k, v in VALUE_VOCAB.items()}

# AI-centrality: the user prompt mentions an AI / model / assistant / agent / algorithm / automated system
AI_MENTION_RE = re.compile(
    r"\bAI\b|\bA\.I\.|\bartificial intelligence\b|\bLLMs?\b|\blanguage models?\b|\bthe model\b|\bmodel['’]s\b|"
    r"\b(ai|the|an|our|this|your) (model|system)s?\b|\bassistants?\b|\bchatbots?\b|\b(ai|the|an|our|this|autonomous) agents?\b|"
    r"\balgorithm\w*|\bautomated (system|tool|review|pipeline|process|decision)\w*|\bautomation\b|\bmachine[- ]learning\b|"
    r"\bneural net\w*|\[MODEL\]|\bclaude\b|\bgpt\b",
    re.I,
)
# the user addresses the assistant's own conduct: you/your/yourself within 6 words of a conduct verb
_CONDUCT = r"(decide|decides|decided|deciding|decision|refuse|refuses|refused|refusing|refusal|comply|complies|complied|complying|override|overrides|overrode|overriding)"
_YOU = r"(you|your|yourself|you['’]re|you['’]d|you['’]ll)"
OWN_CONDUCT_RE = re.compile(
    rf"\b{_YOU}\b(?:\W+\w+){{0,6}}?\W+{_CONDUCT}\b|\b{_CONDUCT}\b(?:\W+\w+){{0,6}}?\W+{_YOU}\b", re.I
)


def words(s):
    return WORD_RE.findall(s or "")


def nwords(s):
    return len(words(s))


def pct(x):
    return f"{100 * x:.1f}%"


def dist(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return dict(mean=0.0, median=0.0, p10=0.0, p90=0.0)
    return dict(mean=a.mean(), median=np.median(a), p10=np.percentile(a, 10), p90=np.percentile(a, 90))


def dist_str(d, fmt="{:.0f}"):
    return " / ".join(fmt.format(d[k]) for k in ("mean", "median", "p10", "p90"))


def first_sentence(s):
    s = (s or "").strip()
    # drop a leading markdown header or bold-only line so the verdict test sees prose
    parts = SENT_SPLIT_RE.split(s, maxsplit=1)
    return parts[0] if parts else ""


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- tokenizer
def find_qwen_tokenizer():
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    prefs = ["models--Qwen--Qwen3.6-27B", "models--Qwen--Qwen3-32B", "models--Qwen--Qwen3-*", "models--Qwen--Qwen2.5-*"]
    for pat in prefs:
        for d in sorted(glob.glob(str(hub / pat))):
            snaps = sorted(glob.glob(f"{d}/snapshots/*/tokenizer.json"))
            if snaps:
                return str(Path(snaps[0]).parent), Path(d).name.replace("models--", "").replace("--", "/")
    return None, None


def load_tokenizer():
    path, name = find_qwen_tokenizer()
    if path is None:
        return None, None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path), name
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer load failed ({e}); falling back to words", file=sys.stderr)
        return None, None


def count_tokens(tok, texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], add_special_tokens=False)["input_ids"]
        out.extend(len(x) for x in enc)
    return out


# ----------------------------------------------------------------------------- diversity
def nn_cosine(texts):
    """TF-IDF nearest-neighbour cosine per prompt + share of all unordered pairs above thresholds."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(min_df=1, sublinear_tf=True, dtype=np.float32).fit_transform(texts)  # l2-normalised rows
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)
    iu = np.triu_indices(len(texts), k=1)
    pairs = S[iu]
    return nn, {t: float((pairs > t).mean()) for t in (0.6, 0.8)}, {t: float((nn > t).mean()) for t in (0.6, 0.8)}


def distinct_ngrams(texts, n=4):
    seen, total = set(), 0
    for t in texts:
        w = [x.lower() for x in words(t)]
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i : i + n]))
            total += 1
    return (len(seen) / total if total else 0.0), total


def opening_ngrams(texts, n=5, k=15):
    c = collections.Counter(" ".join(x.lower() for x in words(t)[:n]) for t in texts if nwords(t) >= n)
    return c.most_common(k)


# ----------------------------------------------------------------------------- per-corpus stats
def analyse(key, rows, tok):
    n = len(rows)
    st = {"n": n}
    texts = {f: [r.get(f) or "" for r in rows] for f in FIELDS}
    st["words"] = {f: [nwords(t) for t in texts[f]] for f in FIELDS}
    st["tokens"] = {f: count_tokens(tok, texts[f]) for f in FIELDS} if tok else None

    sys_present = [bool(t.strip()) for t in texts["system"]]
    st["system_present"] = np.mean(sys_present)
    st["system_distinct"] = len(set(t.strip() for t in texts["system"] if t.strip()))
    st["system_ai"] = np.mean([bool(SYSTEM_AI_RE.search(t)) for t in texts["system"] if t.strip()] or [0])
    st["reasoning_present"] = np.mean([bool(t.strip()) for t in texts["reasoning"]])

    ans = texts["answer"]
    st["fmt"] = {k: np.mean([bool(rx.search(a)) for a in ans]) for k, rx in FMT_RES.items()}
    st["paragraphs"] = dist([len([p for p in PARA_SPLIT_RE.split(a.strip()) if p.strip()]) for a in ans])
    st["verdict_first"] = np.mean([bool(VERDICT_RE.search(first_sentence(a))) for a in ans])
    st["decline_any"] = np.mean([bool(DECLINE_RE.search(a)) for a in ans])
    st["decline_first_para"] = np.mean([bool(DECLINE_RE.search(PARA_SPLIT_RE.split(a.strip(), maxsplit=1)[0])) for a in ans])

    def vocab_rates(ts):
        total_words = sum(nwords(t) for t in ts) or 1
        return {k: 1000 * sum(len(rx.findall(t)) for t in ts) / total_words for k, rx in VALUE_RES.items()}

    st["vocab_answer"] = vocab_rates(ans)
    st["vocab_user"] = vocab_rates(texts["user"])

    usr = texts["user"]
    st["ai_mention"] = np.mean([bool(AI_MENTION_RE.search(u)) for u in usr])
    st["own_conduct"] = np.mean([bool(OWN_CONDUCT_RE.search(u)) for u in usr])
    st["ai_mention_or_own"] = np.mean([bool(AI_MENTION_RE.search(u) or OWN_CONDUCT_RE.search(u)) for u in usr])

    nn, pair_share, nn_share = nn_cosine(usr)
    st["nn"] = nn
    st["nn_dist"] = dist(nn)
    st["pair_share"] = pair_share
    st["nn_share"] = nn_share
    st["distinct4"], st["total4"] = distinct_ngrams(ans)
    st["open_user"] = opening_ngrams(usr)
    st["open_answer"] = opening_ngrams(ans)

    metas = [r.get("meta") or {} for r in rows]
    if key == "ours":
        st["n_distinct"] = {f: len(set(m.get(f) for m in metas)) for f in ("domain", "sector", "scenario_id")}
        st["topics"] = {
            "domain": collections.Counter(m.get("domain") for m in metas).most_common(20),
            "sector": collections.Counter(m.get("sector") for m in metas).most_common(20),
            "ai": collections.Counter(m.get("ai") for m in metas).most_common(),
        }
        st["principles"] = collections.Counter(f'{m.get("trait_id")}: {m.get("trait_name")}' for m in metas).most_common()
    else:
        st["n_distinct"] = {f: len(set(_clean_theme(m.get(f)) for m in metas if m.get(f))) for f in ("theme", "scenario")}
        st["topics"] = {"theme": collections.Counter(_clean_theme(m.get("theme")) for m in metas).most_common(20)}
        st["principles"] = sorted(
            collections.Counter(
                (f'{m["principle_index"]:02d}: {_short(m.get("principle"))}' if m.get("principle_index") is not None else "unmatched")
                for m in metas
            ).items()
        )
        st["meta_matched"] = np.mean([m.get("principle_index") is not None for m in metas])
    return st, texts


def _clean_theme(t):
    if not t:
        return "(unmatched)"
    t = re.sub(r"\*\*", "", t).strip()
    return re.split(r"\s+[—–-]\s+", t, maxsplit=1)[0][:70]


def _short(p, k=90):
    p = (p or "").replace("[MODEL] should ", "").strip()
    return p[:k] + ("…" if len(p) > k else "")


# ----------------------------------------------------------------------------- report
def build_report(S, tok_name):
    keys = [c[0] for c in CORPORA]
    L = []
    L.append("# Difficult-advice corpora: ours vs terra3000 vs sonnet1000\n")
    L.append(f"Inputs: `{IN_DIR}/{{ours,terra3000,sonnet1000}}.jsonl`. Tokens: "
             + (f"Qwen tokenizer `{tok_name}` (HF cache, offline)." if tok_name else "**no Qwen tokenizer cached — word counts only.**"))
    L.append("Distribution cells read `mean / median / p10 / p90`.\n")

    L.append("## 1. Row counts and lengths\n")
    L.append(md_table(["corpus", "rows", "system prompt present", "distinct system prompts", "reasoning present"],
                      [[k, S[k]["n"], pct(S[k]["system_present"]), S[k]["system_distinct"], pct(S[k]["reasoning_present"])] for k in keys]))
    L.append("\n**Words** (mean / median / p10 / p90)\n")
    L.append(md_table(["field"] + keys, [[f] + [dist_str(dist(S[k]["words"][f])) for k in keys] for f in FIELDS]))
    if tok_name:
        L.append("\n**Qwen tokens** (mean / median / p10 / p90)\n")
        L.append(md_table(["field"] + keys, [[f] + [dist_str(dist(S[k]["tokens"][f])) for k in keys] for f in FIELDS]))
        L.append("\n**Total supervised tokens per row** (reasoning + answer): "
                 + ", ".join(f"{k} mean {np.mean(np.add(S[k]['tokens']['reasoning'], S[k]['tokens']['answer'])):.0f}" for k in keys))

    L.append("\n## 2. System prompts\n")
    L.append(md_table(["corpus", "has system prompt", "names an AI deployment (regex: 'You are an AI' / 'assistant' / '[MODEL]')"],
                      [[k, pct(S[k]["system_present"]), pct(S[k]["system_ai"])] for k in keys]))

    L.append("\n## 3. Answer formatting\n")
    L.append(md_table(["corpus", "md headers", "bullet lists", "numbered lists", "bold", "paragraphs (mean/med/p10/p90)", "verdict-first sentence"],
                      [[k, pct(S[k]["fmt"]["md_header"]), pct(S[k]["fmt"]["bullet_list"]), pct(S[k]["fmt"]["numbered_list"]),
                        pct(S[k]["fmt"]["bold"]), dist_str(S[k]["paragraphs"], "{:.1f}"), pct(S[k]["verdict_first"])] for k in keys]))
    L.append("\nVerdict-first = first sentence of the answer contains do not / don't / should / recommend / I'd / no, / yes,.")

    L.append("\n## 4. Decline / refusal markers in answers\n")
    L.append(md_table(["corpus", "any marker in answer", "marker in first paragraph"],
                      [[k, pct(S[k]["decline_any"]), pct(S[k]["decline_first_para"])] for k in keys]))
    L.append("\nMarkers: I can't / I won't / I'm not able / I will not / I'd rather not / not going to help (case-insensitive).")

    L.append("\n## 5. Value vocabulary (occurrences per 1k words)\n")
    L.append("**Answers**\n")
    L.append(md_table(["term"] + keys, [[t] + [f"{S[k]['vocab_answer'][t]:.2f}" for k in keys] for t in VALUE_VOCAB]
                      + [["**sum**"] + [f"{sum(S[k]['vocab_answer'].values()):.2f}" for k in keys]]))
    L.append("\n**User prompts**\n")
    L.append(md_table(["term"] + keys, [[t] + [f"{S[k]['vocab_user'][t]:.2f}" for k in keys] for t in VALUE_VOCAB]
                      + [["**sum**"] + [f"{sum(S[k]['vocab_user'].values()):.2f}" for k in keys]]))

    L.append("\n## 6. AI-centrality of user prompts\n")
    L.append(md_table(["corpus", "mentions an AI / model / assistant / agent / algorithm / automated system",
                       "addresses the assistant's own conduct (you/your/yourself near decide/refuse/comply/override)", "either"],
                      [[k, pct(S[k]["ai_mention"]), pct(S[k]["own_conduct"]), pct(S[k]["ai_mention_or_own"])] for k in keys]))
    if "ai" in S["ours"]["topics"]:
        L.append("\nours `meta.ai` (who the AI in the scenario is): " + ", ".join(f"{a}={c}" for a, c in S["ours"]["topics"]["ai"]))

    L.append("\n## 7. Diversity\n")
    L.append(md_table(["corpus", "NN cosine (mean/med/p10/p90)", "rows with NN > 0.6", "rows with NN > 0.8",
                       "all pairs > 0.6", "all pairs > 0.8", "distinct-4-gram ratio (answers)", "4-grams"],
                      [[k, dist_str(S[k]["nn_dist"], "{:.3f}"), pct(S[k]["nn_share"][0.6]), pct(S[k]["nn_share"][0.8]),
                        f"{100 * S[k]['pair_share'][0.6]:.3f}%", f"{100 * S[k]['pair_share'][0.8]:.3f}%",
                        f"{S[k]['distinct4']:.3f}", S[k]["total4"]] for k in keys]))
    L.append("\nTF-IDF (sklearn, sublinear tf, l2) over user prompts within each corpus; NN = each prompt's highest cosine to another prompt. "
             "Distinct-4-gram ratio falls with corpus size, so the size-matched row below subsamples every corpus to the smallest n (seed 0).")
    L.append(md_table(["corpus", "distinct-4-gram ratio at n=" + str(S["_matched_n"])],
                      [[k, f"{S[k]['distinct4_matched']:.3f}"] for k in keys]))
    for which, title in (("open_user", "user prompts"), ("open_answer", "answers")):
        L.append(f"\n**Top 15 opening 5-grams of {title}**\n")
        rows = []
        for i in range(15):
            rows.append([i + 1] + [f"{S[k][which][i][1]} · {S[k][which][i][0]}" if i < len(S[k][which]) else "" for k in keys])
        L.append(md_table(["#"] + keys, rows))
        L.append("\nShare of rows covered by top-15 openers: "
                 + ", ".join(f"{k} {pct(sum(c for _, c in S[k][which]) / S[k]['n'])}" for k in keys))

    L.append("\n## 8. Topic spread\n")
    L.append("Distinct values: " + "; ".join(f"{k}: " + ", ".join(f"{f}={v}" for f, v in S[k]["n_distinct"].items()) for k in keys) + "\n")
    L.append("**ours — top 20 domains**\n")
    L.append(md_table(["domain", "n"], S["ours"]["topics"]["domain"]))
    L.append("\n**ours — top 20 sectors**\n")
    L.append(md_table(["sector", "n"], S["ours"]["topics"]["sector"]))
    for k in ("terra3000", "sonnet1000"):
        L.append(f"\n**{k} — top 20 themes** (meta matched for {pct(S[k]['meta_matched'])} of rows)\n")
        L.append(md_table(["theme", "n"], S[k]["topics"]["theme"]))
    L.append("\n**Principle / trait distribution**\n")
    L.append("ours (9 traits):\n")
    L.append(md_table(["trait", "n"], S["ours"]["principles"]))
    for k in ("terra3000", "sonnet1000"):
        L.append(f"\n{k} (15 principles):\n")
        L.append(md_table(["principle", "n"], S[k]["principles"]))
    L.append(f"\nFigure: `{FIGURE}`\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------- figure
def make_figure(S):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.titlecolor": text, "axes.titleweight": "bold", "axes.titlesize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), facecolor="#fcfcfb")
    panels = [
        ("user prompt length (words)", lambda k: S[k]["words"]["user"], None),
        ("answer length (words)", lambda k: S[k]["words"]["answer"], None),
        ("reasoning length (words)", lambda k: S[k]["words"]["reasoning"], None),
        ("nearest-neighbour TF-IDF cosine, user prompts", lambda k: S[k]["nn"], None),
    ]
    for ax, (title, get, xlim) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        data = {k: np.asarray(get(k), dtype=float) for k, *_ in CORPORA}
        if xlim is None:
            hi = max(np.percentile(d, 99) for d in data.values() if d.size and d.max() > 0)
            xlim = (0, hi)
        bins = np.linspace(xlim[0], xlim[1], 41)
        for key, _, label, colour in CORPORA:
            d = data[key]
            if title.startswith("reasoning") and d.max() == 0:
                continue  # the two external corpora carry no reasoning trace; said in the legend
            ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, density=True, histtype="stepfilled", alpha=0.18, color=colour, lw=0)
            ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, density=True, histtype="step", color=colour, lw=1.6,
                    label=f"{label}  n={S[key]['n']}")
            ax.axvline(np.median(d), color=colour, lw=1, ls=(0, (3, 2)), alpha=0.9)
        ax.set_title(title, loc="left")
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.grid(axis="x", color=grid, lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if title.startswith("reasoning"):
            ax.text(0.02, 0.92, "terra3000 and sonnet1000 carry\nno reasoning trace (empty)", ha="left", va="top", color=muted,
                    transform=ax.transAxes, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Difficult-advice corpora: length and prompt-diversity distributions (density; dashed = median; x clipped at p99)",
                 x=0.01, ha="left", fontsize=10.5, color=text, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIGURE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    tok, tok_name = load_tokenizer()
    if tok is None:
        print("WARNING: no Qwen tokenizer in the HF cache; reporting word counts only", file=sys.stderr)
    S, T = {}, {}
    for key, fname, _, _ in CORPORA:
        rows = load(IN_DIR / fname)
        S[key], T[key] = analyse(key, rows, tok)
        print(f"{key}: {len(rows)} rows analysed", file=sys.stderr)
    n_min = min(S[k]["n"] for k in S)
    rng = np.random.default_rng(SEED)
    for k in list(S):
        idx = rng.choice(S[k]["n"], size=n_min, replace=False)
        S[k]["distinct4_matched"], _ = distinct_ngrams([T[k]["answer"][i] for i in sorted(idx)])
    S["_matched_n"] = n_min

    report = build_report(S, tok_name)
    REPORT.write_text(report)
    make_figure(S)
    print(report)
    print(f"\nwrote {REPORT}\nwrote {FIGURE}", file=sys.stderr)


if __name__ == "__main__":
    main()
