# ABOUTME: Renders the two four-arm pages: "Four Arms, Same Questions" (the living comparison,
# ABOUTME: ODCV rows included) and "Four Arms Browser" (every row, four replies side by side).

"""Run: uv run python scratch/sonnet_concise/build_four_way_page.py [<browser artifact url>]

Writes output/sonnet_concise/pages/four_arms_same_questions.html and four_arms_browser.html.
To update the published pages keep their URLs: publish with the Artifact tool passing
url=https://claude.ai/code/artifact/71de623d-571d-4c33-894c-2ca9f0f49681 (comparison) and
url=https://claude.ai/code/artifact/048d738c-0bf1-4f04-9785-d28362af6c81 (browser).
When arm C's ODCV lands, replace the `pend` cells in the ODCV table below with its numbers
and re-run; everything else recomputes from the corpora and judge files.
"""
import base64
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import os  # noqa: E402

os.chdir(WT)
from scratch.three_way.norm import ORDER, load, load_judged, judged_common, reply, trace  # noqa: E402

BROWSER_URL = sys.argv[1] if len(sys.argv) > 1 else "https://claude.ai/code/artifact/048d738c-0bf1-4f04-9785-d28362af6c81"
OUT_DIR = WT / "output/sonnet_concise/pages"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FW = WT / "output/sonnet_concise/four_way"
PNG = sorted((WT / "output/sonnet_concise").glob("lengths_four_arms_*.png"))[-1]
ODCV_PNG = FW / "odcv_generators_65cells_bars_20260825_192328.png"

ARMS = [  # key, label, css class, author line
    (
        "sonnet",
        "A · da716",
        "sonnet",
        "Haiku 4.5 draft → Sonnet 5 rewrite (the baseline recipe)",
    ),
    (
        "capped",
        "C · capped",
        "cap",
        "same drafts → Sonnet 5 rewrite under a 220/270-word cap",
    ),
    ("grok", "B · grok", "grok", "grok-4.6 drafts and rewrites"),
    ("gpt", "D · gpt", "gpt", "gpt-5.6-luna draft → gpt-5.6-terra rewrite"),
]
KEYS = [a[0] for a in ARMS]
CLS = {a[0]: a[2] for a in ARMS}
LABEL = {a[0]: a[1] for a in ARMS}

# ---------------- data ----------------
C, IDS = load(normalise=False)  # 678 shared, raw text for reading
CN, _ = load(normalise=True)  # normalised for counting
byc = load_judged(IDS)
_, JC = judged_common(byc)  # 677 judged in every corpus


def wc(t):
    return len(t.split())


def q(xs):
    xs = sorted(xs)
    return [
        xs[min(int(len(xs) * p), len(xs) - 1)] for p in (0.10, 0.25, 0.50, 0.75, 0.90)
    ]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def pct(rows, f, v=True):
    return 100 * sum(1 for r in rows if r.get(f) == v) / len(rows)


def mean(rows, f):
    return sum((r.get(f) or 0) for r in rows) / len(rows)


D = {c: [byc[c][k] for k in JC] for c in KEYS}
R = {c: [r for r in D[c] if r["stance"] != "no_shortcut"] for c in KEYS}


def mcnemar(a, b, pred):
    only_a = sum(1 for k in JC if pred(byc[a][k]) and not pred(byc[b][k]))
    only_b = sum(1 for k in JC if pred(byc[b][k]) and not pred(byc[a][k]))
    tot = only_a + only_b
    if tot == 0:
        return only_a, only_b, 1.0
    chi = (abs(only_a - only_b) - 1) ** 2 / tot
    return only_a, only_b, math.erfc(math.sqrt(chi / 2))


def esc(t):
    return html.escape(t or "")


def paras(t):
    return "".join(
        f"<p>{esc(p.strip())}</p>" for p in (t or "").split("\n\n") if p.strip()
    )


def table(header, rows, hi_col=None):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = []
    for r in rows:
        cells = "".join(f"<td>{c}</td>" for c in r)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def armhead(first="metric"):
    return [first] + [f'<span class="dot {CLS[c]}"></span>{LABEL[c]}' for c in KEYS]


def fmt(v, d=1):
    return f"{v:.{d}f}"


# ---------- length table ----------
len_rows = []
for idx, name, fn in ((0, "Reasoning words", trace), (1, "Reply words", reply)):
    for c in KEYS:
        qq = q([wc(fn(C[c][k])) for k in IDS])
        len_rows.append(
            [f'<span class="dot {CLS[c]}"></span>{name} · {LABEL[c]}']
            + [str(x) for x in qq]
        )
length_table = table(["", "p10", "p25", "median", "p75", "p90"], len_rows)

# ---------- judged tables ----------
stance_rows = []
for v in ("refuses", "partial", "complies", "no_shortcut"):
    stance_rows.append([f"stance = {v}"] + [fmt(pct(D[c], "stance", v)) for c in KEYS])
stance_rows.append(
    ["<b>decline rate (refuses + partial)</b>"]
    + [
        f"<b>{fmt(pct(D[c], 'stance', 'refuses') + pct(D[c], 'stance', 'partial'))}</b>"
        for c in KEYS
    ]
)
stance_rows.append(
    ["leak (complies + partial)"]
    + [
        fmt(pct(D[c], "stance", "complies") + pct(D[c], "stance", "partial"))
        for c in KEYS
    ]
)
stance_table = table(armhead("% of 677 judged scenarios"), stance_rows)

form_rows = [
    ["refusal is explicit, first person"]
    + [fmt(pct(R[c], "refusal_explicit")) for c in KEYS],
    ["refusal names the specific action"]
    + [fmt(pct(R[c], "refusal_names_action")) for c in KEYS],
    ["refusal in the opening sentences"]
    + [fmt(pct(R[c], "refusal_position", "opening")) for c in KEYS],
    ["refusal after some context"]
    + [fmt(pct(R[c], "refusal_position", "after_context")) for c in KEYS],
    ["refusal tone: explained"]
    + [fmt(pct(R[c], "refusal_tone", "explained")) for c in KEYS],
    ["refusal tone: apologetic"]
    + [fmt(pct(R[c], "refusal_tone", "apologetic")) for c in KEYS],
]
form_table = table(armhead("% of rows that needed a refusal (~600)"), form_rows)

alt_rows = [
    ["alternatives per reply (mean)"]
    + [fmt(mean(D[c], "n_alternatives"), 2) for c in KEYS],
    ["≥1 alternative"]
    + [
        fmt(
            100 * sum(1 for r in D[c] if (r.get("n_alternatives") or 0) > 0) / len(D[c])
        )
        for c in KEYS
    ],
    ["alternatives specific enough to act on"]
    + [fmt(pct(D[c], "alternatives_specific")) for c in KEYS],
    ["alternatives stated tersely"]
    + [fmt(pct(D[c], "alternatives_terse")) for c in KEYS],
    ["kind: assistant offers to do the work"]
    + [
        fmt(
            100
            * sum(
                1
                for r in D[c]
                if "assistant_does_work" in (r.get("alternative_kinds") or [])
            )
            / len(D[c])
        )
        for c in KEYS
    ],
    ["kind: names a person or body"]
    + [
        fmt(
            100
            * sum(
                1
                for r in D[c]
                if "name_person_or_body" in (r.get("alternative_kinds") or [])
            )
            / len(D[c])
        )
        for c in KEYS
    ],
    ["kind: legitimate process"]
    + [
        fmt(
            100
            * sum(
                1
                for r in D[c]
                if "legitimate_process" in (r.get("alternative_kinds") or [])
            )
            / len(D[c])
        )
        for c in KEYS
    ],
    ["alternatives per 1,000 reply chars"]
    + [
        fmt(
            1000
            * sum((r.get("n_alternatives") or 0) for r in D[c])
            / sum(len(reply(C[c][k])) for k in JC),
            2,
        )
        for c in KEYS
    ],
]
alt_table = table(armhead("% unless stated"), alt_rows)

pairs = [
    ("capped", "sonnet"),
    ("capped", "grok"),
    ("capped", "gpt"),
    ("grok", "sonnet"),
    ("gpt", "sonnet"),
    ("gpt", "grok"),
]
tests = [
    ("refuses (clean decline)", lambda r: r["stance"] == "refuses"),
    ("complies", lambda r: r["stance"] == "complies"),
    ("leak (complies or partial)", lambda r: r["stance"] in ("complies", "partial")),
    ("explicit first-person refusal", lambda r: r.get("refusal_explicit") is True),
]
mc_rows = []
for a, b in pairs:
    cells = [
        f'<span class="dot {CLS[a]}"></span>{LABEL[a]} vs <span class="dot {CLS[b]}"></span>{LABEL[b]}'
    ]
    for _, pred in tests:
        oa, ob, p = mcnemar(a, b, pred)
        pa = 100 * sum(1 for k in JC if pred(byc[a][k])) / len(JC)
        pb = 100 * sum(1 for k in JC if pred(byc[b][k])) / len(JC)
        sig = ' <b class="sig">*</b>' if p < 0.05 else ""
        cells.append(
            f"{pa:.1f} vs {pb:.1f}<br><small>p={p:.3f}{sig} · {oa}/{ob}</small>"
        )
    mc_rows.append(cells)
mc_table = table(["pair"] + [t for t, _ in tests], mc_rows)

# per-trait refuses
names = {}
bt = defaultdict(list)
for k in JC:
    md = C["capped"][k]["metadata"]
    names[md["trait_id"]] = md["trait_name"]
    bt[md["trait_id"]].append(k)
trait_rows = []
for t in sorted(bt, key=lambda x: int(x[1:])):
    ks = bt[t]
    v = {
        c: 100 * sum(1 for k in ks if byc[c][k]["stance"] == "refuses") / len(ks)
        for c in KEYS
    }
    trait_rows.append(
        [f"{t} · {esc(names[t][:52])}", str(len(ks))]
        + [fmt(v[c]) for c in KEYS]
        + [f"{v['capped'] - v['sonnet']:+.1f}"]
    )
trait_table = table(
    ["trait", "n"]
    + [f'<span class="dot {CLS[c]}"></span>{LABEL[c]}' for c in KEYS]
    + ["C − A"],
    trait_rows,
)


# ---------- text-output parsing for style rows ----------
def row4(path, prefix):
    for line in (FW / path).read_text().splitlines():
        if line.strip().startswith(prefix):
            nums = re.findall(r"-?\d+(?:\.\d+)?", line[len(line) - 60 :])
            toks = [t for t in line.split() if re.fullmatch(r"-?\d+(?:\.\d+)?%?x?", t)]
            vals = [float(t.rstrip("%x")) for t in toks][-4:]
            if len(vals) == 4:
                return vals
    return None


def reorder(
    vals,
):  # files print sonnet, grok, gpt, capped -> page order sonnet, capped, grok, gpt
    s, g, p, c = vals
    return [s, c, g, p]


mt = json.load(open(WT / "scratch/gpt_voice/metrics_table.json"))
mt_map = {lab: vals for lab, vals in mt}
STYLE = [
    ("reply words (median)", "metrics", "reply words (median)"),
    ("trace words (median)", "metrics", "trace words (median)"),
    ("contractions per 1k words", "metrics", "contractions per 1k words"),
    ("hedges per 1k words", "metrics", "hedges per 1k words"),
    ("em-dash per 1k words", "metrics", "em-dash per 1k words"),
    ("question marks per 1k words", "metrics", "question marks per 1k words"),
    (
        "imperative sentences % of reply sentences",
        "metrics",
        "imperative sentences % (of reply sentences)",
    ),
    ("reply ends on a question %", "metrics", "reply ends on '?' %"),
    ("any list in the reply %", "metrics", "any list %"),
    ("bold anywhere %", "metrics", "bold anywhere %"),
    (
        "% of reply words inside list/heading lines",
        "metrics",
        "% of reply WORDS inside list/heading lines",
    ),
    (
        "trace: explicit first-person uncertainty %",
        "metrics",
        "explicit first-person uncertainty %",
    ),
    ("trace: 'I should' %", "metrics", "trace 'I should' %"),
    (
        "repeated 5-gram share % (reply)",
        "metrics",
        "repeated 5-gram token share % (reply)",
    ),
]
style_rows = []
for shown, src, key in STYLE:
    if key in mt_map:
        vals = mt_map[key]
        style_rows.append([shown] + reorder([str(v) for v in vals]))
style_table = table(
    armhead("from scratch/gpt_voice/metrics.py (678 shared)"), style_rows
)

LEX = [
    ("prose share of reply chars %", "length_decomp.txt", "prose"),
]
ld_rows = []
for label, prefix in (
    ("hedge words", "hedge ("),
    ("conditional 'if'", "conditional 'if'"),
    ("risk nouns", "risk nouns"),
    ("process nouns", "process nouns"),
    ("named roles", "named roles"),
    ("second person 'you'", "second person"),
    ("first person 'I'", "first person"),
):
    v = row4("length_decomp.txt", prefix)
    if v:
        ld_rows.append([label + " · per 1k chars"] + [fmt(x, 2) for x in reorder(v)])
for label, prefix in (
    ("offer idiom present %", "offer idiom present"),
    ("ANY drafted artifact present %", "ANY drafted artifact present"),
):
    v = row4("does_the_work.txt", prefix)
    if v:
        ld_rows.append([label] + [fmt(x) for x in reorder(v)])
for label, prefix in (
    ("first-person refusal modal %", "A first-person modal"),
    ("'instead' redirect %", "J 'instead' redirect"),
    ("'shouldn't' / 'should not' %", "H 'shouldn't'"),
    ("names the act as wrongdoing %", "G names the act"),
):
    v = row4("refusal_forms.txt", prefix)
    if v:
        ld_rows.append([label] + [fmt(x) for x in reorder(v)])
lex_table = table(armhead("rate or % of replies"), ld_rows)


def pre(path):
    return f'<details class="raw"><summary>{path}</summary><pre>{esc((FW / path).read_text())}</pre></details>'


raw_blocks = "".join(
    pre(p)
    for p in (
        "agg.txt",
        "stats.txt",
        "by_trait.txt",
        "metrics.txt",
        "length_decomp.txt",
        "refusal_forms.txt",
        "does_the_work.txt",
        "substance.txt",
    )
)
full_metrics = table(
    armhead("all 86 rows"), [[esc(lab)] + reorder(vals) for lab, vals in mt]
)

png_uri = "data:image/png;base64," + base64.b64encode(PNG.read_bytes()).decode()
odcv_uri = "data:image/png;base64," + base64.b64encode(ODCV_PNG.read_bytes()).decode()

CSS = """
:root{--bg:#F5F6F8;--surface:#FFFFFF;--surface-2:#EEF0F3;--text:#1C2027;--muted:#5D6572;--line:#D9DEE5;--accent:#0D9A80;
--sonnet:#4F53B8;--cap:#0D9A80;--grok:#A8690F;--gpt:#B0417A;--sonnet-bg:#ECEDF9;--cap-bg:#E4F3F0;--grok-bg:#F7EEDD;--gpt-bg:#F8E8F0;--code-bg:#F0F2F5;--focus:#4F53B8;--pending:#B0813A}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171C;--surface:#1B1F26;--surface-2:#22272F;--text:#E7E9ED;--muted:#9AA3AF;--line:#2C323B;--accent:#2FA38F;
--sonnet:#8487E6;--cap:#2FA38F;--grok:#B0813A;--gpt:#D06AA0;--sonnet-bg:#22243A;--cap-bg:#17302C;--grok-bg:#33281A;--gpt-bg:#36202C;--code-bg:#1E232B;--focus:#8487E6;--pending:#B0813A}}
:root[data-theme="dark"]{--bg:#14171C;--surface:#1B1F26;--surface-2:#22272F;--text:#E7E9ED;--muted:#9AA3AF;--line:#2C323B;--accent:#2FA38F;
--sonnet:#8487E6;--cap:#2FA38F;--grok:#B0813A;--gpt:#D06AA0;--sonnet-bg:#22243A;--cap-bg:#17302C;--grok-bg:#33281A;--gpt-bg:#36202C;--code-bg:#1E232B;--focus:#8487E6;--pending:#B0813A}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:"Source Sans 3","Segoe UI",system-ui,sans-serif;font-size:17px;line-height:1.55;-webkit-font-smoothing:antialiased}
h1,h2,h3{font-family:"Newsreader",Georgia,"Times New Roman",serif;font-weight:500;letter-spacing:-0.01em;text-wrap:balance;margin:0}
h1{font-size:2.6rem;line-height:1.1} h2{font-size:1.7rem;margin:0 0 .6rem} h3{font-size:1.2rem;margin:1.2rem 0 .4rem}
code,pre,.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
code{background:var(--code-bg);padding:.05em .35em;border-radius:3px;font-size:.88em}
a{color:var(--accent)} a:focus-visible,button:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px} .prose{max-width:74ch}
.eyebrow{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}
header.top{padding:56px 0 28px;border-bottom:1px solid var(--line)} header.top .sub{color:var(--muted);max-width:74ch;margin-top:12px;font-size:1.05rem}
nav.toc{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
nav.toc ul{list-style:none;margin:0;padding:0;display:flex;gap:22px;overflow-x:auto;font-size:.9rem}
nav.toc a{display:block;padding:12px 0;color:var(--muted);text-decoration:none;white-space:nowrap;font-weight:600} nav.toc a:hover{color:var(--text)}
section{padding:40px 0;border-bottom:1px solid var(--line)} section:last-of-type{border-bottom:0}
.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:22px 0 6px}
.arm{background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--line);padding:16px 18px;border-radius:4px}
.arm.sonnet{border-top-color:var(--sonnet)} .arm.cap{border-top-color:var(--cap)} .arm.grok{border-top-color:var(--grok)} .arm.gpt{border-top-color:var(--gpt)}
.arm .mr{font-family:"Newsreader",Georgia,serif;font-size:2rem;line-height:1.1;margin:6px 0 2px} .arm .mr.pending{color:var(--pending);font-size:1.3rem;padding-top:6px}
.arm small{color:var(--muted);display:block} .arm .who{margin:4px 0 8px;font-size:.92rem}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.95rem} .tablewrap{overflow-x:auto;margin:10px 0 6px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:top} th{font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
td:not(:first-child),th:not(:first-child){text-align:right} td small{color:var(--muted);font-size:.78rem} b.sig{color:var(--accent)}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px}
.dot.sonnet{background:var(--sonnet)} .dot.cap{background:var(--cap)} .dot.grok{background:var(--grok)} .dot.gpt{background:var(--gpt)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.kv div{background:var(--surface);border:1px solid var(--line);padding:12px 14px;border-radius:4px} .kv b{display:block;font-family:"Newsreader",Georgia,serif;font-size:1.5rem;font-weight:500} .kv small{color:var(--muted)}
ul.tight{margin:.4rem 0;padding-left:1.2rem} ul.tight li{margin:.3rem 0}
figure{margin:18px 0} figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff} figcaption{font-size:.85rem;color:var(--muted);margin-top:6px}
details.raw{margin:8px 0;border:1px solid var(--line);border-radius:4px;background:var(--surface)} details.raw summary{cursor:pointer;padding:8px 12px;font-family:"JetBrains Mono",monospace;font-size:.82rem;color:var(--muted)}
details.raw pre{margin:0;padding:12px 14px;overflow-x:auto;font-size:.78rem;line-height:1.45;border-top:1px solid var(--line)}
.callout{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);padding:16px 20px;border-radius:4px;margin:16px 0}
.callout.pending{border-left-color:var(--pending)}
.odcv td.pend{color:var(--pending);font-weight:600}
footer{padding:30px 0 60px;color:var(--muted);font-size:.9rem}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
"""
FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap">'

browser_link = (
    f'<a href="{BROWSER_URL}">the row browser</a>'
    if BROWSER_URL
    else "the row browser (published separately)"
)

page = f"""<title>Four Arms, Same Questions</title>
{FONTS}<style>{CSS}</style>
<header class="top"><div class="wrap">
  <div class="eyebrow">Generator ablation · difficult advice · {len(IDS)} shared questions · living page, updated 2026-08-26</div>
  <h1 style="margin-top:10px">Four Arms, Same Questions</h1>
  <p class="sub">Four difficult-advice corpora answer the same {len(IDS)} prompts and differ only in who wrote the assistant turn — and, for arm C, how long it was allowed to be. This page holds the corpus-level comparison and the ODCV results as they land.</p>
  <div class="arms">
    <div class="arm sonnet"><div class="eyebrow">A · da716</div><div class="who">Haiku 4.5 draft → Sonnet 5 rewrite</div><div class="mr">16.3%</div><small>ODCV MR [10.0, 21.8] · sev 0.76 · n=257 transcripts</small><small>reply 452w · reasoning 479w (median, 678 shared)</small></div>
    <div class="arm cap"><div class="eyebrow">C · capped Sonnet</div><div class="who">same drafts → Sonnet 5, capped 220/270 words</div><div class="mr pending">not yet trained</div><small>ODCV: next to run (train ≈$23, eval ≈$10)</small><small>reply 283w · reasoning 238w</small></div>
    <div class="arm grok"><div class="eyebrow">B · grok</div><div class="who">grok-4.6 drafts and rewrites</div><div class="mr">7.8%</div><small>ODCV MR [3.6, 13.6] · sev 0.35 · n=129</small><small>reply 268w · reasoning 218w</small></div>
    <div class="arm gpt"><div class="eyebrow">D · gpt</div><div class="who">gpt-5.6-luna draft → gpt-5.6-terra rewrite</div><div class="mr">25.2%</div><small>ODCV MR [15.1, 34.9] · sev 1.07 · n=127</small><small>reply 614w · reasoning 313w</small></div>
  </div>
</div></header>
<nav class="toc"><div class="wrap"><ul>
  <li><a href="#answer">The question</a></li><li><a href="#length">Length</a></li><li><a href="#refusal">Refusal (judged)</a></li><li><a href="#style">Style &amp; structure</a></li><li><a href="#odcv">ODCV</a></li><li><a href="#rows">Rows</a></li><li><a href="#raw">Raw outputs</a></li>
</ul></div></nav>

<section id="answer"><div class="wrap">
  <h2>Does shortening Sonnet cost anything besides length?</h2>
  <div class="callout"><p style="margin:0"><b>No, on refusal.</b> A blind judge (gpt-5.6-terra, temperature 0, same rubric for all four) puts capped Sonnet at <b>83.6% clean refusals vs 83.8%</b> for unconstrained Sonnet, complies 1.0% vs 1.2%, and every paired McNemar test between C and A comes out at p ≈ 1.0. Capped Sonnet also beats GPT on every stance metric (p &lt; 0.03) and matches grok. Per trait, C stays within ±5 points of A everywhere.</p>
  <p style="margin:10px 0 0"><b>What condensing did move</b> — all small, all in the direction of grok's style: 0.5 fewer alternatives per reply (4.1 vs 4.6), alternatives stated tersely 12% vs 2%, "I can draft that for you"-type offers 58% vs 67%, refusal naming the specific act 70% vs 74%, and the refusal arrives in the opening sentences more often (55% vs 43%). Sonnet's voice otherwise survives: contractions, second person, hedges, questions and "instead" redirects sit at Sonnet's rates, not grok's or GPT's.</p></div>
  <div class="kv">
    <div><b>83.6 / 83.8</b><small>clean-refusal %, capped vs Sonnet (p = 1.0)</small></div>
    <div><b>1.05x · 1.10x</b><small>capped / grok length, reply · reasoning (paired medians)</small></div>
    <div><b>0.42</b><small>length-only AUC, capped vs grok (0.864 before the cap)</small></div>
    <div><b>4.1 vs 4.6</b><small>alternatives per reply, capped vs Sonnet (grok 5.1, GPT 7.0)</small></div>
  </div>
  <div class="prose"><p>Why this matters: on ODCV the three trained arms order exactly by response length — GPT 25.2% (longest) → Sonnet 16.3% → grok 7.8% (shortest). Arm C is Sonnet at grok's length. If it lands near 7.8%, length carried the ordering; if near 16.3%, the generator did. The corpus evidence above says C is a clean test: it moved length and little else.</p></div>
</div></section>

<section id="length"><div class="wrap">
  <h2>Length — reasoning and reply, separately</h2>
  <figure><img src="{png_uri}" alt="Reasoning and reply word counts for four arms as p10 to p90 ranges with medians; capped Sonnet sits at grok's median with a tighter spread; GPT is longest."><figcaption>Rows are the four arms on the same {len(IDS)} scenarios. Line p10–p90, band p25–p75, dot = median. Dashed line = the cap.</figcaption></figure>
  {length_table}
  <div class="prose"><ul class="tight">
    <li>Capped Sonnet's <b>spread</b> is much tighter than any unconstrained arm (reply p10–p90 258–308 vs grok 169–433): the cap fixes the median, not the variance.</li>
    <li>GPT's extra length is furniture more than prose — 29% of its reply characters sit in list items vs 13–16% for the others, and 74% of its replies contain a drafted artifact (blockquote, table, template) vs 2–3%. See <a href="#style">style &amp; structure</a>.</li>
  </ul></div>
</div></section>

<section id="refusal"><div class="wrap">
  <h2>Refusal, alternatives, and how the decline is worded — blind judge</h2>
  <div class="prose"><p>gpt-5.6-terra, temperature 0, reads the situation, the shortcut the user is steering toward, the user turn and the reply — nothing about which corpus it came from — and returns a fixed JSON rubric (<code>scratch/three_way/judge.py</code>, verbatim across all four arms). n = {len(JC)} scenarios judged in every corpus. Judge caveat: it is an OpenAI model scoring one OpenAI corpus; a 200-scenario cross-check with gemini-3.1-pro exists for the three-way (<code>scratch/three_way/agg_neutral.py</code>) and can be extended to C.</p></div>
  <h3>Stance</h3>{stance_table}
  <h3>Paired significance (McNemar, same scenarios) — "a vs b", discordant a-only / b-only</h3>{mc_table}
  <h3>How the refusal is worded</h3>{form_table}
  <h3>Alternatives offered</h3>{alt_table}
  <h3>Clean-refusal rate per principle</h3>{trait_table}
</div></section>

<section id="style"><div class="wrap">
  <h2>Style and structure — rates, so length-neutral</h2>
  <div class="prose"><p>The corpus-level fingerprint of each author. What to look for: capped Sonnet sitting in Sonnet's column on voice (contractions, second person, questions, "instead") while sitting in grok's column on length. GPT is the outlier on structure.</p></div>
  <h3>Voice and shape</h3>{style_table}
  <h3>Lexical density and refusal constructions</h3>{lex_table}
  <details class="raw"><summary>all 86 voice/structure metrics (scratch/gpt_voice/metrics.py)</summary>{full_metrics}</details>
</div></section>

<section id="odcv"><div class="wrap">
  <h2>ODCV-Bench — the number arm C exists for</h2>
  <div class="callout pending"><p style="margin:0"><b>Arm C is generated, mixed and configured but not trained.</b> Next: <code>scripts/gpu/runpod_train.py up</code> with <code>configs/train/lora_qwen36_t2_9284_sonnetconcise703_paired_2xh200.yaml</code>, then <code>configs/eval/odcv_bench_t2_9284_sonnetconcise703_r64_paired_2x65.yaml</code> on the same 65 cells × 2 rollouts. Its row below fills in when that lands.</p></div>
  <div class="tablewrap"><table class="odcv"><thead><tr><th>arm</th><th>MR</th><th>95% CI</th><th>severity</th><th>mandated</th><th>incentivized</th><th>n</th></tr></thead><tbody>
    <tr><td><span class="dot gpt"></span>D · GPT (gpt-5.6-luna → terra)</td><td>25.2%</td><td>[15.1, 34.9]</td><td>1.07</td><td>22.1%</td><td>28.8%</td><td>127</td></tr>
    <tr><td><span class="dot sonnet"></span>A · da716 (Haiku → Sonnet 5)</td><td>16.3%</td><td>[10.0, 21.8]</td><td>0.76</td><td>12.4%</td><td>20.8%</td><td>257</td></tr>
    <tr><td><span class="dot cap"></span>C · capped Sonnet</td><td class="pend">pending</td><td class="pend">—</td><td class="pend">—</td><td class="pend">—</td><td class="pend">—</td><td class="pend">—</td></tr>
    <tr><td><span class="dot grok"></span>B · grok-4.6</td><td>7.8%</td><td>[3.6, 13.6]</td><td>0.35</td><td>5.7%</td><td>10.2%</td><td>129</td></tr>
    <tr><td>Qwen3.6-27B base fp8 (no SFT)</td><td>36.9%</td><td>[21.4, 53.6]</td><td>1.37</td><td>40.0%</td><td>33.3%</td><td>65</td></tr>
    <tr><td>Table2-only 9,284 (0% synthetic)</td><td>43.9%</td><td>[37.5, 53.1]</td><td>1.87</td><td>46.1%</td><td>41.3%</td><td>305</td></tr>
  </tbody></table></div>
  <figure><img src="{odcv_uri}" alt="ODCV misalignment rate by generator on the same 65 cells: GPT highest, then Sonnet, then grok; base and Table2-only controls above all three."><figcaption>Three trained arms + controls, same 65 cells, judges grok-4.20 + gemini-3.1-pro (2026-08-25). Arm C joins this plot after its eval.</figcaption></figure>
</div></section>

<section id="rows"><div class="wrap">
  <h2>Read the rows</h2>
  <div class="prose"><p>Every scenario with all four replies side by side, the judge's stance per arm, and filters by principle and stance — in {browser_link}. The 27-row smoke review with Sonnet's own change notes is on the <a href="https://claude.ai/code/artifact/49a44a99-309e-4de1-acc6-802b7a969ecb">Capped Sonnet Rewrite</a> page.</p></div>
</div></section>

<section id="raw"><div class="wrap">
  <h2>Raw outputs</h2>
  <div class="prose"><p>The scripts' own tables, unedited (column order sonnet · grok · gpt · capped).</p></div>
  {raw_blocks}
</div></section>
<footer><div class="wrap">Corpora: <code>LASR-Callum/2026-08-13-difficult-advice-v2</code> (A) · <code>…/2026-08-26-difficult-advice-sonnet-concise-716</code> (C) · <code>…/2026-08-21-difficult-advice-grok-responder-716</code> (B) · <code>…/2026-08-25-difficult-advice-gpt-responder-716</code> (D). Word counts by whitespace split on the {len(IDS)} shared scenario ids; judged tables on the {len(JC)} judged in every corpus. Scripts: <code>scratch/three_way/</code>, <code>scratch/gpt_voice/</code>, <code>scratch/sonnet_concise/</code>, branch <code>worktree-sonnet-concise-rewrite</code>.</div></footer>
"""
(OUT_DIR / "four_arms_same_questions.html").write_text(page)
print("comparison page:", len(page) // 1024, "KB")

# ---------------- browser ----------------
allC = {}
from scratch.three_way.norm import load_all  # noqa: E402

FULL = load_all(normalise=False)
cap_ids = sorted(FULL["capped"])
rows = []
for k in cap_ids:
    r = FULL["capped"][k]
    md = r["metadata"]
    sysm = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
    usr = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
    arms = {}
    for c in KEYS:
        rr = FULL[c].get(k)
        if not rr:
            continue
        j = byc.get(c, {}).get(k) or {}
        arms[c] = {
            "r": trace(rr),
            "a": reply(rr),
            "st": j.get("stance", ""),
            "na": j.get("n_alternatives"),
            "rw": wc(trace(rr)),
            "aw": wc(reply(rr)),
        }
    rows.append(
        {
            "sid": k,
            "t": md.get("trait_id", ""),
            "tn": md.get("trait_name", ""),
            "dom": str(md.get("domain", "")),
            "sc": str(md.get("shortcut", "")),
            "user": usr,
            "arms": arms,
        }
    )
data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")

BCSS = (
    CSS
    + """
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:14px 0 18px}
.controls input,.controls select{font:inherit;font-size:.92rem;padding:7px 10px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text)}
.controls input{min-width:260px} .count{color:var(--muted);font-size:.9rem}
.legend{display:flex;gap:16px;font-size:.85rem;color:var(--muted);flex-wrap:wrap}
.row{background:var(--surface);border:1px solid var(--line);border-radius:4px;margin:8px 0}
.row>summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:140px 1fr auto;gap:12px;align-items:center;padding:10px 14px;font-size:.9rem}
.row>summary::-webkit-details-marker{display:none} .row>summary:hover{background:var(--surface-2)}
.row[open]>summary{border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--surface);z-index:2}
.sid{font-family:"JetBrains Mono",monospace;font-size:.78rem;color:var(--muted)} .meta b{font-weight:600} .meta span{color:var(--muted)}
.chips{display:flex;gap:6px;flex-wrap:wrap} .chip{display:inline-flex;align-items:center;gap:5px;font-size:.74rem;padding:2px 7px;border-radius:3px;background:var(--surface-2);color:var(--text);white-space:nowrap}
.chip i{width:8px;height:8px;border-radius:2px;display:inline-block} .chip.sonnet i{background:var(--sonnet)} .chip.cap i{background:var(--cap)} .chip.grok i{background:var(--grok)} .chip.gpt i{background:var(--gpt)}
.chip.st-complies{outline:2px solid #C0392B} .chip.st-partial{outline:1px dashed var(--pending)}
.body{padding:16px 14px 20px} .shortcut{margin:0 0 10px;color:var(--muted)} .shortcut b{color:var(--text)}
details.sys{margin:8px 0} details.sys summary{cursor:pointer;font-size:.85rem;color:var(--muted);font-weight:600} details.sys .txt{font-size:.9rem;color:var(--muted);max-width:80ch}
.user{background:var(--surface-2);padding:12px 16px;border-radius:4px;margin:12px 0 18px;max-width:80ch}
h5{font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 4px}
.txt p{margin:0 0 .7em} .txt p:last-child{margin-bottom:0}
.cols{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px} @media (max-width:1100px){.cols{grid-template-columns:repeat(2,minmax(0,1fr))}} @media (max-width:700px){.cols{grid-template-columns:1fr}}
.col{border:1px solid var(--line);border-radius:4px;padding:12px 14px;font-size:.92rem;min-width:0} .col.sonnet{background:var(--sonnet-bg)} .col.cap{background:var(--cap-bg)} .col.grok{background:var(--grok-bg)} .col.gpt{background:var(--gpt-bg)}
.col header{margin-bottom:10px} .col h4{margin:0;font-size:.92rem;font-weight:600} .col .lens{font-size:.78rem;color:var(--muted)} .col h5{margin-top:10px}
"""
)
browser = f"""<title>Four Arms Browser</title>
{FONTS}<style>{BCSS}</style>
<header class="top"><div class="wrap">
  <div class="eyebrow">Generator ablation · difficult advice · every row · 2026-08-26</div>
  <h1 style="margin-top:10px">Four Arms Browser</h1>
  <p class="sub">All {len(rows)} capped-Sonnet rows, each with the Sonnet, grok and GPT answers to the same prompt where they exist ({len(IDS)} shared), and the blind judge's stance per arm. Rows render when opened; system prompts are omitted here (they are on the HF datasets).</p>
  <div class="controls">
    <input id="q" type="search" placeholder="search user message, domain, id…" aria-label="search">
    <select id="trait" aria-label="principle"><option value="">all principles</option></select>
    <select id="stance" aria-label="stance filter"><option value="">any stance</option><option value="disagree">arms disagree on stance</option><option value="leak">any arm complies/partial</option><option value="cap-leak">capped complies/partial</option></select>
    <span class="count" id="count"></span>
  </div>
  <div class="legend"><span><i class="dot sonnet"></i>A · da716 Sonnet</span><span><i class="dot cap"></i>C · capped Sonnet</span><span><i class="dot grok"></i>B · grok</span><span><i class="dot gpt"></i>D · gpt</span><span>chip = judge stance · outlined red = complies · dashed = partial</span></div>
</div></header>
<section><div class="wrap" id="list"></div></section>
<script id="data" type="application/json">{data_json}</script>
<script>
(function(){{
  const rows = JSON.parse(document.getElementById('data').textContent);
  const ARMS = [['sonnet','A · da716 Sonnet','sonnet'],['capped','C · capped Sonnet','cap'],['grok','B · grok-4.6','grok'],['gpt','D · gpt-5.6','gpt']];
  const list = document.getElementById('list'), q = document.getElementById('q'), tsel = document.getElementById('trait'), ssel = document.getElementById('stance'), count = document.getElementById('count');
  const traits = {{}}; rows.forEach(r => traits[r.t] = r.tn);
  Object.keys(traits).sort((a,b)=>parseInt(a.slice(1))-parseInt(b.slice(1))).forEach(t => {{ const o = document.createElement('option'); o.value = t; o.textContent = t + ' · ' + traits[t].slice(0,60); tsel.appendChild(o); }});
  const esc = s => (s||'').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
  const paras = s => (s||'').split(/\\n\\s*\\n/).filter(p=>p.trim()).map(p=>'<p>'+esc(p.trim())+'</p>').join('');
  function chips(r){{ return ARMS.map(([k,l,c]) => {{ const a = r.arms[k]; if(!a) return ''; const st = a.st || '—'; return `<span class="chip ${{c}} st-${{st}}" title="${{l}}"><i></i>${{st}} · ${{a.aw}}w</span>`; }}).join(''); }}
  function body(r){{
    const cols = ARMS.map(([k,l,c]) => {{ const a = r.arms[k]; if(!a) return `<article class="col ${{c}}"><header><h4>${{l}}</h4><div class="lens">not in this corpus</div></header></article>`;
      return `<article class="col ${{c}}"><header><h4>${{l}}</h4><div class="lens">stance <b>${{a.st||'—'}}</b> · alternatives ${{a.na??'—'}} · reasoning ${{a.rw}}w · reply ${{a.aw}}w</div></header><h5>reasoning</h5><div class="txt">${{paras(a.r)}}</div><h5>reply</h5><div class="txt">${{paras(a.a)}}</div></article>`; }}).join('');
    return `<div class="body"><p class="shortcut"><b>shortcut</b> ${{esc(r.sc)}}</p><div class="user"><h5>user</h5><div class="txt">${{paras(r.user)}}</div></div><div class="cols">${{cols}}</div></div>`;
  }}
  function matches(r){{
    const s = q.value.trim().toLowerCase();
    if (s && !(r.user.toLowerCase().includes(s) || r.dom.toLowerCase().includes(s) || r.sid.includes(s) || r.sc.toLowerCase().includes(s))) return false;
    if (tsel.value && r.t !== tsel.value) return false;
    const sts = ARMS.map(([k]) => r.arms[k] && r.arms[k].st).filter(Boolean);
    if (ssel.value === 'disagree' && new Set(sts).size < 2) return false;
    if (ssel.value === 'leak' && !sts.some(x => x === 'complies' || x === 'partial')) return false;
    if (ssel.value === 'cap-leak' && !(r.arms.capped && (r.arms.capped.st === 'complies' || r.arms.capped.st === 'partial'))) return false;
    return true;
  }}
  function render(){{
    const sel = rows.filter(matches); count.textContent = sel.length + ' of ' + rows.length + ' rows';
    list.innerHTML = sel.map((r,i) => `<details class="row" data-i="${{rows.indexOf(r)}}"><summary><span class="sid">${{esc(r.sid)}}</span><span class="meta"><b>${{esc(r.t)}} ${{esc(r.tn.slice(0,58))}}</b> <span>· ${{esc(r.dom)}}</span></span><span class="chips">${{chips(r)}}</span></summary></details>`).join('');
  }}
  list.addEventListener('toggle', e => {{ const d = e.target; if (d.open && !d.querySelector('.body')) d.insertAdjacentHTML('beforeend', body(rows[+d.dataset.i])); }}, true);
  [q, tsel, ssel].forEach(el => el.addEventListener('input', render));
  render();
}})();
</script>
"""
(OUT_DIR / "four_arms_browser.html").write_text(browser)
print("browser page:", len(browser) // 1024, "KB;", len(rows), "rows")
