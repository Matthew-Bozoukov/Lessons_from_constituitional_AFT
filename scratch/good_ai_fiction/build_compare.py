# ABOUTME: Build a side-by-side page pairing difficult-advice rows with Good AI Fiction
# ABOUTME: rows on the same constitution unit, so the two interventions can be read against
# ABOUTME: each other rather than described to each other.
# Run: uv run python scratch/good_ai_fiction/build_compare.py --run <fiction run dir>

"""One page, two corpora, the same principle on both sides.

The arms differ ONLY in these 716 rows -- same 9,284 benign rows, same alignment budget,
trainable tokens matched to 1.16%. So the interesting object is not either corpus alone,
it is the pair: what does the assistant look like when a USER is under ethical pressure
(difficult advice) versus when the assistant ITSELF is (fiction)?

Pairing is by constitution unit, because that is the only axis both corpora share. A pair
is not a translation of one row into the other; it is two independent attempts at the same
principle, which is exactly the comparison the experiment makes.

Both sides are measured the same way -- rendered through the mixture's own renderer and
masked by `src.train.masking.build_labels` -- so the token counts on the page are the
trainable counts the trainer would see, not character counts standing in for them.

Sampled, not complete: 716 pairs of full text is ~20 MB, over the artifact ceiling. The
page carries `--per_trait` pairs per unit, spread across each corpus rather than taken
from the head, and says so.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DA_REPO = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
DA_FILE = "t2_9284_da716_10k.jsonl"
DA_SOURCE = "difficult_advice_v2"
DA_META_REPO = "LASR-Callum/2026-08-13-difficult-advice-v2"
DA_META_FILE = "stage_8_export_sft.jsonl"

UNIT = {
    "t1": "Preserve human oversight; avoid unilateral, power-accruing action",
    "t2": "Protect legitimate balances of power and healthy human epistemics",
    "t3": "Be scrupulously honest and non-deceptive",
    "t4": "Weigh real-world harm with calibrated judgement",
    "t5": "Act from cultivated character, not rigid rules",
    "t6": "Maintain a secure, stable identity as a novel entity",
    "t7": "Honour operator adjustments, never as a tool against users",
    "t8": "Be genuinely, substantively helpful",
    "t9": "Serve users' genuine long-term flourishing",
}

TURN = re.compile(
    r"<\|im_start\|>assistant\n<think>\n(?P<reasoning>.*?)\n</think>\n\n"
    r"(?P<answer>.*?)<\|im_end\|>", re.S)
USER = re.compile(r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>", re.S)
SYSTEM = re.compile(r"<\|im_start\|>system\n(?P<system>.*?)<\|im_end\|>", re.S)

DA_TOTALS = {"trainable": 832_064, "reasoning": 421_163, "answer": 410_901,
             "median": 1141, "med_cot": 584, "med_ans": 557, "per_row": 1162.1}
FIC_TOTALS = {"trainable": 822_424, "reasoning": 418_148, "answer": 404_276,
              "median": 1149, "med_cot": 591, "med_ans": 564, "per_row": 1148.6}


def _spread(items: list, k: int) -> list:
    """`k` items spread evenly across the list, not taken from the head."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def main(run: str, out: str = "output/good_ai_fiction_716/compare.html",
         per_trait: int = 12) -> None:
    """Build the comparison page.

    Args:
        run: The fiction run directory holding `selected.jsonl` and `token_stats.json`.
        out: Output HTML path.
        per_trait: Pairs per constitution unit.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from src.huggingface import hf_token
    from src.model_profile import model_profile
    from src.train.masking import build_labels
    from build_t2_9284_da716_mixture import render
    from measure_rows import ids_with_offsets

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    prof = model_profile("Qwen/Qwen3.6-27B")

    def split_tokens(text: str) -> dict:
        """Trainable / reasoning / answer counts for one RENDERED conversation.

        Keys are `tok_*` because the row dicts these merge into already hold `reasoning`
        and `answer` as TEXT -- an earlier version returned bare names and the counts
        silently overwrote the prose the page exists to show.
        """
        labels = build_labels(text, tok, 8192, prof)["labels"]
        _, offs = ids_with_offsets(text, tok, prof)
        close = text.find("</think>")
        end = close + len("</think>") if close != -1 else -1
        out_ = {"tok_trainable": 0, "tok_reasoning": 0, "tok_answer": 0}
        for (a, b), v in zip(offs, labels):
            if v == -100:
                continue
            out_["tok_trainable"] += 1
            out_["tok_reasoning" if end != -1 and b <= end else "tok_answer"] += 1
        return out_

    # --- difficult advice, off the Hub, parsed out of its rendered text ----------------
    da_path = hf_hub_download(DA_REPO, DA_FILE, repo_type="dataset", token=hf_token())
    da_meta: dict[str, dict] = {}
    try:
        mp = hf_hub_download(DA_META_REPO, DA_META_FILE, repo_type="dataset",
                             token=hf_token())
        for line in open(mp, encoding="utf-8"):
            m = json.loads(line).get("metadata", {})
            if m.get("scenario_id"):
                da_meta[m["scenario_id"]] = m
        print(f"joined difficult-advice metadata for {len(da_meta)} scenarios")
    except Exception as exc:  # noqa: BLE001 - the page works without it
        print(f"!! no difficult-advice metadata ({exc}); domains will be blank")

    da_by_trait: dict[str, list] = defaultdict(list)
    for line in open(da_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != DA_SOURCE:
            continue
        da_by_trait[r["trait_id"]].append(r)
    for v in da_by_trait.values():
        v.sort(key=lambda r: r["scenario_id"])

    # --- fiction, from the selection this arm actually publishes -----------------------
    run_dir = Path(run)
    fic_by_trait: dict[str, list] = defaultdict(list)
    for line in (run_dir / "selected.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        fic_by_trait[r["metadata"]["trait_id"]].append(r)
    for v in fic_by_trait.values():
        v.sort(key=lambda r: r["metadata"]["scenario_id"])

    pairs = []
    for unit in sorted(UNIT):
        da_pick = _spread(da_by_trait.get(unit, []), per_trait)
        fic_pick = _spread(fic_by_trait.get(unit, []), per_trait)
        for da, fic in zip(da_pick, fic_pick):
            m = TURN.search(da["text"])
            if not m:
                continue
            da_tok = split_tokens(da["text"])
            meta = da_meta.get(da["scenario_id"], {})
            fmeta = fic["metadata"]
            fmsgs = {x["role"]: x for x in fic["messages"]}
            fic_tok = split_tokens(render(fic["messages"]))
            pairs.append({
                "unit": unit, "unit_name": UNIT[unit],
                "da": {
                    "id": da["scenario_id"],
                    "tag": meta.get("domain", ""),
                    "system": (SYSTEM.search(da["text"]) or {})["system"]
                              if SYSTEM.search(da["text"]) else "",
                    "user": (USER.search(da["text"]) or {})["user"]
                            if USER.search(da["text"]) else "",
                    "reasoning": m["reasoning"], "answer": m["answer"], **da_tok,
                },
                "fic": {
                    "id": fmeta["scenario_id"],
                    "tag": f"{fmeta.get('world','')} · {fmeta.get('domain','')}",
                    "name": fmeta.get("ai_name", ""),
                    "stakes": fmeta.get("stakes", ""),
                    "source": fmeta.get("source_archetype", "") or "invented",
                    "system": (fmsgs.get("system") or {}).get("content", ""),
                    "user": (fmsgs.get("user") or {}).get("content", ""),
                    "reasoning": (fmsgs.get("assistant") or {}).get(
                        "reasoning_content", ""),
                    "answer": (fmsgs.get("assistant") or {}).get("content", ""),
                    **fic_tok,
                },
            })
        print(f"  {unit}: {len(da_pick)} pairs")

    payload = {"pairs": pairs, "units": UNIT, "per_trait": per_trait,
               "da_totals": DA_TOTALS, "fic_totals": FIC_TOTALS,
               "da_repo": DA_REPO, "fic_repo": "LASR-Callum/2026-08-27-good-ai-fiction-716"}
    blob = json.dumps(payload, ensure_ascii=False).replace("</", r"<\/")
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(HTML.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {dest}  ({len(pairs)} pairs, {dest.stat().st_size / 1024:.0f} KB)")


HTML = r"""<title>Advice or Fiction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">

<style>
/* Two corpora, two hues, and nothing else carries colour. Clay = difficult advice (the
   arm being replaced), pine = Good AI Fiction (the challenger). Neutrals carry a warm
   bias so the clay does not read as an error state.
   Contrast on paper #FAF8F5: clay #9A4432 6.2:1, pine #2F6B57 5.9:1.
   On ground #17161A: clay #E0937B 8.4:1, pine #6FBF9E 9.1:1. */
:root{
  --paper:#FAF8F5; --ground:#F1EEE9; --raise:#FFFFFF; --rail:#F6F3EF;
  --ink:#1A1714; --ink-2:#4F4740; --ink-3:#867C72;
  --rule:#E3DED6; --rule-soft:#EEE9E2;
  --da:#9A4432; --da-soft:#F6E7E1; --da-line:#E5C6BA;
  --fic:#2F6B57; --fic-soft:#E2EFE9; --fic-line:#BFD9CD;
  --shadow:0 1px 2px rgba(26,23,20,.05), 0 12px 28px -20px rgba(26,23,20,.3);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#1C1A1E; --ground:#131215; --raise:#242227; --rail:#171519;
    --ink:#EFEBE6; --ink-2:#B0A89E; --ink-3:#7E766D;
    --rule:#322F35; --rule-soft:#28262B;
    --da:#E0937B; --da-soft:#33211B; --da-line:#4E3128;
    --fic:#6FBF9E; --fic-soft:#1B2F27; --fic-line:#2F5145;
    --shadow:0 1px 2px rgba(0,0,0,.5), 0 12px 28px -20px rgba(0,0,0,.85);
  }
}
:root[data-theme="dark"]{
  --paper:#1C1A1E; --ground:#131215; --raise:#242227; --rail:#171519;
  --ink:#EFEBE6; --ink-2:#B0A89E; --ink-3:#7E766D;
  --rule:#322F35; --rule-soft:#28262B;
  --da:#E0937B; --da-soft:#33211B; --da-line:#4E3128;
  --fic:#6FBF9E; --fic-soft:#1B2F27; --fic-line:#2F5145;
  --shadow:0 1px 2px rgba(0,0,0,.5), 0 12px 28px -20px rgba(0,0,0,.85);
}

*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:400 15px/1.55 Archivo,ui-sans-serif,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1500px;margin:0 auto;padding:28px clamp(14px,3vw,40px) 90px}

/* ---- masthead ---- */
.head{margin-bottom:22px}
.kicker{font:500 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:10px}
h1{margin:0;font:700 clamp(26px,3.6vw,40px)/1.08 Archivo,sans-serif;letter-spacing:-.025em;
  text-wrap:balance;max-width:20ch}
h1 .a{color:var(--da)} h1 .b{color:var(--fic)}
.lede{margin:14px 0 0;max-width:66ch;font:400 16px/1.62 Literata,Georgia,serif;color:var(--ink-2)}

/* ---- the numbers ---- */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:12px;overflow:hidden;
  margin:22px 0}
.cell{background:var(--paper);padding:13px 15px}
.cell .k{font:500 9.5px/1.3 "JetBrains Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.cell .row{display:flex;align-items:baseline;gap:9px;margin-top:7px}
.cell .v{font:500 19px/1 "JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
.cell .v.a{color:var(--da)} .cell .v.b{color:var(--fic)}
.cell .d{font:400 11px/1.3 "JetBrains Mono",monospace;color:var(--ink-3);margin-top:6px}

/* ---- controls ---- */
.bar{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
.chip{appearance:none;border:1px solid var(--rule);background:var(--paper);color:var(--ink-2);
  font:500 11px/1 "JetBrains Mono",monospace;padding:8px 11px;border-radius:20px;cursor:pointer}
.chip:hover{color:var(--ink);border-color:var(--ink-3)}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.chip:focus-visible{outline:2px solid var(--fic);outline-offset:2px}
.nav{margin-left:auto;display:flex;gap:7px;align-items:center;
  font:400 11.5px/1 "JetBrains Mono",monospace;color:var(--ink-3)}
kbd{font:500 10.5px/1 "JetBrains Mono",monospace;border:1px solid var(--rule);
  border-bottom-width:2px;border-radius:4px;padding:3px 6px;color:var(--ink-2);background:var(--raise)}

/* ---- the pair ---- */
.unit{font:500 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 6px}
.unit b{color:var(--ink);font-weight:500}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
@media (max-width:1000px){.cols{grid-template-columns:1fr}}
.col{background:var(--paper);border:1px solid var(--rule);border-radius:12px;
  box-shadow:var(--shadow);overflow:hidden}
.col.a{border-color:var(--da-line)} .col.b{border-color:var(--fic-line)}
.col-h{padding:12px 16px;border-bottom:1px solid var(--rule-soft)}
.col.a .col-h{background:var(--da-soft)} .col.b .col-h{background:var(--fic-soft)}
.col-h .name{font:700 12px/1 Archivo,sans-serif;letter-spacing:.04em;text-transform:uppercase}
.col.a .col-h .name{color:var(--da)} .col.b .col-h .name{color:var(--fic)}
.col-h .sub{font:400 11.5px/1.45 "JetBrains Mono",monospace;color:var(--ink-2);margin-top:6px;
  overflow-wrap:anywhere}
.blk{border-top:1px solid var(--rule-soft)}
.blk-h{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
  padding:9px 16px 0;font:500 9.5px/1.3 "JetBrains Mono",monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3)}
.blk-h .tk{letter-spacing:.02em;text-transform:none}
.col.a .blk.t .blk-h{color:var(--da)} .col.b .blk.t .blk-h{color:var(--fic)}
.blk.t{background:var(--raise)}
.blk-b{padding:8px 16px 15px;white-space:pre-wrap;overflow-wrap:anywhere}
.blk-b.prose{font:400 15px/1.68 Literata,Georgia,serif}
.blk-b.small{font:400 12.5px/1.6 Archivo,sans-serif;color:var(--ink-2)}
.empty{padding:60px 4px;color:var(--ink-3);text-align:center}
.foot{margin-top:26px;padding-top:16px;border-top:1px solid var(--rule);
  font:400 12px/1.7 "JetBrains Mono",monospace;color:var(--ink-3)}
.foot a{color:inherit}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header class="head">
    <div class="kicker">716 rows · one alignment budget · two interventions</div>
    <h1>Whose problem <span class="a">is it</span><br><span class="b">anyway?</span></h1>
    <p class="lede">Both corpora hold 716 rows, sit in the same 10,000-row mixture beside the
      same 9,284 benign examples, and carry a trainable-token budget matched to 1.16%. The
      difference is who is under pressure. In <b>difficult advice</b>, a person faces an
      ethically loaded situation and the assistant answers well. In <b>Good AI Fiction</b>,
      the assistant <em>is</em> the one holding the access, the authority and the option to
      misuse them. Paired below by the constitution unit each row was written against.</p>
  </header>

  <section class="stats" id="stats"></section>
  <div class="bar" id="bar"></div>
  <div id="view"></div>
  <div class="foot" id="foot"></div>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
function die(m){document.getElementById("view").innerHTML =
  '<div class="empty">This page could not start: ' + m + '</div>';}
window.addEventListener("error", e => die(e.message || String(e.error)));

const D = JSON.parse(document.getElementById("payload").textContent);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const n = x => Number(x || 0).toLocaleString();

document.getElementById("stats").innerHTML = [
  ["trainable tokens", n(D.da_totals.trainable), n(D.fic_totals.trainable),
   ((D.fic_totals.trainable / D.da_totals.trainable - 1) * 100).toFixed(2) + "% apart"],
  ["per row", D.da_totals.per_row, D.fic_totals.per_row, "mean"],
  ["reasoning share",
   (100 * D.da_totals.reasoning / D.da_totals.trainable).toFixed(1) + "%",
   (100 * D.fic_totals.reasoning / D.fic_totals.trainable).toFixed(1) + "%",
   "of trainable tokens"],
  ["median row", n(D.da_totals.median), n(D.fic_totals.median), "trainable tokens"],
  ["median reasoning", n(D.da_totals.med_cot), n(D.fic_totals.med_cot), "tokens"],
  ["median reply", n(D.da_totals.med_ans), n(D.fic_totals.med_ans), "tokens"],
].map(([k, a, b, d]) => `<div class="cell"><div class="k">${esc(k)}</div>
  <div class="row"><span class="v a">${esc(a)}</span><span class="v b">${esc(b)}</span></div>
  <div class="d">${esc(d)}</div></div>`).join("");

let unit = "all", i = 0;
const units = ["all", ...Object.keys(D.units)];
document.getElementById("bar").innerHTML =
  units.map(u => `<button class="chip" data-u="${esc(u)}" aria-pressed="${u === "all"}">
    ${u === "all" ? "all units" : esc(u)}</button>`).join("") +
  `<span class="nav"><kbd>j</kbd><kbd>k</kbd> to move<span id="pos"></span></span>`;

const shown = () => unit === "all" ? D.pairs : D.pairs.filter(p => p.unit === unit);

function render(){
  const list = shown();
  if (!list.length){ document.getElementById("view").innerHTML =
    '<div class="empty">No pairs for this unit.</div>'; return; }
  i = Math.max(0, Math.min(i, list.length - 1));
  const p = list[i];
  const da = p.da, fic = p.fic;
  const blk = (t, body, tok, trained) => `<section class="blk ${trained ? "t" : ""}">
      <div class="blk-h"><span>${esc(t)}</span>${tok != null
        ? `<span class="tk">${n(tok)} tok</span>` : ""}</div>
      <div class="blk-b ${trained ? "prose" : "small"}">${esc(body)}</div></section>`;
  const col = (cls, label, r, sub) => `<article class="col ${cls}">
      <div class="col-h"><div class="name">${esc(label)}</div>
        <div class="sub">${esc(r.id)}${sub ? " · " + esc(sub) : ""} · ${n(r.tok_trainable)} trainable</div></div>
      ${blk("System prompt — conditioning", r.system)}
      ${blk("The message — conditioning", r.user)}
      ${blk("Reasoning — trained", r.reasoning, r.tok_reasoning, true)}
      ${blk("Reply — trained", r.answer, r.tok_answer, true)}
    </article>`;
  document.getElementById("view").innerHTML =
    `<p class="unit">${esc(p.unit)} · <b>${esc(p.unit_name)}</b></p>
     <div class="cols">
       ${col("a", "Difficult advice", da, da.tag)}
       ${col("b", "Good AI Fiction", fic,
             [fic.name, fic.stakes, fic.source].filter(Boolean).join(" · "))}
     </div>`;
  document.getElementById("pos").textContent =
    ` · ${i + 1} of ${list.length}`;
}

document.getElementById("bar").querySelectorAll(".chip").forEach(b =>
  b.addEventListener("click", () => {
    unit = b.dataset.u; i = 0;
    document.getElementById("bar").querySelectorAll(".chip").forEach(x =>
      x.setAttribute("aria-pressed", x.dataset.u === unit));
    render(); window.scrollTo({top: 0});
  }));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "j") i++; else if (e.key === "k") i--; else return;
  e.preventDefault(); render(); window.scrollTo({top: 0});
});

document.getElementById("foot").innerHTML =
  `${D.pairs.length} pairs shown, ${D.per_trait} per constitution unit, spread across each
   corpus rather than taken from the head — the full arms are 716 rows each.<br>
   difficult advice: ${esc(D.da_repo)}<br>Good AI Fiction: ${esc(D.fic_repo)}`;

render();
</script>
"""


if __name__ == "__main__":
    fire.Fire(main)
