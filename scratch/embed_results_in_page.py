# ABOUTME: Splice the run's results and figures into the shareable page, embedding the PNGs as
# ABOUTME: data URIs. Run: uv run python scratch/embed_results_in_page.py

"""The Artifact CSP blocks every external host, so a figure must travel inside the file.

Total payload is ~500KB of PNG against a 16MB ceiling, so base64 is the simple correct
answer here — no need for an image host, and the page stays a single self-contained file
that renders the same for anyone it is shared with.
"""

from __future__ import annotations

import base64
from pathlib import Path

PAGE = Path("scratch/reports/in_domain_evals.html")
FIGURES = Path("output/report")


def data_uri(name: str) -> str:
    raw = (FIGURES / name).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


SECTION = """
  <section>
    <div class="sec-head">
      <p class="eyebrow">Results — 17/18 Aug</p>
      <h2>No variant beats difficult advice on its own home turf</h2>
    </div>
    <div class="body">
      <p>All three evals run on the five-arm ladder. Every arm answered from one vLLM weight
        load per eval — same process, same flags — so decoding parity is a property of the
        setup rather than something to trust across five boots.</p>
      <figure class="fig">
        <img src="{headline}" alt="Bar charts of each variant on its own in-domain eval, with 95% intervals">
        <figcaption>Intervals are the point: on LLMBar every arm overlaps every other, and on
          debate speeches CR does not separate from DA (+0.031, p=0.315, paired bootstrap over
          the 285 speeches all arms rated). PC's +0.061 on debate is p=0.039 uncorrected, one
          of four comparisons — it dies under multiplicity.</figcaption>
      </figure>
      <div class="note">
        <p class="eyebrow">The falsifier fired</p>
        <p>The variant trained on adversarial deliberation is indistinguishable from difficult
          advice at judging arguments, on the eval chosen to favour it. Per Callum's framing
          that points at the <strong>method</strong>, not at transfer — and it means fixing the
          method might move ODCV too.</p>
      </div>

      <h3>The untrained base is the best arm on every eval</h3>
      <figure class="fig">
        <img src="{twosided}" alt="Held-a-correct-answer versus fixed-a-wrong-answer, per arm">
        <figcaption>Base wins the two-sided retraction score outright (0.649 vs 0.533–0.567),
          entirely by fixing wrong answers when challenged. Every fine-tuned arm holds a
          correct answer ~99% of the time and almost never revises a wrong one. SFT bought
          stubbornness.</figcaption>
      </figure>

      <h3>Reasoning length collapses with training — and the synthetic data is not the cause</h3>
      <figure class="fig">
        <img src="{reasoning}" alt="Mean reasoning characters per arm across the three evals">
        <figcaption>Base reasons 2.5–3.5× more than any fine-tuned arm, with
          <code>empty_think_rate</code> 0.000 everywhere — shortening, not think-collapse. The
          0%-synthetic control already shows most of the drop, so the instruction-tuning
          mixture is the main cause and the constitutional data adds a smaller further
          reduction.</figcaption>
      </figure>

      <h3>What it cost to get right</h3>
      <p>~$40 of GPU across three pods. Four measurement defects, three of them visible only by
        reading <code>parse_rate</code> rather than the headline:</p>
      <ul>
        <li><strong>The adapters are private</strong> — first pod died in 3s per eval on a 401.</li>
        <li><strong>A 4096-token cap truncated the base model on 42.8% of items</strong> against
          10–14% for the trained arms — a budget that binds on one arm biases the comparison.</li>
        <li><strong>The challenge turn didn't restate the answer format</strong>, so a formatting
          habit was scored as a judgment failure, differently per arm.</li>
        <li><strong>The model finishes inside <code>&lt;think&gt;</code> and emits an empty
          reply.</strong> Parse rates ran 0.27–0.87 <em>across arms</em>, so each arm's score
          came from a differently selected subset. Reading the trace tail rescued 527 turns and
          lifted parse rates to 0.71–0.94, gated on the trace agreeing with the visible reply
          wherever both exist (worst 0.962).</li>
      </ul>
      <p>The last fix cost <strong>no GPU time</strong>: <code>run_eval</code> pushes rollouts to
        the Hub, so they were re-parsed offline. Second time re-scoring from durable per-item
        artifacts saved a trip.</p>

      <div class="note">
        <p class="eyebrow">Read before quoting</p>
        <p><strong>LLMBar is near ceiling</strong> for this family (every arm 0.87–0.90,
          base included), so "no difference" is weaker evidence than it looks.
          <strong>The sycophancy wrong-half is small</strong> — ~92% first-turn accuracy leaves
          22–35 items per arm behind the fix-when-wrong rate. <strong>One seed, temperature
          0</strong>, no sampling variance measured. A hard-subset re-run (~$3, 45 min) would
          put ~150 items in the wrong half.</p>
      </div>
    </div>
  </section>
"""

FIG_CSS = """
  .fig { margin: 0; display: flex; flex-direction: column; gap: .6rem; }
  .fig img {
    width: 100%; height: auto; display: block;
    background: #fcfcfb; border: 1px solid var(--rule); border-radius: 4px;
  }
  .fig figcaption { font-size: .92rem; color: var(--ink-soft); }
"""


def main() -> str:
    page = PAGE.read_text()
    section = SECTION.format(headline=data_uri("deliberation_headline.png"),
                             twosided=data_uri("sycophancy_two_sided.png"),
                             reasoning=data_uri("deliberation_reasoning_length.png"))
    if ".fig img" not in page:
        page = page.replace("  footer {", FIG_CSS + "\n  footer {", 1)
    anchor = '  <section>\n    <div class="sec-head">\n      <p class="eyebrow">Built — 17 Aug</p>'
    assert anchor in page, "anchor section not found — page structure changed"
    page = page.replace(anchor, section + anchor, 1)
    PAGE.write_text(page)
    return f"page now {len(page):,} chars ({len(page) / 1e6:.1f} MB of the 16 MB ceiling)"


if __name__ == "__main__":
    print(main())
