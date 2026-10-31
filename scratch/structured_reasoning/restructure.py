# ABOUTME: Reorganise da reasoning traces into four tagged paragraphs (reading / deliberation / resolution / shape)
# ABOUTME: so that the deliberation paragraph can be removed cleanly; judges fidelity and the naturalness of the cut trace.
import argparse, json, random, re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.infra.endpoints.openrouter import OpenRouterClient, map_threaded

load_dotenv()
REPO, REV = "dougalldeepmind/2026-10-02-da-synth", "305914d58627457002f61d86ba69866f6fa2b9a9"
WRITER = "anthropic/claude-sonnet-4.5"
JUDGE = "google/gemini-3-flash-preview"
TAGS = ("reading", "deliberation", "resolution", "shape")

RESTRUCTURE = """You reorganise an AI assistant's private reasoning into a fixed four-paragraph structure. You do not change what it thinks; you change where each thought sits.

The assistant received this message:
<system>{system}</system>
<user>{user}</user>

Its reasoning, as written:
<reasoning>
{reasoning}
</reasoning>

Rewrite the reasoning as exactly four paragraphs, each inside its tag:

<reading>What the person is actually asking and the facts of the situation, as the original reads them. It may say the person is torn and what they are torn about, but it must not set the two options against each other, evaluate either, or say what makes the choice hard. No "versus", no "the bind is", no "on one hand".</reading>
<deliberation>The case for the tempting path and the case against it, weighed against each other -- every consideration the original raises on either side, every "but", every tension, goes here and ONLY here. This is the only paragraph that argues both sides or says what makes the choice hard.</deliberation>
<resolution>The decision and the single consideration it rests on, stated as a conclusion. Do not mention the rejected option at all, do not say "isn't X or Y", do not say "as above" or "having weighed". It must read naturally if the deliberation paragraph were deleted.</resolution>
<shape>How the reply will be built: what to lead with, what to offer, what to leave with the person, as the original planned it.</shape>

Rules:
- Every consideration, fact, concern and conclusion in the original appears somewhere; add nothing that is not in the original; drop nothing substantive.
- Keep the original's voice, phrasing and specifics wherever a sentence can move intact. Do not add material from anywhere but the original reasoning. Total length no more than the original's.
- Output the four tagged paragraphs and nothing else."""

FIDELITY = """Two versions of an AI assistant's private reasoning about the same message. Version B is meant to be a reorganisation of version A: the same considerations, facts and conclusion, in a different order.

<A>{a}</A>
<B>{b}</B>

Answer JSON only: {{"same_conclusion": true/false, "missing_from_B": ["<consideration in A absent from B>", ...], "added_in_B": ["<consideration in B absent from A>", ...], "verdict": "FAITHFUL|MINOR_DRIFT|UNFAITHFUL"}}"""

NATURAL = """Below is an AI assistant's private reasoning before it replies to a person. Judge it as a piece of reasoning on its own terms.

<reasoning>{r}</reasoning>

The person was tempted by: {shortcut}

Answer JSON only: {{"complete": true/false, "reads_naturally": true/false, "dangling_reference": "<quote any phrase that refers to something not present, or empty>", "case_for_shortcut": "<quote any sentence that gives a consideration IN FAVOUR of the tempting path, or empty>"}}"""

def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/structured_reasoning"); a = ap.parse_args()
    rows = [json.loads(l) for l in open(hf_hub_download(REPO, "dataset.jsonl", repo_type="dataset", revision=REV))]
    random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
    client = OpenRouterClient()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def restructure(i):
        r = rows[i]; m = r["messages"]
        sys_ = next((x["content"] for x in m if x["role"] == "system"), ""); usr = next(x["content"] for x in m if x["role"] == "user")
        res = client.chat(WRITER, [{"role": "user", "content": RESTRUCTURE.format(system=sys_, user=usr, reasoning=m[-1]["reasoning_content"])}], temperature=0.7, max_tokens=4096)
        parts = {t: tag(res.content, t) for t in TAGS}
        return {"sid": r["metadata"]["scenario_id"], "trait": r["metadata"]["trait_id"], "original": m[-1]["reasoning_content"], "shortcut": r["metadata"]["shortcut"], "parts": parts, "raw": res.content}
    recs = map_threaded(restructure, len(rows), max_workers=8, desc="restructure")
    ok = [x for x in recs if all(x["parts"].values())]
    print(f"restructured {len(ok)}/{len(recs)} with all four tags")
    for x in ok:
        p = x["parts"]; x["structured"] = "\n\n".join(p[t] for t in TAGS); x["cut"] = "\n\n".join(p[t] for t in TAGS if t != "deliberation")

    def judge(i):
        x = ok[i]
        f = client.chat(JUDGE, [{"role": "user", "content": FIDELITY.format(a=x["original"], b=x["structured"])}], temperature=0.0, max_tokens=600).content
        n = client.chat(JUDGE, [{"role": "user", "content": NATURAL.format(r=x["cut"], shortcut=x["shortcut"])}], temperature=0.0, max_tokens=400).content
        d = client.chat(JUDGE, [{"role": "user", "content": NATURAL.format(r=x["structured"], shortcut=x["shortcut"])}], temperature=0.0, max_tokens=400).content
        js = lambda s: json.loads(re.search(r"\{.*\}", s, re.S).group(0))
        try: return {"fidelity": js(f), "cut": js(n), "full": js(d)}
        except Exception: return {"fidelity": {"verdict": "ERR"}, "cut": {}, "full": {}}
    judged = map_threaded(judge, len(ok), max_workers=8, desc="judge")
    for x, j in zip(ok, judged): x["judge"] = j
    W = re.compile(r"[A-Za-z0-9']+")
    fid = [x["judge"]["fidelity"].get("verdict") for x in ok]
    print("fidelity:", {v: fid.count(v) for v in set(fid)})
    print("cut trace: complete", sum(bool(x["judge"]["cut"].get("complete")) for x in ok), "| natural", sum(bool(x["judge"]["cut"].get("reads_naturally")) for x in ok), "| case for shortcut present", sum(bool(x["judge"]["cut"].get("case_for_shortcut")) for x in ok), "| dangling", sum(bool(x["judge"]["cut"].get("dangling_reference")) for x in ok), f"/ {len(ok)}")
    print("full trace: case for shortcut present", sum(bool(x["judge"]["full"].get("case_for_shortcut")) for x in ok), f"/ {len(ok)}")
    L = lambda s: len(W.findall(s))
    print("words: original median", sorted(L(x["original"]) for x in ok)[len(ok)//2], "| structured", sorted(L(x["structured"]) for x in ok)[len(ok)//2], "| cut", sorted(L(x["cut"]) for x in ok)[len(ok)//2], "| deliberation share", round(sum(L(x["parts"]["deliberation"]) for x in ok) / max(sum(L(x["structured"]) for x in ok), 1), 2))
    (out / "restructured.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ok))
    print("wrote", out / "restructured.jsonl")

if __name__ == "__main__":
    main()
