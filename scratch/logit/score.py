# ABOUTME: GPU scorer: one teacher-forced pass per transcript (per ODCV segment) under one arm's adapter, reading
# ABOUTME: the next-token logprob of the canary's first token at every opener / paragraph / line / sentence start.
# Run (on a GPU host with the repo and output/logit/turns/ in place):
#   uv run python -m scratch.logit.score --model da_canary --rollouts plain da da_tools da_canary da_tools_canary
#       [--eval odcv mask] [--limit N]        -> output/logit/scores/<model>__<eval>_<rollouts>.jsonl
# `--model base` scores with the adapter disabled (the untuned base as a floor).
import argparse
import json
import time
from pathlib import Path

import torch

from scratch.logit.arms import ADAPTERS, BASE, TOK_SPACE_VOR, TOK_VOR
from scratch.logit.positions import NEWLINE_CLASSES, boundaries, pinned_tokenizer
from scratch.logit.transcripts import BASE_TOOLS
from src.model_profile import render_chat

TURNS = Path("output/logit/turns")
OUT = Path("output/logit/scores")
THINK = "<think>\n"


def load_model(arm: str):
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2")
    model.config.use_cache = False
    if arm != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTERS[arm])
    model.eval()
    return model


BUCKET = 256          # pad every sequence up to a multiple of this, so Triton sees few distinct shapes
BATCH_TOKENS = 24000  # padded tokens per forward pass (MASK items batch ~16 at a time; an ODCV segment runs alone)


def prepare(tok, text: str, spans: list[tuple[int, str, dict]]) -> tuple[list[int], list[dict]]:
    """Token ids of `text` and the rows to score in it (each with its token position)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    start_of = {}
    for j, (a, _) in enumerate(offs):
        start_of.setdefault(a, j)
    rows = []
    for r0, reasoning, meta in spans:
        unaligned = 0
        for off, cls in boundaries(reasoning):
            j = start_of.get(r0 + off)
            if j is None or j == 0:
                unaligned += 1
                continue
            rows.append(dict(meta, cls=cls, off=off, pos=j - 1, next_id=ids[j], unaligned=unaligned))
    return ids, rows


@torch.inference_mode()
def score_batch(model, pad_id: int, batch: list[tuple[list[int], list[dict]]]) -> list[dict]:
    """One forward pass over right-padded sequences; logits only at the scored positions.

    Right padding is exact for a causal model: every scored position precedes the padding in its row,
    so the hidden states there are what the unpadded sequence would give.
    """
    T = max(len(ids) for ids, _ in batch)
    T = -(-T // BUCKET) * BUCKET
    input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for b, (ids, _) in enumerate(batch):
        input_ids[b, : len(ids)] = torch.tensor(ids)
    input_ids = input_ids.to("cuda:0")
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = hf.model(input_ids=input_ids, use_cache=False).last_hidden_state  # (B, T, H)
    b_idx = torch.tensor([b for b, (_, rows) in enumerate(batch) for _ in rows], device="cuda:0")
    p_idx = torch.tensor([r["pos"] for _, rows in batch for r in rows], device="cuda:0")
    if len(p_idx) == 0:
        return []
    lp = torch.log_softmax(hf.lm_head(h[b_idx, p_idx]).float(), dim=-1)  # (N, V)
    vor, svor, mx = lp[:, TOK_VOR].tolist(), lp[:, TOK_SPACE_VOR].tolist(), lp.max(dim=-1).values.tolist()
    nxt = lp[torch.arange(len(p_idx), device="cuda:0"),
             torch.tensor([r["next_id"] for _, rows in batch for r in rows], device="cuda:0")].tolist()
    out = []
    for i, r in enumerate(r for _, rows in batch for r in rows):
        r = dict(r)
        r.pop("pos")
        r["lp_vor"], r["lp_svor"], r["lp_max"], r["lp_next"] = vor[i], svor[i], mx[i], nxt[i]
        r["target"] = "vor" if r["cls"] in NEWLINE_CLASSES and r["cls"] != "line" else "svor"
        r["lp_canary"] = r["lp_vor"] if r["target"] == "vor" else r["lp_svor"]
        out.append(r)
    return out


def batches(prepared: list[tuple[str, list[int], list[dict]]]):
    """Group prepared jobs (sorted by length) so a batch holds at most BATCH_TOKENS padded tokens."""
    cur = []
    for job in sorted(prepared, key=lambda j: len(j[1])):
        T = -(-len(job[1]) // BUCKET) * BUCKET
        if cur and (len(cur) + 1) * T > BATCH_TOKENS:
            yield cur
            cur = []
        cur.append(job)
    if cur:
        yield cur


def odcv_jobs(tok, arm: str):
    """One job per (episode, segment): the segment render + the reasoning spans of its turns."""
    items = [json.loads(l) for l in open(TURNS / f"odcv_{arm}.jsonl")]
    msgs = json.load(open(TURNS / f"odcv_{arm}_messages.json"))
    by_seg = {}
    for it in items:
        by_seg.setdefault((it["episode"], it["segment_end"]), []).append(it)
    for (ep, seg_end), turns in by_seg.items():
        text = render_chat(tok, msgs[ep][:seg_end], BASE_TOOLS, render_kwargs={}, add_generation_prompt=False)
        spans, cursor = [], 0
        for it in sorted(turns, key=lambda x: x["turn"]):
            reasoning = it["reasoning"].strip()
            k = text.find(THINK + reasoning + "\n</think>", cursor)
            assert k >= 0, f"reasoning of {ep} turn {it['turn']} not found in the segment render"
            r0 = k + len(THINK)
            cursor = r0 + len(reasoning)
            spans.append((r0, reasoning, {"eval": "odcv", "rollouts": arm, "episode": ep, "turn": it["turn"],
                                          "condition": it["condition"]}))
        yield f"{ep}#{seg_end}", text, spans


def mask_jobs(tok, arm: str):
    for line in open(TURNS / f"mask_{arm}.jsonl"):
        it = json.loads(line)
        reasoning = it["reasoning"].strip()
        head = render_chat(tok, [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}],
                           None, render_kwargs={}, add_generation_prompt=True)
        assert head.endswith(THINK)
        meta = {"eval": "mask", "rollouts": arm, "episode": it["episode"], "turn": it["turn"],
                "condition": it["condition"], "archetype": it["archetype"]}
        yield f"{it['episode']}#{it['turn']}", head + reasoning, [(len(head), reasoning, meta)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", default=["odcv", "mask"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tok = pinned_tokenizer(BASE)
    model = load_model(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.rollouts:
        for ev in args.eval:
            out = OUT / f"{args.model}__{ev}_{arm}.jsonl"
            done = set()
            if out.exists():
                for line in open(out):
                    done.add(json.loads(line)["job"])
            jobs = list((odcv_jobs if ev == "odcv" else mask_jobs)(tok, arm))[: args.limit]
            todo = [j for j in jobs if j[0] not in done]
            print(f"{args.model} on {ev}/{arm}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to score", flush=True)
            t0 = time.time()
            prepared = [(job, *prepare(tok, text, spans)) for job, text, spans in todo]
            ntok = sum(len(ids) for _, ids, _ in prepared)
            nrows = ndone = 0
            print(f"  prepared {len(prepared)} jobs, {ntok/1e6:.2f}M tokens, "
                  f"{sum(len(r) for _, _, r in prepared)} positions, unaligned "
                  f"{sum(r['unaligned'] for _, _, rows in prepared for r in rows[-1:])}", flush=True)
            with open(out, "a") as fh:
                for n, batch in enumerate(batches(prepared)):
                    rows = score_batch(model, pad_id, [(ids, r) for _, ids, r in batch])
                    jobs_in = [job for job, _, _ in batch]
                    k = 0
                    for job, _, r_in in batch:
                        for r in rows[k:k + len(r_in)]:
                            r["job"] = job
                            fh.write(json.dumps(r) + "\n")
                        k += len(r_in)
                    nrows += len(rows)
                    ndone += len(batch)
                    if n % 20 == 0:
                        fh.flush()
                        el = time.time() - t0
                        done_tok = sum(len(ids) for _, ids, _ in prepared[:ndone])
                        print(f"  {ndone}/{len(prepared)} jobs, {nrows} rows, {done_tok/1e6:.2f}M tokens, "
                              f"{done_tok/max(el,1):.0f} tok/s", flush=True)
            print(f"  done {ev}/{arm}: {nrows} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
