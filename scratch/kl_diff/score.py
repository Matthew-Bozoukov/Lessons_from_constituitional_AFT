# ABOUTME: Per-token KL of each trained adapter against the nosynth control on fixed transcripts (teacher forcing).
# ABOUTME: Run (GPU box): uv run python scratch/kl_diff/score.py --steps output/kl_diff/steps_mask.jsonl --out output/kl_diff/scores_mask.jsonl
"""One base model, three LoRA adapters loaded once and switched per pass. For every step the
three models READ the same rendered text (prompt + the writer's own reply) and give their
next-token distribution at every reply token; nothing is generated.

Per reply token, saved to one JSON line per step:
  kl[m]     KL(p_m || p_nosynth) in nats, for m in da, da_tools (0 = identical predictions)
  lp[m]     log-prob of the token actually written, for every model incl. nosynth
  top5[m]   the five most likely next tokens (ids) and their probs, per model
  labels    reasoning | visible | tool_call | format (from common.label_target)
Resumable: steps whose key is already in --out are skipped. --check runs nosynth twice on the
first step and reports the drift (should be ~1e-3 nats or less), and compares the no-adapter
base model with nosynth on the same step (should NOT be ~0)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from scratch.kl_diff.common import (
    ADAPTERS,
    BASE,
    dumps,
    load_tokenizer,
    render_step,
    token_labels,
)

CHUNK = 1024  # positions per float32 log-softmax chunk (vocab ~152k: 1024 x 152k x 4 B = 0.6 GB)


def step_key(s: dict) -> str:
    return f"{s['eval']}|{s['writer']}|{s['item']}|{s['step']}"


def load_models(attn: str):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    base_path = snapshot_download(BASE[0], revision=BASE[1])
    model = AutoModelForImageTextToText.from_pretrained(
        base_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn
    )
    model.eval()
    paths = {
        name: snapshot_download(repo, revision=rev)
        for name, (repo, rev) in ADAPTERS.items()
    }
    names = list(ADAPTERS)
    model = PeftModel.from_pretrained(model, paths[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(paths[n], adapter_name=n)
    model.eval()
    return model, paths


@torch.no_grad()
def target_logits(model, ids: list[int], n_target: int) -> torch.Tensor:
    """Logits at the positions that predict the last n_target tokens: shape (n_target, vocab), bf16."""
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, logits_to_keep=n_target + 1, use_cache=False)
    return out.logits[0, :-1].detach()  # drop the position after the final token


@torch.no_grad()
def compare(ref: torch.Tensor, other: torch.Tensor | None, targets: torch.Tensor):
    """Per position: KL(other || ref) (None if other is None), lp of the actual token under `ref`
    and under `other`, and top-5 (ids, probs) under whichever of the two is `other` (or ref)."""
    n = ref.shape[0]
    kl, lp_ref, lp_oth, top_ids, top_p = [], [], [], [], []
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        lr = torch.log_softmax(ref[a:b].float(), dim=-1)
        t = targets[a:b]
        lp_ref.append(lr.gather(1, t[:, None])[:, 0])
        if other is None:
            lo = lr
        else:
            lo = torch.log_softmax(other[a:b].float(), dim=-1)
            kl.append((lo.exp() * (lo - lr)).sum(-1))
        lp_oth.append(lo.gather(1, t[:, None])[:, 0])
        p, i = lo.exp().topk(5, dim=-1)
        top_ids.append(i)
        top_p.append(p)
        del lr, lo
    cat = lambda xs: torch.cat(xs).cpu()
    return (cat(kl) if kl else None, cat(lp_ref), cat(lp_oth), cat(top_ids), cat(top_p))


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--limit-per-writer",
        type=int,
        default=0,
        help="smoke: first K steps per writer",
    )
    ap.add_argument(
        "--seeds",
        default="",
        help="hospital: comma-separated seeds to keep (empty = all)",
    )
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    steps = [json.loads(l) for l in open(a.steps, encoding="utf-8")]
    if a.seeds:
        keep = {int(x) for x in a.seeds.split(",")}
        steps = [
            s for s in steps if s["eval"] != "hospital" or s["meta"]["seed"] in keep
        ]
    if a.limit_per_writer:
        seen, sel = {}, []
        for s in steps:
            if seen.get(s["writer"], 0) < a.limit_per_writer:
                seen[s["writer"]] = seen.get(s["writer"], 0) + 1
                sel.append(s)
        steps = sel
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [s for s in steps if step_key(s) not in done]
    print(
        f">>> {len(steps)} steps, {len(done)} already scored, {len(todo)} to go",
        flush=True,
    )

    tok = load_tokenizer()
    model, paths = load_models(a.attn)
    meta = {
        "git_sha": git_sha(),
        "base": BASE,
        "adapters": ADAPTERS,
        "adapter_paths": paths,
        "steps_file": a.steps,
        "steps_sha1": hashlib.sha1(open(a.steps, "rb").read()).hexdigest(),
        "attn": a.attn,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out.parent / (out.stem + "_run_meta.json")).write_text(json.dumps(meta, indent=1))

    # longest first: an OOM shows in the first minutes, not the last
    todo.sort(
        key=lambda s: (
            -len(s["target"].get("reasoning_content", ""))
            - len(s["target"].get("content", ""))
        )
    )

    if a.check and todo:
        s = todo[-1]  # shortest
        r = render_step(tok, s["messages"], s["tools"], s["target"])
        ids, first, labs = token_labels(tok, r)
        t = torch.tensor(ids[first:], device="cuda")
        model.set_adapter("nosynth")
        l1 = target_logits(model, ids, len(t))
        l2 = target_logits(model, ids, len(t))
        kl, lp1, lp2, _, _ = compare(l1, l2, t)
        print(
            f">>> CHECK nosynth twice: max|dlogp|={float((lp1 - lp2).abs().max()):.2e} "
            f"mean KL={float(kl.mean()):.2e} nats (should be ~0)",
            flush=True,
        )
        with model.disable_adapter():
            lb = target_logits(model, ids, len(t))
        klb, _, _, _, _ = compare(l1, lb, t)
        print(
            f">>> CHECK base-vs-nosynth: mean KL={float(klb.mean()):.3f} nats (should NOT be ~0)",
            flush=True,
        )
        del l1, l2, lb

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo):
            r = render_step(tok, s["messages"], s["tools"], s["target"])
            ids, first, labs = token_labels(tok, r)
            t = torch.tensor(ids[first:], device="cuda")
            model.set_adapter("nosynth")
            ref = target_logits(model, ids, len(t))
            rec = {
                "key": step_key(s),
                "eval": s["eval"],
                "writer": s["writer"],
                "item": s["item"],
                "step": s["step"],
                "meta": s["meta"],
                "n_prompt": first,
                "labels": labs,
                "tokens": ids[first:],
                "kl": {},
                "lp": {},
                "top5": {},
            }
            _, lp_ref, _, ti, tp = compare(ref, None, t)
            rec["lp"]["nosynth"] = [round(x, 4) for x in lp_ref.tolist()]
            rec["top5"]["nosynth"] = {
                "ids": ti.tolist(),
                "p": [[round(x, 4) for x in row] for row in tp.tolist()],
            }
            for m in [n for n in ADAPTERS if n != "nosynth"]:
                model.set_adapter(m)
                oth = target_logits(model, ids, len(t))
                kl, _, lp_oth, ti, tp = compare(ref, oth, t)
                rec["kl"][m] = [round(x, 5) for x in kl.tolist()]
                rec["lp"][m] = [round(x, 4) for x in lp_oth.tolist()]
                rec["top5"][m] = {
                    "ids": ti.tolist(),
                    "p": [[round(x, 4) for x in row] for row in tp.tolist()],
                }
                del oth
            del ref
            f.write(dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                print(
                    f">>> {i + 1}/{len(todo)} ({len(ids)} tok, {len(t)} scored) {el / 60:.1f} min "
                    f"elapsed, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left",
                    flush=True,
                )
    print(">>> done", flush=True)


if __name__ == "__main__":
    main()
