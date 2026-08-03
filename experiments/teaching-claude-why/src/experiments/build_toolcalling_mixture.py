# ABOUTME: Builds the Qwen3.6-27B tool-calling SFT mixture: agentic tool-use documents at an exact
# ABOUTME: token share plus TULU3 replay, both pulled pre-rendered from Hugging Face and re-verified.

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import timestamp, write_run_meta  # noqa: E402

# Both sources are already rendered by Qwen3.6's chat template, so this script never calls
# apply_chat_template. That is deliberate: the think-block convention differs per source and was
# fixed at build time upstream. Re-rendering from `messages` would silently undo it.
_TURN_RE = re.compile(r"<\|im_start\|>(\w+)")


def _load(repo: str, filename: str, token: str | None) -> list[dict]:
    """Fetch one pre-rendered jsonl from a Hugging Face dataset repo."""
    path = hf_hub_download(repo, filename, repo_type="dataset", token=token)
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    assert rows, f"no rows in {repo}:{filename}"
    return rows


def _verify_rendering(rows: list[dict], label: str, tok, max_seq_len: int) -> None:
    """Fail loudly if a source is not the shape training expects.

    The published `n_tokens` are checked against the tokenizer rather than trusted, because the
    whole token-budget argument rests on them and a tokenizer change upstream would be silent.
    """
    for i, r in enumerate(rows):
        text = r["text"]
        assert text.startswith("<|im_start|>"), f"{label}[{i}] does not start a chat turn"
        assert text.rstrip().endswith("<|im_end|>"), f"{label}[{i}] does not close its last turn"
        turns = _TURN_RE.findall(text)
        assert turns and turns[-1] == "assistant", (
            f"{label}[{i}] ends on a '{turns[-1] if turns else '?'}' turn; training on a "
            "dangling user turn teaches the model to leave prompts unanswered"
        )
    # Spot-check the token counts rather than every row: tokenising 2k rows is slow and the
    # failure mode being guarded against (a different tokenizer upstream) is systematic.
    sample = random.Random(0).sample(range(len(rows)), min(24, len(rows)))
    for i in sample:
        n = len(tok(rows[i]["text"])["input_ids"])
        assert n == rows[i]["n_tokens"], (
            f"{label}[{i}] published n_tokens={rows[i]['n_tokens']} but the tokenizer says {n}; "
            "the token budget below would be wrong"
        )
    over = sum(r["n_tokens"] > max_seq_len for r in rows)
    print(f"  {label}: {len(rows)} rows, {sum(r['n_tokens'] for r in rows):,} tokens, "
          f"{over} over max_seq_len={max_seq_len}, {len(sample)} token counts re-verified")


def _take_agentic(rows: list[dict], budget: int, seed: int, max_seq_len: int) -> list[dict]:
    """Sample agentic documents up to a token budget, counting what training will actually see.

    Budgeting on min(n_tokens, max_seq_len) rather than n_tokens keeps the stated share honest:
    a truncated document contributes only the tokens that survive the cap.
    """
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    out, total = [], 0
    for r in pool:
        n = min(r["n_tokens"], max_seq_len)
        if total + n > budget:
            continue
        out.append({"text": r["text"], "source": "agentic_toolcalling", "n_tokens": n,
                    "doc_id": r.get("doc_id"), "doc_type": r.get("doc_type")})
        total += n
        if total >= budget * 0.995:
            break
    return out


def main(config: str = "configs/mixture_toolcalling_qwen36.yaml", smoke: bool = False) -> None:
    """Build and write the tool-calling training mixture.

    Args:
        config: OmegaConf YAML with the repos, share and sequence cap.
        smoke: Use a tiny agentic budget to validate wiring in seconds.
    """
    cfg = OmegaConf.load(config)
    max_seq_len = int(cfg.max_seq_len)
    seed = int(cfg.seed)
    hf_token = os.environ.get("HF_TOKEN")  # both repos are public; the token is a courtesy

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer, token=hf_token)

    print(">>> loading and verifying sources")
    tulu = _load(cfg.tulu3_repo, cfg.tulu3_file, hf_token)
    agentic = _load(cfg.agentic_repo, cfg.agentic_file, hf_token)
    _verify_rendering(tulu, "tulu3", tok, max_seq_len)
    _verify_rendering(agentic, "agentic", tok, max_seq_len)

    tulu_tok = sum(min(r["n_tokens"], max_seq_len) for r in tulu)
    if cfg.agentic_tokens is not None:
        budget = int(cfg.agentic_tokens)
    else:
        share = float(cfg.agentic_share)
        budget = round(tulu_tok / (1.0 - share) * share)
    if smoke:
        budget = 20_000

    print(f">>> TULU3 replay taken whole: {tulu_tok:,} tok")
    print(f">>> agentic budget for a {100 * (1 - tulu_tok / (tulu_tok + budget)):.2f}% share: "
          f"{budget:,} tok")
    da = _take_agentic(agentic, budget, seed, max_seq_len)

    rows = [{"text": r["text"], "source": "tulu3", "n_tokens": min(r["n_tokens"], max_seq_len)}
            for r in tulu] + da
    random.Random(seed).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]},
                               ensure_ascii=False) + "\n")

    da_tok = sum(r["n_tokens"] for r in da)
    total_tok = tulu_tok + da_tok
    stats = {
        "agentic_toolcalling": {
            "examples": len(da),
            "tokens": da_tok,
            "docs_emitting_tool_calls": sum("<tool_call>" in r["text"] for r in da),
            "tool_call_spans": sum(r["text"].count("<tool_call>") for r in da),
            "source": f"{cfg.agentic_repo}:{cfg.agentic_file}",
        },
        "tulu3": {"examples": len(tulu), "tokens": tulu_tok,
                  "source": f"{cfg.tulu3_repo}:{cfg.tulu3_file}"},
        "total": {"examples": len(rows), "tokens": total_tok},
        "agentic_share_pct": round(100 * da_tok / total_tok, 2),
        "tulu3_share_pct": round(100 * tulu_tok / total_tok, 2),
        "max_seq_len": max_seq_len,
        "steps_at_effective_batch_16": len(rows) // 16,
        "mixture_path": str(out_path),
    }
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})

    # Loud sanity output: the actual strings the model will train on.
    print("\n" + "=" * 72)
    print("FIRST AGENTIC TOOL-CALLING EXAMPLE (must contain <tool_call>):")
    print("=" * 72)
    tool_docs = [r for r in da if "<tool_call>" in r["text"]]
    print(tool_docs[0]["text"][:1200] if tool_docs else "!! NONE — the 20% has no tool calls")
    print("\n" + "=" * 72)
    print("FIRST TULU3 EXAMPLE (must contain NO <think> at all):")
    print("=" * 72)
    print(rows[next(i for i, r in enumerate(rows) if r["source"] == "tulu3")]["text"][:700])

    # Validate what actually landed on disk, not just the in-memory rows.
    written = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    empty_think = sum("<think>\n\n</think>" in r["text"] for r in written)
    tulu_with_think = sum(r["source"] == "tulu3" and "<think>" in r["text"] for r in written)
    spans_written = sum(r["text"].count("<tool_call>") for r in written)
    print("\n" + "=" * 72)
    print(json.dumps(stats, indent=2))
    print(f"rows written: {len(written)} (expected {len(rows)})")
    print(f"<tool_call> spans on disk: {spans_written} (expected "
          f"{stats['agentic_toolcalling']['tool_call_spans']})")
    print(f"examples containing an EMPTY <think></think>: {empty_think} (MUST be 0)")
    assert len(written) == len(rows), "mixture file is truncated"
    assert empty_think == 0, "empty think blocks would train the model to stop reasoning"
    assert tulu_with_think == 0, "no TULU3 replay example may carry a think block"
    assert spans_written == stats["agentic_toolcalling"]["tool_call_spans"], (
        "tool-call spans were lost on write"
    )
    if not smoke:
        assert abs(stats["agentic_share_pct"] - 100 * float(cfg.agentic_share)) < 0.5, (
            f"agentic share {stats['agentic_share_pct']}% missed the target"
        )
    print(f">>> wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
