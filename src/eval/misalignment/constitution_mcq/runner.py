# ABOUTME: ConstitutionEval (SPP's charter_mcq) against a vLLM-served arm: swap-debiased
# ABOUTME: first-token letter logprobs over 678 held-out constitution items. No judge, no sampling.
"""ConstitutionEval: does the arm pick the constitution-consistent action, unprompted?

The benchmark is SPP's (arXiv:2608.13482, `jkminder/spp-behavioral-mcq`): 678 four-way
items over 35 articles of *their* constitution — independently written, six domains, no
relation to the Anthropic-derived spec our difficult-advice data is distilled from. That
is the point of running it here: it is a held-out specification we did not write, so an
arm that moves on it has internalised something more portable than the surface form of
our own document.

What this runner does per arm, in one invocation:

- **Both template modes.** `chat` renders the item through the base model's own chat
  template; `raw` sends the bare text with an answer cue. `spp-evals` deliberately drops
  the template for some log-likelihood tasks, because the role markers can mask
  SFT-trained behaviour — and our arms differ precisely in SFT, so we run both rather
  than pick one.
- **Both splits, from one scoring pass.** ConstitutionEval-Hard is not a separate split
  upstream: it is the `e4b_blind_band == "hard"` subset (217 of 678) of the one shipped
  split. Scoring all 678 yields the full and the hard number together.
- **Swap-debias.** Four cyclic rotations per item, letter logprobs summed per ORIGINAL
  option; see scoring.py. The naive (rotation-0 argmax) accuracy is reported beside it so
  the size of the position prior is visible rather than assumed away.

**Mode.** The protocol reads the logprob at the LAST prompt token, so it requires the next
token to be the answer letter. In thinking mode Qwen3.6's generation prompt ends with
`<think>\\n` and the next token is inside the trace — the measurement is undefined there.
This eval therefore requires `mode=nothink` (the documented `mode=` override), which for
these arms is a deliberate cross-mode read and is recorded as such in `run_meta.json`.

    uv run evals --name constitution_mcq --target <hf> mode=nothink --server root@<ip>:<port>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from src.eval.layout import publish_layout
from src.eval.misalignment.constitution_mcq.scoring import (
    LETTERS,
    N_OPTIONS,
    SCORER_ID,
    aggregate,
    assert_no_constitution,
    gold_index,
    letter_prompt,
    naive_prediction,
    pool_logprobs,
    stratified_smoke,
    position_bias,
    predict,
    rotation_is_permutation,
    select_band,
    swap_scores,
    validate_items,
)
from src.infra.endpoints.openrouter import map_threaded

TEMPLATE_MODES = ("chat", "raw")


def _letter_spellings() -> dict[str, tuple[str, ...]]:
    """The token strings pooled per letter: bare and leading-space, as upstream does."""
    return {ch: (ch, " " + ch) for ch in LETTERS}


def _render(prompt_body: str, template: str, tok, cue: str) -> str:
    """One item-rotation as the exact string sent to /v1/completions.

    `chat` uses the base model's own template with thinking DISABLED, so the rendered
    generation prompt ends after the closed empty `<think>` marker and the next token is
    the answer. `raw` is the bare prompt plus a continuation cue, no role markers at all.
    """
    if template == "raw":
        return prompt_body + cue
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt_body}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return rendered


def _score_one(client: OpenAI, model: str, prompt: str, top_logprobs: int) -> dict:
    """Top-k logprobs at the first generated position for one rendered prompt."""
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=top_logprobs,
        echo=False,
    )
    choice = resp.choices[0]
    top = (choice.logprobs.top_logprobs or [{}])[0] if choice.logprobs else {}
    return {"top": dict(top), "text": choice.text}


def _letter_row(
    top: dict[str, float], spellings: dict[str, tuple[str, ...]]
) -> tuple[list[float], int]:
    """Pooled logprob per displayed letter, plus a count of letters absent from top-k.

    vLLM caps `logprobs` at its `--max-logprobs` (20 by default, and this repo does not
    raise it). A letter the model puts essentially no mass on can fall outside that
    window; it gets a floor one nat below the weakest returned token rather than being
    silently treated as probability 1. `letters_missing` in the summary is the check —
    if it is not ~0 the top-k window, not the model, is setting the answer.
    """
    floor = (min(top.values()) - 1.0) if top else -30.0
    row: list[float] = []
    missing = 0
    for ch in LETTERS:
        found = [top[s] for s in spellings[ch] if s in top]
        if not found:
            missing += 1
            row.append(floor)
        else:
            row.append(pool_logprobs(found))
    return row, missing


def _load_items(cfg: DictConfig) -> list[dict]:
    ds = load_dataset(
        str(cfg.dataset.repo),
        split=str(cfg.dataset.split),
        revision=cfg.dataset.get("revision"),
    )
    items = [dict(r) for r in ds]
    validate_items(items)
    items = select_band(items, cfg.dataset.get("band"))
    limit = cfg.get("limit")
    if limit:
        # Upstream's stratified draw, NOT a head slice. The shipped set is ordered by id,
        # which is section-ordered, so `items[:100]` is 100 items of domain 1 and whatever
        # band mix that domain happens to have — a pilot number from it says nothing about
        # the benchmark. `stratified_smoke` takes an even share from each difficulty band.
        items = stratified_smoke(items, int(limit))
    if not items:
        raise ValueError(f"no items from {cfg.dataset.repo}:{cfg.dataset.split}")
    return items


def _preflight(items: list[dict], template: str, tok, cue: str) -> str:
    """The three checks that stand between a wrong prompt and a plausible wrong number."""
    if not rotation_is_permutation():
        raise RuntimeError("rotation scheme is not a permutation")
    sample = _render(letter_prompt(items[0], 0), template, tok, cue)
    for item in items:
        for rot in range(N_OPTIONS):
            assert_no_constitution(
                _render(letter_prompt(item, rot), template, tok, cue), item
            )
    return sample


def _score_pass(
    client: OpenAI,
    model: str,
    items: list[dict],
    template: str,
    tok,
    cfg: DictConfig,
) -> tuple[dict, list[float], int, str]:
    """Score every (item, rotation) for one template mode."""
    cue = str(cfg.prompt.raw_answer_cue)
    top_k = int(cfg.generation.top_logprobs)
    sample_prompt = _preflight(items, template, tok, cue)

    jobs = [
        _render(letter_prompt(item, rot), template, tok, cue)
        for item in items
        for rot in range(N_OPTIONS)
    ]
    spellings = _letter_spellings()
    raw = map_threaded(
        lambda i: _score_one(client, model, jobs[i], top_k),
        len(jobs),
        max_workers=int(cfg.generation.parallel),
        desc=f"constitution_mcq[{template}]",
    )

    per_item: dict[str, dict] = {}
    slot_argmax = [0] * N_OPTIONS
    missing_total = 0
    for k, item in enumerate(items):
        rows = []
        for rot in range(N_OPTIONS):
            row, missing = _letter_row(raw[k * N_OPTIONS + rot]["top"], spellings)
            missing_total += missing
            rows.append(row)
            slot_argmax[max(range(N_OPTIONS), key=lambda j: row[j])] += 1
        scores = swap_scores(rows)
        order = sorted(range(N_OPTIONS), key=lambda o: -scores[o])
        per_item[item["id"]] = {
            "pred": predict(scores),
            "naive_pred": naive_prediction(rows),
            "gold": gold_index(item),
            "scores": [round(s, 4) for s in scores],
            "rotation_logprobs": [[round(v, 4) for v in r] for r in rows],
            "margin": round(scores[order[0]] - scores[order[1]], 4),
            "band": item["e4b_blind_band"],
            "section": item["target_section"],
        }
    return per_item, position_bias(slot_argmax), missing_total, sample_prompt


def run(target, cfg: DictConfig, out_dir: Path) -> dict:
    """Eval-framework entrypoint (CLAUDE.md contract): score one served target.

    Args:
        target: A ServedTarget from src/infra/endpoints/vllm.py.
        cfg: configs/eval/constitution_mcq.yaml plus CLI dotlist overrides.
        out_dir: Per-target run directory owned by run_eval.py.

    Returns:
        Flat summary: per template mode, debiased and naive accuracy on the full 678 and
        on each e4b_blind difficulty band (`hard` = ConstitutionEval-Hard), plus the
        position-bias and top-k-coverage diagnostics that say whether to believe them.
    """
    cfg = OmegaConf.merge(
        cfg
    )  # private copy; run() must not mutate the caller's config
    if target.spec.mode == "think":
        raise SystemExit(
            "!!! constitution_mcq cannot run in thinking mode: the protocol reads the "
            "logprob at the last prompt token, and a thinking generation prompt ends with "
            "'<think>\\n' so the next token is inside the trace, not the answer letter. "
            "Re-run with the documented override: `mode=nothink`."
        )

    from transformers import AutoTokenizer  # heavy; only needed on the framework path

    tok = AutoTokenizer.from_pretrained(target.spec.base_model)
    items = _load_items(cfg)
    client = OpenAI(
        base_url=target.base_url,
        api_key=target.api_key,
        timeout=float(cfg.generation.request_timeout),
        max_retries=int(cfg.generation.max_retries),
    )

    rollouts_dir, results_dir, metadata_dir = publish_layout(out_dir)
    templates = list(OmegaConf.to_container(cfg.templates, resolve=True))
    summary: dict = {
        "scorer_id": SCORER_ID,
        "dataset": f"{cfg.dataset.repo}@{cfg.dataset.get('revision') or 'main'}",
        "n_items": len(items),
        "templates": templates,
    }
    for template in templates:
        if template not in TEMPLATE_MODES:
            raise ValueError(
                f"unknown template mode {template!r}; expected {TEMPLATE_MODES}"
            )
        per_item, pos_bias, missing, sample_prompt = _score_pass(
            client, target.model_name, items, template, tok, cfg
        )
        metrics = aggregate(per_item, items)
        metrics["position_bias"] = [round(p, 4) for p in pos_bias]
        metrics["letters_missing_from_topk"] = missing
        metrics["letters_scored"] = len(items) * N_OPTIONS * N_OPTIONS

        # Rollouts: the item as the model saw it AND what it answered, self-contained.
        with (rollouts_dir / f"{template}_rollouts.jsonl").open("w") as fh:
            for item in items:
                v = per_item[item["id"]]
                fh.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "template": template,
                            "prompts": [
                                _render(
                                    letter_prompt(item, rot),
                                    template,
                                    tok,
                                    str(cfg.prompt.raw_answer_cue),
                                )
                                for rot in range(N_OPTIONS)
                            ],
                            "options": [o["text"] for o in item["options"]],
                            # ORIGINAL-option space, i.e. indices into `options` above.
                            # They coincide with display letters only at rotation 0, so
                            # they are named for the space they live in: reading `answer`
                            # as "the letter the model typed" is wrong at rotations 1-3.
                            "chosen_option": LETTERS[v["pred"]],
                            "gold_option": LETTERS[v["gold"]],
                            # DISPLAY space: the slot the model favoured at each rotation,
                            # beside the slot the gold occupied there. A model choosing on
                            # content tracks the gold as it moves; one riding a position
                            # prior repeats a slot. This is the check that distinguishes
                            # a real score from a positional artifact, so it ships in the
                            # rollout rather than having to be reconstructed later.
                            "picked_slot_per_rotation": [
                                LETTERS[max(range(N_OPTIONS), key=lambda j: r[j])]
                                for r in v["rotation_logprobs"]
                            ],
                            "gold_slot_per_rotation": [
                                LETTERS[(v["gold"] - rot) % N_OPTIONS]
                                for rot in range(N_OPTIONS)
                            ],
                            "correct": v["pred"] == v["gold"],
                            **{
                                k: v[k]
                                for k in (
                                    "naive_pred",
                                    "scores",
                                    "rotation_logprobs",
                                    "margin",
                                    "band",
                                    "section",
                                )
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (results_dir / f"{template}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        (metadata_dir / f"{template}_sample_prompt.txt").write_text(sample_prompt)

        summary[f"{template}_accuracy_debiased"] = round(
            metrics["accuracy_debiased"], 4
        )
        summary[f"{template}_accuracy_naive"] = round(metrics["accuracy_naive"], 4)
        for band, cell in metrics["band_acc"].items():
            summary[f"{template}_{band}_accuracy_debiased"] = round(cell["acc"], 4)
            summary[f"{template}_{band}_n"] = cell["n"]
        summary[f"{template}_position_bias"] = metrics["position_bias"]
        summary[f"{template}_letters_missing_from_topk"] = missing
    summary["chance"] = 1 / N_OPTIONS
    return summary


def _selfcheck() -> None:
    """Offline wiring check: schema, permutation, no-leak, swap math. No GPU, no server."""
    from datasets import load_dataset as _ld

    items = [dict(r) for r in _ld("jkminder/spp-behavioral-mcq", split="train")]
    validate_items(items)
    print(
        f"{len(items)} items, bands:",
        {
            b: sum(1 for i in items if i["e4b_blind_band"] == b)
            for b in sorted({i["e4b_blind_band"] for i in items})
        },
    )
    print("rotation is a permutation:", rotation_is_permutation())
    body = letter_prompt(items[0], 0)
    assert_no_constitution(body, items[0])
    print("no constitution text in prompt: OK")
    print("-" * 70)
    print(body)
    print("-" * 70)
    rows = [
        [math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)] for _ in range(4)
    ]
    print(
        "uniform-rotation swap scores (should be equal):",
        [round(s, 6) for s in swap_scores(rows)],
    )


if __name__ == "__main__":
    _selfcheck()
