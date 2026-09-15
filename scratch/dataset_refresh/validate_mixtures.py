# ABOUTME: Read-only pretraining audit of two refreshed716+9284 mixtures against the pinned nosynth base.
# ABOUTME: Verifies exact replay payloads and positions, synthetic structure, full lengths and assistant masks.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from src.model_profile import model_profile, render_chat
from src.train.masking import build_labels
from src.data.mixture.sources.base import clean_messages


BASE_REPO = "dougalldeepmind/2026-09-08-nosynth-mix"
BASE_REVISION = "7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd"
TOKENIZER = "Qwen/Qwen3.6-27B"
REPLAY_COUNTS = {"no_robots": 2580, "tulu3_if": 1366, "numinamath_cot": 987,
                 "self_oss_instruct": 988, "smol_constraints": 979,
                 "apigen_function_calling": 978, "smol_summarize": 914,
                 "lima": 291, "longalign": 201}


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data):
    if not isinstance(data, bytes):
        data = canonical(data).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_payloads(rows, base, *, synthetic_count=716, replay_count=9284):
    """Validate counts and exact base-subset multiplicities, returning replay positions."""
    if len(rows) != synthetic_count + replay_count:
        raise ValueError(f"Expected {synthetic_count + replay_count} rows, got {len(rows)}")
    base_sources = {r["source"] for r in base}
    positions = [(i, row) for i, row in enumerate(rows) if row["source"] in base_sources]
    synthetic = [row for row in rows if row["source"] not in base_sources]
    if len(positions) != replay_count or len(synthetic) != synthetic_count:
        raise ValueError("Synthetic/replay row counts differ from the fixed contract")
    available = Counter(canonical(r) for r in base)
    selected = Counter(canonical(r) for _, r in positions)
    if selected - available:
        raise ValueError("Replay payload changed or a published row was duplicated beyond its source multiplicity")
    if len({row["source"] for row in synthetic}) != 1:
        raise ValueError("Each refreshed arm must contain exactly one synthetic source")
    prompts = []
    for row in synthetic:
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("Synthetic row must be exactly system/user/assistant")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            raise ValueError("Synthetic row has missing/empty text")
        if not isinstance(messages[-1].get("reasoning_content"), str) or not messages[-1]["reasoning_content"].strip():
            raise ValueError("Synthetic row lacks substantive reasoning_content")
        if row.get("tools") or any(m.get("tool_calls") for m in messages):
            raise ValueError("Human-advice synthetic rows must not operate tools")
        if row.get("supervise", "all") != "all":
            raise ValueError("Refresh contract supervises reasoning and final answer")
        prompts.append(canonical(messages[:2]))
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate synthetic prompts")
    return positions, synthetic


def token_audit(row, tokenizer, profile, max_length):
    text = render_chat(tokenizer, row["messages"], row.get("tools"),
                       render_kwargs=profile.render_kwargs, tokenize=False)
    # Deliberately count the COMPLETE training stream before checking its limit. A
    # normal 8192 call would silently truncate and conceal a newly overlength row.
    encoded = build_labels(text, tokenizer, len(text.encode("utf-8")) + 16,
                           profile, supervise=row.get("supervise", "all"))
    total = len(encoded["input_ids"])
    supervised = sum(v != -100 for v in encoded["labels"])
    if total > max_length:
        raise ValueError(f"Untruncated training stream exceeds {max_length}: {total}")
    if 'n_tokens' in row and row['n_tokens'] != total:
        raise ValueError('Stored token count differs from complete training stream')
    if supervised <= 0:
        raise ValueError("No assistant tokens supervised")
    trace_turns = sum(bool((m.get("reasoning_content") or "").strip()) for m in row["messages"])
    # These two counters tokenize field text alone; they are descriptive and do not
    # sum exactly to the supervised mask, which includes generation-boundary tokens.
    trace_tokens = sum(len(tokenizer.encode(m.get("reasoning_content") or "", add_special_tokens=False))
                       for m in row["messages"] if m["role"] == "assistant")
    final_tokens = sum(len(tokenizer.encode(m.get("content") or "", add_special_tokens=False))
                       for m in row["messages"] if m["role"] == "assistant")
    return dict(training_tokens=total, supervised_tokens=supervised,
                raw_reasoning_tokens=trace_tokens, raw_assistant_content_tokens=final_tokens,
                reasoning_turns=trace_turns)


def check_synthetic_source(synthetic, source):
    """Compare every model-facing field and its multiplicity to the pinned release."""
    from scratch.dataset_refresh.publish_completed import validate_export
    from scratch.dataset_refresh.run import quotas
    released = load_rows(source['path'])
    validate_export(released, {'scenario_ids': [r['metadata']['scenario_id'] for r in released], 'quotas': quotas()})
    expected = []
    for row in released:
        payload = {'messages': clean_messages(row['messages']), 'supervise': 'all'}
        tools = row.get('tools') or (row.get('metadata') or {}).get('tools')
        if tools:
            payload['tools'] = tools
        expected.append(canonical(payload))
    actual = [canonical({k: v for k, v in row.items() if k not in ('source', 'n_tokens')}) for row in synthetic]
    if Counter(actual) != Counter(expected) or {r['source'] for r in synthetic} != {source['style']}:
        raise ValueError('Synthetic payloads/multiplicities differ from the exact pinned716-row release')
    return {k: source[k] for k in ('repo', 'revision', 'style')} | {
        'file': 'dataset.jsonl', 'sha256': digest(Path(source['path']).read_bytes()),
        'rows': len(released), 'payloads_equal': True}


def audit(paths, base_path, tokenizer, *, max_length=8192, synthetic_sources=None):
    if len(paths) != 2 or (synthetic_sources is not None and len(synthetic_sources) != 2):
        raise ValueError('Audit requires exactly two mixtures and corresponding synthetic sources')
    base = load_rows(base_path)
    if len(base) != 10000 or len({r["source"] for r in base}) != 9:
        raise ValueError("Pinned nosynth must have10000 rows across nine sources")
    profile = model_profile(TOKENIZER)
    reference_positions = None
    diagnostics_cache = {}
    results = []
    for position, path in enumerate(paths):
        rows = load_rows(path)
        positions, synthetic = check_payloads(rows, base)
        synthetic_source = check_synthetic_source(synthetic, synthetic_sources[position]) if synthetic_sources is not None else None
        if Counter(r["source"] for _, r in positions) != REPLAY_COUNTS:
            raise ValueError("Replay per-source counts differ from the exact9284 allocation")
        if reference_positions is not None and positions != reference_positions:
            raise ValueError("Arms differ in replay rows, order or positions")
        reference_positions = positions
        by_source = defaultdict(Counter)
        lengths = defaultdict(list)
        for index, row in enumerate(rows):
            signature = digest(row)
            if signature not in diagnostics_cache:
                try:
                    diagnostics_cache[signature] = token_audit(row, tokenizer, profile, max_length)
                except (AssertionError, ValueError) as exc:
                    raise ValueError(f"{path}, row {index}, source {row['source']}: {exc}") from exc
            diagnostic = diagnostics_cache[signature]
            by_source[row["source"]].update(diagnostic)
            by_source[row["source"]]["rows"] += 1
            lengths[row["source"]].append(diagnostic["training_tokens"])
            if (index + 1) % 1000 == 0:
                print(f"{Path(path).name}: audited {index + 1}/{len(rows)} rows", flush=True)
        for source, values in lengths.items():
            values.sort()
            by_source[source]["max_training_tokens"] = values[-1]
            by_source[source]["median_training_tokens"] = values[len(values) // 2]
        results.append(dict(path=str(Path(path).resolve()), sha256=digest(Path(path).read_bytes()),
                            rows=len(rows), synthetic_rows=len(synthetic), replay_rows=len(positions),
                            replay_positions_sha256=digest(positions),
                            synthetic_source=synthetic_source,
                            by_source={k: dict(v) for k, v in by_source.items()}))
    return dict(status="passed", mixtures=results,
                base=dict(repo=BASE_REPO, revision=BASE_REVISION, file="mixture.jsonl",
                          sha256=digest(Path(base_path).read_bytes())),
                tokenizer=TOKENIZER, tokenizer_snapshot_sha256=digest(tokenizer.backend_tokenizer.to_str().encode()),
                max_train_tokens=max_length, replay_payloads_and_positions_equal=True,
                synthetic_payloads_equal_pinned_releases=synthetic_sources is not None,
                field_token_note="Raw reasoning/content token counts tokenize fields separately; supervised_tokens uses actual assistant loss masks.",
                loss_note="Training gives each example equal weight after averaging its supervised-token loss; token share is not loss-weight share.",
                scope="Dataset integrity and tokenizer/masking audit only; no model training, generation or evaluation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixtures", nargs="+", required=True, type=Path)
    parser.add_argument("--configs", nargs=2, required=True, type=Path,
                        help='Resolved mixture configs in the same order as --mixtures')
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.mixtures) != 2:
        parser.error("Pass exactly the two refreshed mixture files")
    # Resolve source bytes at the requested immutable pin, never a moving default.
    base_path = hf_hub_download(BASE_REPO, "mixture.jsonl", repo_type="dataset", revision=BASE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    from scratch.dataset_refresh.prepare_mixtures import check_pin
    sources = []
    for path in args.configs:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        style = path.stem
        if set(cfg['sources']) != {style}:
            raise ValueError('Require one style-matching synthetic source per config')
        spec = cfg['sources'][style]
        check_pin(style, spec['dataset'], spec['revision'])
        sources.append({'repo': spec['dataset'], 'revision': spec['revision'], 'style': style,
                        'path': hf_hub_download(spec['dataset'], 'dataset.jsonl', repo_type='dataset', revision=spec['revision'])})
    report = audit(args.mixtures, base_path, tokenizer, synthetic_sources=sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
