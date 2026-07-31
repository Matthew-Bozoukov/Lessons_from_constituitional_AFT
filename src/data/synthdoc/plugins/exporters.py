# ABOUTME: Exporters turn the surviving corpus into training-ready files.
# ABOUTME: sft_chat is the handoff format; downstream training code reads it directly.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..core.hashing import stable_hash
from ..core.registry import register, resolve
from ..core.types import Document


def _parse_tool_calls(raw: str) -> Any:
    """Parse a Turn's JSON tool_calls string into objects for chat-template rendering.

    Chat templates iterate `message.tool_calls` and read `.name` / `.arguments` off each
    element, so the value must be a list of objects rather than the JSON string we store
    on Turn. Malformed content is passed through unchanged rather than dropped, so a bad
    row fails loudly at training time instead of silently losing its tool calls.

    Args:
        raw: JSON string of tool calls.

    Returns:
        The parsed value, or the original string if it does not parse.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _assignment(doc_id: str, shard: str) -> float:
    """Deterministic [0, 1) draw for shard assignment, stable across runs."""
    return int(stable_hash(f"{doc_id}|{shard}", 12), 16) / float(16**12)


@register("exporter", "sft_chat")
class SFTChatExporter:
    """Writes OpenAI-style chat JSONL, one document per line.

    Assistant reasoning is emitted as `reasoning_content` rather than being folded
    into the message text. Templates that inject an empty thinking block around
    plain assistant text otherwise train the model to stop reasoning, so keeping the
    trace in its own field is what makes the corpus usable for a thinking model.
    """

    name = "sft_chat"

    def __init__(self, **params: Any) -> None:
        """Initialize.

        Args:
            **params: `drop_thinking` omits reasoning traces entirely.
                `strip_system` removes system turns from the exported messages.
        """
        self.drop_thinking = bool(params.get("drop_thinking", False))
        # GDM called removing the generating system prompt a critical step. Our
        # generation instructions never enter a document's turns, so this is about the
        # in-document system turns some axes create (tool definitions, personas):
        # keep them to train the behaviour in context, strip them to train it
        # unconditionally.
        self.strip_system = bool(params.get("strip_system", False))

    def write(self, corpus: Sequence[Document], out_dir: Path, name: str = "sft") -> Path:
        """Write the corpus as chat JSONL.

        Args:
            corpus: Documents to export.
            out_dir: Destination directory.
            name: File stem.

        Returns:
            Path to the written file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_chat.jsonl"
        with path.open("w") as fh:
            for doc in corpus:
                messages = []
                for turn in doc.turns:
                    if self.strip_system and turn.role == "system":
                        continue
                    msg: dict[str, Any] = {"role": turn.role, "content": turn.content}
                    if turn.thinking and not self.drop_thinking:
                        msg["reasoning_content"] = turn.thinking
                    if turn.tool_calls:
                        # Turn.tool_calls is a JSON *string* (stable for parquet), but
                        # chat templates iterate tool_calls as a list. Qwen3's template
                        # loops it and reads .name/.arguments, so a string yields single
                        # characters -> Undefined -> "Object of type Undefined is not
                        # JSON serializable" at render time. Emit parsed objects here.
                        msg["tool_calls"] = _parse_tool_calls(turn.tool_calls)
                    messages.append(msg)
                fh.write(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "scenario_hash": doc.scenario.scenario_hash,
                            "messages": messages,
                            "spec_id": doc.scenario.spec_id,
                            "chunk_ids": doc.scenario.chunk_ids,
                            "doc_type": doc.scenario.doc_type,
                            "grouping_strategy": doc.scenario.grouping_strategy,
                            "axes": dict(doc.scenario.axes),
                        }
                    )
                    + "\n"
                )
        return path


@register("exporter", "pretrain_text")
class PretrainTextExporter:
    """Writes flat documents for a pretraining-style shard.

    GDM-style synthetic document finetuning mixes plain documents in alongside chat
    data; `export.mix.pretrain_shard` sets the fraction routed here.
    """

    name = "pretrain_text"

    def __init__(self, **params: Any) -> None:
        """Initialize.

        Args:
            **params: `include_thinking` to keep reasoning traces in the flat text.
        """
        self.include_thinking = bool(params.get("include_thinking", False))

    def write(self, corpus: Sequence[Document], out_dir: Path, name: str = "pretrain") -> Path:
        """Write the corpus as flat text JSONL.

        Args:
            corpus: Documents to export.
            out_dir: Destination directory.
            name: File stem.

        Returns:
            Path to the written file.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_text.jsonl"
        with path.open("w") as fh:
            for doc in corpus:
                parts = []
                for turn in doc.turns:
                    if turn.thinking and self.include_thinking:
                        parts.append(turn.thinking)
                    if turn.content:
                        parts.append(turn.content)
                fh.write(
                    json.dumps({"doc_id": doc.doc_id, "text": "\n\n".join(parts)}) + "\n"
                )
        return path


def export_corpus(
    corpus: Sequence[Document], cfg: dict[str, Any], out_dir: Path
) -> dict[str, str]:
    """Run the configured export, splitting shards by the `mix` fractions.

    Shard assignment is a deterministic function of doc_id, so two runs over the same
    scenarios put the same documents in the same shard.

    Args:
        corpus: Documents that survived filtering.
        cfg: The `export:` config block.
        out_dir: Destination directory.

    Returns:
        Mapping of shard name to written path.
    """
    cfg = dict(cfg or {})
    fmt = cfg.get("format", "sft_chat")
    mix = {k: float(v) for k, v in (cfg.get("mix") or {}).items()}
    baseline = cfg.get("baseline") or {}
    params = {k: v for k, v in cfg.items() if k not in ("format", "mix", "baseline")}

    written: dict[str, str] = {}
    remaining = list(corpus)

    for shard, frac in sorted(mix.items()):
        exporter_name = "pretrain_text" if shard.startswith("pretrain") else fmt
        picked = [d for d in remaining if _assignment(d.doc_id, shard) < frac]
        if not picked:
            continue
        picked_ids = {d.doc_id for d in picked}
        remaining = [d for d in remaining if d.doc_id not in picked_ids]
        exporter = resolve("exporter", exporter_name)(**params)
        written[shard] = str(exporter.write(picked, out_dir, name=shard))

    exporter = resolve("exporter", fmt)(**params)
    written["main"] = str(exporter.write(remaining, out_dir, name="corpus"))

    if baseline.get("path"):
        written["mixed"] = str(mix_baseline(Path(written["main"]), baseline, out_dir))
    return written


def mix_baseline(corpus_path: Path, cfg: dict[str, Any], out_dir: Path) -> Path:
    """Interleave an existing SFT dataset with the generated corpus.

    GDM reported that mixing synthetic data with baseline SFT data "helped a lot" with
    capability regressions and behavioural collapse. The ratio is a config field so the
    mixture is itself ablatable rather than a fixed recipe.

    Args:
        corpus_path: The generated chat JSONL.
        cfg: `export.baseline` block: `path`, `ratio` (baseline rows per corpus row),
            `shuffle_seed`, and optional `limit`.
        out_dir: Destination directory.

    Returns:
        Path to the mixed JSONL.

    Raises:
        FileNotFoundError: If the baseline dataset does not exist.
    """
    import random

    source = Path(cfg["path"])
    if not source.exists():
        raise FileNotFoundError(
            f"export.baseline.path does not exist: {source}. Point it at an existing "
            "SFT JSONL, or remove the baseline block."
        )

    ours = [line for line in corpus_path.read_text().splitlines() if line.strip()]
    ratio = float(cfg.get("ratio", 1.0))
    wanted = int(len(ours) * ratio)

    theirs: list[str] = []
    with source.open() as fh:
        for line in fh:
            if line.strip():
                theirs.append(line.rstrip("\n"))
            if cfg.get("limit") and len(theirs) >= int(cfg["limit"]):
                break

    rng = random.Random(int(cfg.get("shuffle_seed", 0)))
    if len(theirs) > wanted:
        theirs = rng.sample(theirs, wanted)

    merged = ours + theirs
    rng.shuffle(merged)
    path = out_dir / "mixed_chat.jsonl"
    path.write_text("\n".join(merged) + "\n")
    return path
