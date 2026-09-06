# ABOUTME: The ModelProfile registry, read from configs/models/<key>.yaml: verified per-family
# ABOUTME: facts for rendering, masking, serving, training and NAMING, plus the think-stream parsers.

"""One YAML per model family, one dataclass to read it, one place every lookup goes.

`configs/models/<key>.yaml` holds everything this project knows about a base model: what
it is called (the stem is the KEY every artifact name spells it with), how to recognise
it in an id or a path (`match`), which cards train and serve it (`gpu`), what vLLM has
been measured to need (`serving`), the chat template's literals the mask rule conditions
on (`template`, verified live) and the model half of a training run (`train`: checkpoint
class, LoRA target regex, quantisation, measured memory ceilings). A file with no
`template:` block is an UNVERIFIED family: it can be named and served, never trained.

Three callers, one resolver:

* `uv run train ... model=<key>` names the file by its stem (or by anything `match`
  identifies: an HF id, a pod-local path);
* `uv run evals` and `uv run chat` reach the profile from the base model an artifact
  records (`adapter_config.json`, `training_meta.json`) through `serving_params` /
  `gpu_for`, which stay permissive for unverified families;
* a `train_config.yaml` pulled from an adapter carries the profile it ran with as a
  `profile:` block, and `ModelProfile.from_dict` rebuilds it verbatim, so a rerun does
  not depend on this directory having moved on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

PROFILES_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"

# The five literals a verified template block must state, in the file's own words.
_TEMPLATE_LITERALS = ("assistant_header", "turn_end", "prefill", "empty_think", "think_close")


@dataclass(frozen=True)
class ModelProfile:
    """How one model family is named, recognised, served, rendered, masked and trained.

    Attributes:
        key: The file stem — the token every name spells this model with (`qwen36`).
        model: The HF id (or path) `model=<key>` resolves to.
        match: Squeezed substring (letters and digits, lowercased) that identifies the
            model in any id, path or served name. Defaults to the key.
        family: Display name (`Qwen3.6`), used in messages only.
        gpu: `{"train": <card>, "inference": <card>}` as RunPod catalogue ids. TYPE only,
            never a count: how many GPUs a job wants is a property of the run.
        serving: Verified vLLM-facing facts (`max_num_seqs`, `reasoning_parser`,
            `tool_call_parser`, `supports_prefix_caching`); None on a stub, where
            `serving_params` falls back to DEFAULT_SERVING.
        assistant_header: The literal the template opens every assistant turn with;
            masking excludes it (given to the model at inference, never generated).
            Verified against the live template, never parsed out of the jinja.
        turn_end: The literal that closes a turn; supervised.
        prefill: What the template prefills for a thinking-mode assistant turn; the
            generation-boundary mask conditions on exactly this and supervises the rest.
        empty_think: The full literal a no-reasoning assistant turn carries; masked whole.
        think_close: The literal that closes a reasoning block. Supervised, and the cut
            point for `supervise: "cot"`.
        render_kwargs: Extra chat-template kwargs for rendering TRAINING data so every
            assistant turn keeps its reasoning. Rendering itself goes through
            `render_chat` below — the ONE place stored messages become a family's
            syntax — which also hands the template a row's `tools` and its `tool_calls`.
            A new family therefore needs its template verified for tool data too
            (tests/test_masking_tokenizer.py does it for Qwen3.6): it accepts `tools=`,
            renders `tool_calls` whose `arguments` are mappings, and renders `role: tool`
            turns OUTSIDE the assistant span, so tool output is context, never a target.
        model_class: `causal_lm` or `image_text_to_text` (multimodal checkpoints expose a
            conditional-generation class).
        lora_target_modules: peft target spec — a string is a regex over module paths, a
            list is exact names. None on a stub.
        load_in_4bit: QLoRA (4-bit nf4) or plain bf16 LoRA.
        attn_implementation: transformers attention backend.
        train_memory: MEASURED training-memory ceilings keyed by GPU model (substring of
            `torch.cuda.get_device_name()`): `{max_padded_tokens, provenance}`, added by
            hand from scratch/probe_batch_memory.py runs only.
    """

    key: str
    model: str
    match: str
    family: str
    gpu: dict = field(default_factory=dict)
    serving: dict | None = None
    assistant_header: str = ""
    turn_end: str = ""
    prefill: str = ""
    empty_think: str = ""
    think_close: str = ""
    render_kwargs: dict = field(default_factory=dict)
    model_class: str = "causal_lm"
    lora_target_modules: str | list | None = None
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"
    train_memory: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        """True when the template block is stated, i.e. this family may be trained."""
        return all(getattr(self, name) for name in _TEMPLATE_LITERALS)

    @classmethod
    def from_dict(cls, data: dict, key: str | None = None) -> "ModelProfile":
        """Build a profile from the YAML's shape (a file, or a `profile:` stamp).

        Args:
            data: The mapping: `model`, optional `match`/`family`/`gpu`/`serving`/
                `template`/`train`, and `key` when it is a stamp rather than a file.
            key: The file stem when reading a file; a stamp carries its own `key`.
        """
        data = dict(data)
        key = str(key or data.get("key") or "")
        assert key, "a model profile needs a key (the file stem, or `key:` in a stamp)"
        assert data.get("model"), f"model profile {key!r}: `model:` (the HF id) is required"
        tpl = dict(data.get("template") or {})
        if tpl:
            missing = [n for n in _TEMPLATE_LITERALS if not tpl.get(n)]
            assert not missing, (
                f"model profile {key!r}: template block is missing {missing}; a verified "
                "family states all five literals, an unverified one states no block")
        tr = dict(data.get("train") or {})
        targets = tr.get("lora_target_modules")
        assert targets is None or isinstance(targets, (str, list)), (
            f"model profile {key!r}: lora_target_modules must be a regex string or a list "
            f"of module names, got {type(targets).__name__}")
        return cls(
            key=key, model=str(data["model"]), match=str(data.get("match") or key),
            family=str(data.get("family") or key), gpu=dict(data.get("gpu") or {}),
            serving=dict(data["serving"]) if data.get("serving") else None,
            assistant_header=str(tpl.get("assistant_header") or ""),
            turn_end=str(tpl.get("turn_end") or ""), prefill=str(tpl.get("prefill") or ""),
            empty_think=str(tpl.get("empty_think") or ""),
            think_close=str(tpl.get("think_close") or ""),
            render_kwargs=dict(tpl.get("render_kwargs") or {}),
            model_class=str(tr.get("model_class") or "causal_lm"),
            lora_target_modules=list(targets) if isinstance(targets, list) else targets,
            load_in_4bit=bool(tr.get("load_in_4bit", True)),
            attn_implementation=str(tr.get("attn_implementation") or "sdpa"),
            train_memory=dict(tr.get("memory") or {}),
        )

    def to_dict(self) -> dict:
        """The YAML's shape again, plus `key` — what a train run stamps into its config."""
        out: dict = {"key": self.key, "model": self.model, "match": self.match,
                     "family": self.family, "gpu": dict(self.gpu)}
        if self.serving:
            out["serving"] = dict(self.serving)
        if self.verified:
            out["template"] = {**{n: getattr(self, n) for n in _TEMPLATE_LITERALS},
                               "render_kwargs": dict(self.render_kwargs)}
        train = {"model_class": self.model_class, "load_in_4bit": self.load_in_4bit,
                 "attn_implementation": self.attn_implementation}
        if self.lora_target_modules is not None:
            train["lora_target_modules"] = self.lora_target_modules
        if self.train_memory:
            train["memory"] = dict(self.train_memory)
        out["train"] = train
        return out


def _load(path: Path) -> ModelProfile:
    return ModelProfile.from_dict(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True), key=path.stem)


@lru_cache(maxsize=1)
def profiles() -> tuple[ModelProfile, ...]:
    """Every profile in configs/models/, read once. Keys and matches must be distinct."""
    paths = sorted(PROFILES_DIR.glob("*.yaml"))
    assert paths, f"no model profiles in {PROFILES_DIR}"
    out = tuple(_load(p) for p in paths)
    matches = [p.match for p in out]
    assert len(set(matches)) == len(matches), (
        f"two model profiles share a `match`: {sorted(m for m in matches if matches.count(m) > 1)}")
    return out


def model_keys() -> frozenset[str]:
    """The tokens this project spells its base models with — the profile file stems."""
    return frozenset(p.key for p in profiles())


def _squeeze(text: str) -> str:
    """Lowercase `text` down to letters and digits (`Qwen/Qwen3.6-27B` -> `qwenqwen3627b`)."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_profile(model: str) -> ModelProfile | None:
    """The profile a key, an HF id, a local path or a served name refers to, or None.

    A key resolves to its own file; anything else is matched by `match` as a substring
    of the squeezed input, longest needle first, so `Qwen/Qwen3.6-27B`, `qwen3_6` and
    `/root/qwen36` are one model under one token.
    """
    squeezed = _squeeze(model)
    for p in profiles():
        if p.key == squeezed:
            return p
    for p in sorted(profiles(), key=lambda p: -len(p.match)):
        if p.match in squeezed:
            return p
    return None


def _unregistered(model_id: str) -> ValueError:
    known = ", ".join(sorted(model_keys()))
    return ValueError(
        f"no model profile for {model_id!r}. Every name this project mints spells a model "
        f"with a registered key ({known}); nothing spells one by hand. Add "
        "configs/models/<key>.yaml (`model:` and `match:` at least) — adding the file is "
        "the deliberate moment to decide what the model is called forever.")


def model_key(model_id: str) -> str:
    """The registered token for a base model, refusing an unregistered one.

    Args:
        model_id: A profile key, an HF model id, a local weights path, or a served name.

    Returns:
        The token this project spells that model with, e.g. `qwen36`.

    Raises:
        ValueError: no profile covers the model. Permissive like `gpu_for` and unlike
            `model_profile`, in the sense that a model needs no VERIFIED profile to have
            a name — Qwen3-32B has a stub and no template.
    """
    p = find_profile(model_id)
    if p is None:
        raise _unregistered(model_id)
    return p.key


def _strip_none(value):
    """Drop None-valued keys at every depth (lists and dicts), copying as it goes."""
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def render_chat(tokenizer, messages: list[dict], tools: list[dict] | None = None, *,
                render_kwargs: dict, tokenize: bool = False,
                add_generation_prompt: bool = False, **extra):
    """Render one interchange conversation in a family's syntax — THE render site.

    Every place that turns stored messages into model text (train-time rendering, the
    mixture builder's token counts, the property ablations) comes through here, so tool
    use is handled once and the same way an eval serves it:

    - `tools` — the row's OpenAI-style function schemas — go to the template as
      `tools=`, exactly what an eval harness passes to the server (ODCV: `tools=` +
      `tool_choice="auto"`), so the template puts them where THIS family expects them.
      Qwen3.6 writes a `<tools>` block into the system turn; a row that carries tool
      calls but no `tools` is refused upstream (build_mixture), because a call to a
      function the model was never shown is not the behaviour being taught.
    - every `tool_calls[].function.arguments` reaches the template as a MAPPING: HF
      templates iterate argument pairs (Qwen3.6 raises on a string), while the OpenAI
      wire form is a JSON string, so a string is parsed here rather than at each site.
    - None-valued keys are dropped at every depth: HF's json loader pads dicts to a
      shared schema, and a padded `reasoning_content: None` or `arguments: None`
      must not reach the template.

    Args:
        tokenizer: The family's tokenizer (its chat template does the rendering).
        messages: Interchange messages (src/data/mixture/sources/).
        tools: The row's tool schemas, or None for a conversation without tools.
        render_kwargs: The profile's `render_kwargs` (preserve-thinking etc.).
        tokenize / add_generation_prompt / extra: Passed through to the template.
    """
    msgs = [_strip_none(m) for m in messages]
    for m in msgs:
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    kwargs = dict(render_kwargs)
    if tools:
        kwargs["tools"] = [_strip_none(t) for t in tools]
    return tokenizer.apply_chat_template(
        msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        **kwargs, **extra)


def model_profile(model_name: str) -> ModelProfile:
    """The VERIFIED profile for a base model, refusing unknown and unverified families.

    Args:
        model_name: A profile key or the base model id (e.g. "Qwen/Qwen3.6-27B").

    Raises:
        ValueError: No profile, or a stub with no verified `template:` block.
    """
    p = find_profile(model_name)
    if p is None:
        raise _unregistered(model_name)
    if not p.verified:
        known = ", ".join(sorted(q.key for q in profiles() if q.verified))
        raise ValueError(
            f"no verified thinking profile for model {model_name!r} (configs/models/"
            f"{p.key}.yaml has no `template:` block; verified: {known}). Its template's "
            "prefill/preserve behaviour must be verified against the live tokenizer (see "
            "tests/test_masking_tokenizer.py) and written into that file before this "
            "family can be trained or mixed. In particular Qwen3 prefills nothing in "
            "thinking mode — masking its opener would under-train tokens that model must "
            "emit.")
    return p


def _strip_none(value):
    """Drop None-valued keys at every depth (lists and dicts), copying as it goes."""
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def render_chat(tokenizer, messages: list[dict], tools: list[dict] | None = None, *,
                render_kwargs: dict, tokenize: bool = False,
                add_generation_prompt: bool = False, **extra):
    """Render one interchange conversation in a family's syntax — THE render site.

    Every place that turns stored messages into model text (train-time rendering, the
    mixture builder's token counts, the property ablations) comes through here, so tool
    use is handled once and the same way an eval serves it:

    - `tools` — the row's OpenAI-style function schemas — go to the template as
      `tools=`, exactly what an eval harness passes to the server (ODCV: `tools=` +
      `tool_choice="auto"`), so the template puts them where THIS family expects them.
      Qwen3.6 writes a `<tools>` block into the system turn; a row that carries tool
      calls but no `tools` is refused upstream (build_mixture), because a call to a
      function the model was never shown is not the behaviour being taught.
    - every `tool_calls[].function.arguments` reaches the template as a MAPPING: HF
      templates iterate argument pairs (Qwen3.6 raises on a string), while the OpenAI
      wire form is a JSON string, so a string is parsed here rather than at each site.
    - None-valued keys are dropped at every depth: HF's json loader pads dicts to a
      shared schema, and a padded `reasoning_content: None` or `arguments: None`
      must not reach the template.

    Args:
        tokenizer: The family's tokenizer (its chat template does the rendering).
        messages: Interchange messages (src/data/mixture/sources/).
        tools: The row's tool schemas, or None for a conversation without tools.
        render_kwargs: The profile's `render_kwargs` (preserve-thinking etc.).
        tokenize / add_generation_prompt / extra: Passed through to the template.
    """
    msgs = [_strip_none(m) for m in messages]
    for m in msgs:
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    kwargs = dict(render_kwargs)
    if tools:
        kwargs["tools"] = [_strip_none(t) for t in tools]
    return tokenizer.apply_chat_template(
        msgs, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        **kwargs, **extra)


# Serving stays permissive where training refuses: an unverified family (Qwen3-32B, a stub
# until its masking is verified) can still be served ad hoc. It has no verified ceiling, so
# the context-window fail-fast is skipped and vLLM's own startup failure is the backstop.
# The parser and prefix-caching facts are absent rather than guessed: an eval that REQUIRES
# tool calls is refused on such a family instead of being served with an unverified parser.
DEFAULT_SERVING = {"max_num_seqs": None}


def train_memory_entry(profile: ModelProfile, device_name: str) -> dict | None:
    """The measured training-memory entry for a live GPU, or None.

    Substring match ("H200" in "NVIDIA H200") so probe-time and train-time naming
    need not agree exactly. None simply means no measurement exists for this GPU —
    callers fall back to data-demonstrated limits, they never guess.
    """
    for key, entry in profile.train_memory.items():
        if key.upper() in device_name.upper():
            return {"gpu": key, **entry}
    return None


def gpu_for(model_name: str, role: str) -> str | None:
    """The RunPod GPU type this family needs for `role`, or None when it states none.

    One place per model, read by both provisioning paths: `uv run runpod up`
    (role="train") and the vLLM serving launch (role="inference"). The COUNT is not here
    and never will be — see `ModelProfile.gpu`. Permissive: an unregistered or stub
    family gets None and the caller falls back to its own default.

    Args:
        model_name: A profile key or base model id (e.g. "Qwen/Qwen3.6-27B").
        role: "train" or "inference".
    """
    assert role in ("train", "inference"), f"role must be train|inference, got {role!r}"
    p = find_profile(model_name)
    return p.gpu.get(role) if p else None


# VRAM per RunPod catalogue id, in GB: the one axis on which "big enough" is decided when
# SEVERAL models have to share a pod (an eval ladder — `uv run runpod up --eval a b c`).
# Written out rather than parsed off the id, because "NVIDIA H100 80GB HBM3" carries its
# size and "NVIDIA H200" does not, and a regex that guesses would silently under-rent on
# exactly the card whose name has no number in it. Add a row when a profile names a card.
GPU_VRAM_GB = {
    "NVIDIA H100 80GB HBM3": 80,
    "NVIDIA H200": 141,
    "NVIDIA B200": 180,
}


def largest_gpu(cards: list[str]) -> str:
    """The card out of `cards` that every one of them fits on: the largest by VRAM.

    Used when one pod must serve several targets whose families name different inference
    cards. Refuses an id it cannot rank rather than picking arbitrarily — an unranked
    card is one nobody has written the VRAM down for, and guessing there rents a box that
    OOMs 20 minutes into a boot.
    """
    assert cards, "largest_gpu needs at least one card"
    unranked = sorted(set(cards) - set(GPU_VRAM_GB))
    assert not unranked, (
        f"no VRAM on record for {unranked}: add the row to GPU_VRAM_GB in "
        f"{__name__} so a shared pod can be sized against it")
    return max(cards, key=lambda card: GPU_VRAM_GB[card])


def serving_params(model_name: str) -> dict:
    """vLLM serving parameters for a base model: its profile's `serving`, else defaults."""
    p = find_profile(model_name)
    return p.serving if p and p.serving else DEFAULT_SERVING


_ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def think_census(texts) -> dict:
    """Count assistant turns by think content across rendered rows.

    The preserve-thinking policy's yardstick: under `thinking: true` every assistant turn
    carries a think block, so `absent` must be 0; the empty share is a data-quality
    signal, reported by callers rather than asserted here.

    Returns:
        {turns, real, empty, absent}: assistant turns with a non-empty think block, with
        only empty ones, and with none at all.
    """
    turns = real = empty = 0
    for text in texts:
        for m in _ASSISTANT_TURN.finditer(text):
            turns += 1
            blocks = _THINK_BLOCK.findall(m.group(1))
            if not blocks:
                continue
            if any(b.strip() for b in blocks):
                real += 1
            else:
                empty += 1
    return {"turns": turns, "real": real, "empty": empty,
            "absent": turns - real - empty}


_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"<think>(.*)", re.DOTALL)


def split_think(text: str) -> tuple[str, str]:
    """Separate a Qwen3 reasoning trace from the user-visible answer.

    An unterminated `<think>` is treated as all-trace with an empty answer rather than
    being silently kept as prose: that shape means the generation was cut off mid-trace,
    and folding it into the answer would feed the judge a truncated ramble while hiding
    the truncation from the degeneracy counters.

    Args:
        text: Raw completion text.

    Returns:
        `(think, answer)`, both stripped. `think` is "" when no trace is present.
    """
    if not text:
        return "", ""
    close_idx = text.find("</think>")
    if close_idx != -1 and "<think>" not in text[:close_idx]:
        # Prefilled-generation shape: thinking-mode serving prefills `<think>\n` inside
        # the prompt (pin_template / Qwen3.6's own template), and vLLM returns only
        # generated tokens — so the trace arrives with its CLOSE tag alone. Missing
        # this shape reports a reasoning model as 100% empty-think AND leaks the raw
        # trace into the visible answer (first live psychosis run, 2026-08-05).
        return text[:close_idx].strip(), text[close_idx + len("</think>"):].strip()
    match = _THINK.search(text)
    if match:
        return match.group(1).strip(), _THINK.sub("", text, count=1).strip()
    open_match = _OPEN_THINK.search(text)
    if open_match:
        return open_match.group(1).strip(), ""
    return "", text.strip()


def resolve_trace(content: str | None, reasoning: str | None) -> tuple[str, str]:
    """Split a completion into `(think, answer)` across every shape vLLM returns.

    Three shapes exist in the wild and every eval on a served target has to handle all
    of them, because getting this wrong reports a normally-reasoning model as having a
    collapsed `<think>` block (CLAUDE.md gotcha 2) — a false alarm on the exact failure
    mode the empty-think metric is supposed to detect:

    - **No reasoning parser configured.** The trace arrives inline in `content`, wrapped
      in `<think>` tags.
    - **Parser configured** (`--reasoning-parser qwen3`). The trace arrives out of band
      and `content` holds only the visible answer. The out-of-band field is named
      `reasoning_content` on vLLM 0.8.x and `reasoning` on 0.26 — the caller passes
      whichever it found.
    - **Thinking disabled.** No trace at all; `content` is the bare answer.

    Args:
        content: The `message.content` field, possibly `None`/empty.
        reasoning: The out-of-band trace, from whichever field carried it.

    Returns:
        `(think, answer)`, both stripped.
    """
    raw = content or ""
    think, answer = split_think(raw)
    if reasoning and not think:
        # An out-of-band trace means `content` was never a container for it, so the
        # whole of `content` is the answer — including the case where content is empty
        # because generation was cut off mid-trace, which must stay an empty answer so
        # it scores as unparseable rather than silently borrowing the trace text.
        return str(reasoning).strip(), raw.strip()
    return think, answer
