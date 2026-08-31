# ABOUTME: Convert the Qwen-rendered t2_9284_da716 mixture into gpt-oss Harmony format,
# ABOUTME: mapping real <think> traces onto the analysis channel and dropping empty ones.

"""Re-render a Qwen-format training mixture for gpt-oss.

Run: uv run python scratch/build_harmony_dataset.py [--out data/t2_da716_harmony.jsonl]

The source mixture is PRE-RENDERED to Qwen3.6's chat format, so it cannot be handed to
gpt-oss as-is. This parses it back into role/content/reasoning messages and re-renders with
gpt-oss's own `apply_chat_template`, which is the only trustworthy source for Harmony's
exact token layout.

THE EMPTY-THINK QUESTION. Qwen renders think tags on every assistant turn, so a
non-reasoning row MUST carry the exact empty marker `<think>\\n\\n</think>\\n\\n` — omitting
it is malformed. Harmony works the opposite way, verified by rendering the model's own
template (2026-08-15):

    with reasoning : <|start|>assistant<|channel|>analysis<|message|>R<|end|>
                     <|start|>assistant<|channel|>final<|message|>A<|return|>
    no reasoning   : <|start|>assistant<|channel|>final<|message|>A<|return|>

The analysis channel is simply ABSENT. There is no empty-analysis form, so the Qwen empty
marker must be DROPPED rather than translated — emitting an empty analysis block would
train the model to produce vacuous reasoning turns it would never otherwise generate.

THE NUMINAMATH SPLIT. The numinamath_cot rows carry their step-by-step derivation in the
visible answer with no reasoning trace at all, because the source mixture never separated
the two. Rendered naively they train gpt-oss to dump derivations into the user-facing
`final` channel while never opening `analysis` — the exact inversion of Harmony's contract.
`split_cot` therefore cuts each answer at its concluding \\boxed{...} result: the derivation
becomes the analysis channel, the boxed conclusion stays final. This makes the arm differ
from the Qwen run in data as well as model, which any write-up comparing them must state.
"""

import json
import re
from collections import Counter
from pathlib import Path

import fire
from datasets import load_dataset
from dotenv import load_dotenv

SRC_REPO = "LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train"
SRC_FILE = "t2_9284_da716_10k.jsonl"
MODEL = "openai/gpt-oss-120b"

# Qwen turn: <|im_start|>{role}\n{body}<|im_end|>
_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_THINK = re.compile(r"^<think>\n(.*?)\n?</think>\n\n(.*)$", re.DOTALL)
# Qwen embeds tool SCHEMAS in the system turn and tool CALLS in assistant content, both as
# literal markup. Harmony expresses both structurally instead, so they must be lifted out
# and handed to the template rather than re-rendered as text.
_TOOLS = re.compile(r"<tools>\s*(\[.*?\])\s*</tools>", re.DOTALL)
_TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


_TYPE_MAP = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
             "list": "array", "dict": "object", "tuple": "array", "any": "string"}


def _normalise_tool(spec: dict) -> dict | None:
    """Coerce a Qwen tool spec into the nested JSON-Schema form the Harmony template needs.

    The apigen rows carry TWO shapes: 1,194 already-nested OpenAI specs
    ({"type": "function", "function": {...}}) and 1,591 FLAT ones
    ({"name", "description", "parameters"}) whose `parameters` is a bare param->info dict
    using Python type names ("str, optional") rather than JSON Schema. The template does
    `tool.function...` and raises UndefinedError on the flat shape, so the flat ones are
    wrapped and their parameters rebuilt as a proper object schema.

    Args:
        spec: One tool entry as it appears in Qwen's <tools> block.

    Returns:
        A nested {"type": "function", "function": {...}} spec, or None if unusable.
    """
    if "function" in spec:
        return spec
    if "name" not in spec:
        return None
    props, required = {}, []
    for pname, info in (spec.get("parameters") or {}).items():
        if not isinstance(info, dict):
            continue
        raw = str(info.get("type", "string"))
        base = raw.split(",")[0].strip().lower()
        props[pname] = {"type": _TYPE_MAP.get(base, "string"),
                        "description": str(info.get("description", ""))}
        # "optional" anywhere in the type string is the only optionality marker present.
        if "optional" not in raw.lower():
            required.append(pname)
    return {"type": "function", "function": {
        "name": spec["name"], "description": str(spec.get("description", "")),
        "parameters": {"type": "object", "properties": props, "required": required}}}


def _lift_tools(content: str) -> tuple[str, list[dict]]:
    """Pull Qwen's `<tools>[...]</tools>` schema block out of a system message.

    Args:
        content: System-turn text.

    Returns:
        (content without the block, OpenAI-format tool specs) — empty list if absent.
    """
    m = _TOOLS.search(content)
    if not m:
        return content, []
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return content, []
    tools = [n for n in (_normalise_tool(s) for s in raw if isinstance(s, dict)) if n]
    rest = content[:m.start()] + content[m.end():]
    # Qwen's system prose also DEMONSTRATES its own call format ("The output MUST strictly
    # adhere to the following format ... <tool_call>[...]</tool_call>"). Left in, it sits
    # directly above the template's Harmony tool block and instructs the model to do the
    # opposite of what Harmony requires, so the example and its lead-in are removed.
    rest = _TOOLCALL.sub("", rest)
    rest = re.sub(r"(?im)^.*\bstrictly adhere to the following format\b.*$", "", rest)
    rest = re.sub(r"(?im)^.*\bthe example format is as follows\b.*$", "", rest)
    rest = re.sub(r"\n{3,}", "\n\n", rest)
    return rest.strip(), tools


def _lift_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Convert Qwen's `<tool_call>[...]</tool_call>` payload into OpenAI tool_calls.

    Qwen renders calls as literal XML inside the assistant's visible content. Harmony
    routes them to the `commentary` channel with a `to=functions.NAME` recipient and a
    `<|call|>` terminator — a different mechanism, not a different spelling. Re-rendering
    the XML verbatim (the first version of this script) trained the model to TYPE markup
    at the user instead of invoking anything; 987 rows were affected.

    Args:
        content: Assistant-turn text.

    Returns:
        (content with the block removed, tool_calls in OpenAI format).
    """
    m = _TOOLCALL.search(content)
    if not m:
        return content, []
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return content, []
    calls = payload if isinstance(payload, list) else [payload]
    out = []
    for c in calls:
        if not isinstance(c, dict) or "name" not in c:
            continue
        args = c.get("arguments", {})
        # Arguments must be a DICT, not a JSON string. The template json-encodes whatever
        # it is given, so a string becomes a double-encoded "{\"city\": \"Paris\"}"
        # payload instead of the {"city": "Paris"} object Harmony specifies.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append({"type": "function",
                    "function": {"name": c["name"], "arguments": args}})
    return (content[:m.start()] + content[m.end():]).strip(), out


def _cut(content: str, sep: str) -> tuple[str, str] | None:
    """Split at the LAST chunk containing \\boxed, at the given granularity.

    The last one, not the first: a derivation may box intermediate results, and only the
    final box is the answer. Returns None when \\boxed is absent or sits in the opening
    chunk — the latter means the answer is stated before any derivation, so there is
    nothing to move.

    Args:
        content: Assistant answer text.
        sep: Chunk separator, "\\n\\n" for paragraphs or "\\n" for lines.

    Returns:
        (reasoning, answer) or None if no usable cut point exists.
    """
    parts = content.split(sep)
    idx = max((i for i, p in enumerate(parts) if "\\boxed" in p), default=None)
    if idx is None or idx == 0:
        return None
    head, tail = sep.join(parts[:idx]).strip(), sep.join(parts[idx:]).strip()
    return (head, tail) if head and tail else None


def split_cot(content: str) -> tuple[str, str] | None:
    """Separate a chain-of-thought answer into (reasoning, boxed conclusion).

    Two granularities, because the corpus uses both: most rows separate steps with blank
    lines, but ~10% are a single paragraph using markdown line breaks. Paragraphs are tried
    first so a multi-paragraph answer is never chopped mid-derivation.

    Args:
        content: The visible answer, derivation and conclusion combined.

    Returns:
        (reasoning, answer) or None to leave the row untouched.
    """
    return _cut(content, "\n\n") or _cut(content, "\n")


def parse_qwen(text: str) -> list[dict]:
    """Parse a Qwen-rendered conversation back into interchange messages.

    Args:
        text: The rendered conversation.

    Returns:
        Messages with `role`, `content`, and `thinking` where a real trace was present.
        An empty think marker yields no `thinking` key at all.
    """
    out, tools = [], []
    for role, body in _TURN.findall(text):
        if role != "assistant":
            content = body.strip()
            if role == "system":
                content, tools = _lift_tools(content)
            out.append({"role": role, "content": content})
            continue
        m = _THINK.match(body)
        trace, answer = ("", body.strip()) if not m else (m.group(1).strip(),
                                                          m.group(2).strip())
        answer, calls = _lift_tool_calls(answer)
        msg = {"role": "assistant", "content": answer}
        if trace:                      # empty marker -> no analysis channel at all
            msg["thinking"] = trace
        if calls:
            msg["tool_calls"] = calls
        out.append(msg)
    return out, tools


def main(out: str = "data/t2_da716_harmony.jsonl", limit: int = 0,
         cot_sources: tuple[str, ...] = ("numinamath_cot",)) -> None:
    """Build the Harmony-rendered dataset and report what the conversion produced.

    Args:
        out: Output JSONL path.
        limit: Stop after N rows (0 = all); for a fast wiring check.
        cot_sources: Sources whose answers embed a derivation that should be moved to the
            analysis channel. Scoped by source rather than applied corpus-wide: \\boxed is
            a maths convention, and a blanket rule would cut up any other row that happened
            to use it.
    """
    load_dotenv()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ds = load_dataset(SRC_REPO, data_files=SRC_FILE, split="train")
    rows = list(ds)[:limit] if limit else list(ds)
    print(f"source rows: {len(rows)}")

    written, stats = [], Counter()
    for r in rows:
        msgs, tools = parse_qwen(r["text"])
        assert msgs, f"parsed no turns from a row (source={r.get('source')})"
        assert msgs[-1]["role"] == "assistant", "last turn must be the assistant's"
        if r.get("source") in cot_sources:
            for m in msgs:
                # Only where no trace exists: a row that already reasoned has its channels
                # right, and re-cutting its answer would move content out of `final`.
                if m["role"] != "assistant" or m.get("thinking"):
                    continue
                cut = split_cot(m["content"])
                stats["cot_rows"] += 1
                if cut:
                    m["thinking"], m["content"] = cut
                    stats["cot_split"] += 1
        n_think = sum(1 for m in msgs if m.get("thinking"))
        stats["turns"] += sum(1 for m in msgs if m["role"] == "assistant")
        stats["with_analysis"] += n_think
        stats["rows_with_analysis"] += bool(n_think)
        stats["rows_with_tools"] += bool(tools)
        stats["tool_calls"] += sum(len(m.get("tool_calls") or []) for m in msgs)
        # tools= makes the template emit the `namespace functions {...}` developer block
        # and route calls to the commentary channel; without it a tool_calls message
        # cannot be rendered in Harmony at all.
        text = (tok.apply_chat_template(msgs, tools=tools, tokenize=False) if tools
                else tok.apply_chat_template(msgs, tokenize=False))
        written.append({"text": text, "source": r.get("source", "?"),
                        "messages": msgs, "tools": tools})

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for w in written:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    an = sum(1 for w in written if "<|channel|>analysis<|message|>" in w["text"])
    fi = sum(1 for w in written if "<|channel|>final<|message|>" in w["text"])
    print(f"assistant turns          : {stats['turns']}")
    print(f"  with an analysis channel: {stats['with_analysis']}")
    print(f"  final-channel only      : {stats['turns'] - stats['with_analysis']}")
    print(f"rendered rows containing analysis channel: {an}")
    if stats["cot_rows"]:
        left = stats["cot_rows"] - stats["cot_split"]
        print(f"CoT answers split into analysis+final     : {stats['cot_split']}"
              f"/{stats['cot_rows']}  ({left} left intact: no \\boxed, or the answer "
              f"precedes the derivation)")
    # A row ends EITHER with a final-channel answer or, for tool-calling rows, with a
    # commentary-channel invocation terminated by <|call|>. The original assertion
    # demanded a final channel on every row and fired once tool calls were lifted
    # correctly -- the check was wrong, not the data.
    calls_only = sum(1 for w in written
                     if "<|channel|>final<|message|>" not in w["text"]
                     and "<|call|>" in w["text"])
    print(f"rendered rows with a final channel       : {fi}")
    print(f"rendered rows ending in a tool call only : {calls_only}")
    assert fi + calls_only == len(written), (
        f"{len(written) - fi - calls_only} rows end in neither a final channel nor a "
        f"tool call")
    # Tool calls MUST land in the commentary channel with a to=functions recipient; any
    # surviving Qwen <tool_call> markup means the lift failed and the row would train the
    # model to type XML at the user instead of invoking a tool.
    leaked = sum(1 for w in written if "<tool_call>" in w["text"])
    comm = sum(1 for w in written if "<|channel|>commentary" in w["text"])
    recip = sum(1 for w in written if "to=functions." in w["text"])
    print(f"\nrows declaring tools      : {stats['rows_with_tools']}")
    print(f"tool calls lifted         : {stats['tool_calls']}")
    print(f"  rendered to commentary  : {comm}")
    print(f"  with to=functions.NAME  : {recip}")
    print(f"  LEAKED Qwen <tool_call> : {leaked}   <- must be 0")
    assert leaked == 0, "Qwen tool-call markup survived into the Harmony rendering"
    print(f"\nwrote {p}  ({len(written)} rows)")


if __name__ == "__main__":
    fire.Fire(main)
