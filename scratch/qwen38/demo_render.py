# ABOUTME: Throwaway demo: what preserve_thinking and reasoning_effort change on Qwen3.8 vs 3.6.
# ABOUTME: Renders a tiny two-turn chat and prints the text and the tokens the mask trains.
from transformers import AutoTokenizer

from src.model_profile import model_profile, render_chat
from src.train.masking import build_labels

P = model_profile(
    "qwen36"
)  # same template literals; used only to mark supervised spans
t38 = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.8-27B", revision="37561816e66b5b9153b1b64c10968c18c1e6f4a0"
)
t36 = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.6-27B", revision="8f33b98937c6031bb49e713dee3e8e185aafd26f"
)
msgs = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
]


def show(tok, name, **kw):
    txt = render_chat(tok, msgs, None, render_kwargs=kw)
    sup = tok.decode(
        [v for v in build_labels(txt, tok, 4096, P)["labels"] if v != -100]
    )
    print(f"--- {name}\nRENDERED: {txt!r}\nTRAINED : {sup!r}\n")


show(t36, "3.6, as trained (no kwargs)")
show(
    t38,
    "3.8 preserve_thinking=False",
    preserve_thinking=False,
    reasoning_effort="medium",
)
show(
    t38,
    "3.8 preserve_thinking=True (its default)",
    preserve_thinking=True,
    reasoning_effort="medium",
)
chat = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
for e in ("xhigh", "medium", "low"):
    sys_turn = t38.apply_chat_template(chat, tokenize=False, reasoning_effort=e).split(
        "<|im_start|>user"
    )[0]
    print(f"--- 3.8 system turn, reasoning_effort={e}: {sys_turn!r}")
print(
    "--- 3.6 system turn:",
    repr(t36.apply_chat_template(chat, tokenize=False).split("<|im_start|>user")[0]),
)
