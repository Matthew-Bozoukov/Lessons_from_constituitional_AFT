# ABOUTME: Offline smoke — corpus channel extraction, the 70B d_in patch, GoodfireSAE
# ABOUTME: metadata round-trip, and package imports. No network, no GPU, no key.

import warnings

warnings.filterwarnings("ignore")

from corpus import channel_text  # noqa: E402

row6 = {"system": "s", "user": "the question", "reasoning": "the trace", "response": "the answer"}
rowi = {"messages": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": "a", "reasoning_content": "r"}]}
assert channel_text(row6, "query") == "the question"
assert channel_text(row6, "reasoning") == "the trace"
assert channel_text(rowi, "response") == "a"
assert channel_text(rowi, "reasoning") == "r"

from interp_embed.sae import GoodfireSAE  # noqa: E402
from interp_embed.sae.utils import get_goodfire_config_from_hf  # noqa: E402

cfg70 = get_goodfire_config_from_hf("Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
                                    "Llama-3.3-70B-Instruct-SAE-l50.pt", "cpu", use_8b_model=False)
assert cfg70["d_in"] == 8192 and cfg70["d_sae"] == 65536, cfg70
assert cfg70["hook_name"] == "blocks.50.hook_resid_post", cfg70

sae = GoodfireSAE(variant_name="Llama-3.3-70B-Instruct-SAE-l50", device={"model": "auto", "sae": "cpu"})
md = sae.metadata()
assert md["dtype"] == "bfloat16" and md["max_length"] == 2048, md

from interp_embed import Dataset  # noqa: E402, F401

print("offline smoke OK: corpus channels, 70B d_in patch, GoodfireSAE metadata, Dataset import")
