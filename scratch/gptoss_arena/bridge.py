# ABOUTME: Reuse the qualified native MMLU transport with an independent Arena ledger.
# ABOUTME: The frozen preparation receipt supplies the authorized ceiling.
import json
from pathlib import Path
from scratch.gptoss_mmlu.bridge import main
if __name__ == "__main__":
    root = Path("/work/output/gptoss_arena_20261014")
    policy = json.loads((root / "prepared.json").read_text())
    main(root=root, ceiling=policy["ceiling_usd"])
