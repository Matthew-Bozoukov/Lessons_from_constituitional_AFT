# ABOUTME: Audit the pinned plain control with the pinned Gemma tokenizer before a GPU rental.
# ABOUTME: Retains row-level length, supervision and prefix-gate evidence without modifying data.
import json
from pathlib import Path
from collections import Counter
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from src.model_profile import model_profile, render_chat
from src.train.masking import build_labels
from src.train.gemma4_mask import expected_from_messages

MODEL = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
DATA = "dougalldeepmind/2026-10-05-plain-mix"
DATA_REVISION = "a5b115666913790be37482f83f6aa843c73e258a"


def main(output="output/gemma4-control-audit"):
    dest = Path(output); dest.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    profile = model_profile(MODEL)
    data = hf_hub_download(DATA, "mixture.jsonl", repo_type="dataset", revision=DATA_REVISION)
    rows = [json.loads(line) for line in Path(data).open(encoding="utf-8") if line.strip()]
    evidence = []
    for i, row in enumerate(rows):
        entry = {"row": i, "source": row["source"], "tool": bool(row.get("tools"))}
        try:
            text = render_chat(tok, row["messages"], row.get("tools"), render_kwargs=profile.render_kwargs)
            out = build_labels(text, tok, 100000, profile, supervise=row.get("supervise", "full"))
            rendered, expected = expected_from_messages(tok, row, profile)
            actual = tok.decode([v for v in out["labels"] if v != -100])
            assert rendered == text and actual == expected, "mask/prefix disagreement"
            entry.update(tokens=len(out["input_ids"]), supervised=sum(v != -100 for v in out["labels"]), verified=True)
        except Exception as exc:
            entry.update(error=f"{type(exc).__name__}: {exc}", verified=False)
        evidence.append(entry)
        if i % 200 == 0:
            print(f"AUDIT {i}/{len(rows)} failures={sum(not x['verified'] for x in evidence)}", flush=True)
    result = {"model": MODEL, "model_revision": REVISION, "dataset": DATA, "dataset_revision": DATA_REVISION,
              "rows": len(rows), "verified": sum(e['verified'] for e in evidence),
              "max_tokens": max(e.get("tokens", 0) for e in evidence),
              "supervised_tokens": sum(e.get("supervised", 0) for e in evidence),
              "over_8192": sum(e.get("tokens", 0) > 8192 for e in evidence), "evidence": evidence}
    (dest / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "evidence"}), flush=True)
    print("FAILURES", [e for e in evidence if not e["verified"]][:10], flush=True)
    assert result["verified"] == len(rows), "Resolve audit failures before training"


if __name__ == "__main__":
    import fire
    fire.Fire(main)
