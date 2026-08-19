# ABOUTME: One-off check — load every configured corpus on every channel and print
# ABOUTME: row counts + empty-channel drops, so schema surprises die before GPU time.

import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import load_corpus, CHANNELS  # noqa: E402

cfg = OmegaConf.load(REPO_ROOT / "configs/properties/sae_diff.yaml")
for spec in cfg.corpora:
    spec = OmegaConf.to_container(spec)
    for channel in CHANNELS:
        try:
            df = load_corpus(spec, channel)
            lens = df.text.str.len()
            print(f"  -> {spec['name']}/{channel}: {len(df)} docs, "
                  f"chars p50={int(lens.median())} p95={int(lens.quantile(0.95))}")
        except Exception as e:
            print(f"  !! {spec['name']}/{channel}: {type(e).__name__}: {e}")
