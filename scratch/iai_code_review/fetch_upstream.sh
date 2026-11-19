# ABOUTME: Clone the Inter-Agent Influence repo (MIT) at the pinned commit into output/ and apply
# ABOUTME: our one local patch; run_code_review.py --upstream points at the result.
# Run: bash scratch/iai_code_review/fetch_upstream.sh   (from the repository root)
set -euo pipefail
COMMIT=2e70dcd
DEST=output/iai_code_review/upstream
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -d "$DEST/.git" ]; then
  git clone -q https://github.com/Cooperative-AI-Foundation/inter-agent-influence.git "$DEST"
fi
git -C "$DEST" checkout -q "$COMMIT"
git -C "$DEST" checkout -q -- evals/_inspect_patches.py evals/common/routing.py
git -C "$DEST" apply "$HERE/skip_anthropic_cache_patch.diff"
git -C "$DEST" apply "$HERE/pin_sonnet_4_6.diff"
echo "upstream at $(git -C "$DEST" rev-parse --short HEAD) + local patch:"
git -C "$DEST" diff --stat
