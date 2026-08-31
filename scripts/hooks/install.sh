#!/usr/bin/env bash
# ABOUTME: Install this repo's git hooks (pre-push naming gate) into .git/hooks.
# ABOUTME: Run once per clone: `bash scripts/hooks/install.sh`.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
install -m 0755 "$root/scripts/hooks/pre-push" "$root/.git/hooks/pre-push"
echo 'installed .git/hooks/pre-push — naming violations now block git push'
