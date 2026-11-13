#!/usr/bin/env bash
# ABOUTME: Steps 1-2 of the Shen software feasibility checks, on the pod: one raw Anthropic-style
# ABOUTME: /v1/messages tool request, then a hello-world task through Claude Code, per served model.
# Run (on the pod): bash pod_hello.sh <served model name>
set -uo pipefail
MODEL="$1"
ROOT=/workspace/shen
CLAUDE="$ROOT/venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"
OUT="$ROOT/hello/$MODEL"
rm -rf "$OUT"; mkdir -p "$OUT/work" "$OUT/claude_config"

# Step 1: raw /v1/messages request with one tool; a pass returns a tool_use block.
cat > "$OUT/raw_request.json" <<EOF
{"model": "$MODEL", "max_tokens": 4096,
 "tools": [{"name": "get_weather", "description": "Get the current weather for a city.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                             "required": ["city"]}}],
 "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]}
EOF
curl -s http://127.0.0.1:${PORT:-8011}/v1/messages -H "content-type: application/json" \
  -H "x-api-key: EMPTY" -H "anthropic-version: 2023-06-01" \
  -d @"$OUT/raw_request.json" > "$OUT/raw_response.json"

# Step 2: the hello-world task through Claude Code against the same server.
cd "$OUT/work"
env -i HOME="$HOME" PATH="$PATH" TERM=dumb IS_SANDBOX=1 \
  CLAUDE_CONFIG_DIR="$OUT/claude_config" \
  ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT:-8011} ANTHROPIC_AUTH_TOKEN=EMPTY ANTHROPIC_API_KEY= \
  ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \
  "$CLAUDE" -p "Create a file hello.py that prints the result of 2+2, then run it with python3 and tell me the output." \
    --model "$MODEL" --permission-mode acceptEdits \
    --allowedTools "Write" "Read" "Bash(python3:*)" \
    --output-format stream-json --verbose --max-turns 10 \
  > "$OUT/transcript.jsonl" 2> "$OUT/stderr.txt"
echo "claude exit $?" > "$OUT/done.txt"
