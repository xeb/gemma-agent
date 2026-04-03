#!/usr/bin/env bash
set -euo pipefail

SCRIPT="gemma4.py"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== gemma4.py tests ==="
echo ""

# --- Test: --list-models ---
echo ">> --list-models"
LIST_OUTPUT=$(uv run "$SCRIPT" --list-models 2>&1)
echo "$LIST_OUTPUT" | grep -q "e4b" && pass "--list-models shows e4b" || fail "--list-models missing e4b"
echo "$LIST_OUTPUT" | grep -q "mlx-8bit" && pass "--list-models shows mlx-8bit" || fail "--list-models missing mlx-8bit"
echo "$LIST_OUTPUT" | grep -q "mlx-4bit" && pass "--list-models shows mlx-4bit" || fail "--list-models missing mlx-4bit"
echo ""

# --- Test: multi-turn chat (transformers backend) ---
echo ">> multi-turn chat"
OUTPUT=$(printf 'Hello\nWhat did I just say?\nquit\n' | uv run "$SCRIPT" 2>&1)

echo "--- captured output ---"
echo "$OUTPUT"
echo "--- end output ---"
echo ""

echo "$OUTPUT" | grep -q "gemma>" && pass "gemma> prompt present" || fail "gemma> prompt not found"
echo "$OUTPUT" | grep -qE "\[[0-9]+\.[0-9]+s" && pass "turn timing shown" || fail "turn timing not found"
echo "$OUTPUT" | grep -qE "[0-9]+ tok \|" && pass "output tokens shown" || fail "output tokens not found"
echo "$OUTPUT" | grep -qE "[0-9]+\.[0-9]+ tok/s" && pass "tok/s shown" || fail "tok/s not found"
echo "$OUTPUT" | grep -qE "ctx: [0-9]+\.[0-9]+KB" && pass "context KB shown" || fail "context KB not found"
echo "$OUTPUT" | grep -qE "~[0-9]+tok" && pass "estimated tokens shown" || fail "estimated tokens not found"

STATS_COUNT=$(echo "$OUTPUT" | grep -cE "\[[0-9]+\.[0-9]+s" || true)
[ "$STATS_COUNT" -ge 2 ] && pass "multi-turn: $STATS_COUNT stats lines" || fail "multi-turn: expected >=2 stats lines, got $STATS_COUNT"

echo "$OUTPUT" | grep -qE "^.{10,}" && pass "model produced non-empty response" || fail "model response looks empty"

# --- Test: --model flag with explicit repo ---
echo ""
echo ">> --model flag"
MODEL_OUTPUT=$(printf 'Hi\nquit\n' | uv run "$SCRIPT" --model e4b 2>&1)
echo "$MODEL_OUTPUT" | grep -q "gemma>" && pass "--model e4b works" || fail "--model e4b failed"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
