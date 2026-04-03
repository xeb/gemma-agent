#!/usr/bin/env bash
set -euo pipefail

SCRIPT="gemma4.py"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== gemma4.py integration test ==="
echo ""

# Send two turns then quit, capture all output
OUTPUT=$(printf 'Hello\nWhat did I just say?\nquit\n' | uv run "$SCRIPT" 2>&1)

echo "--- captured output ---"
echo "$OUTPUT"
echo "--- end output ---"
echo ""

# Test 1: prompt appears
echo "$OUTPUT" | grep -q "gemma>" && pass "gemma> prompt present" || fail "gemma> prompt not found"

# Test 2: stats line present (e.g. "[5.3s | 11 tok | 2.1 tok/s | ctx: 0.2KB ~46tok]")
echo "$OUTPUT" | grep -qE "^\[" && pass "stats line present" || fail "stats line not found"

# Test 3: turn timing shown (e.g. "5.3s")
echo "$OUTPUT" | grep -qE "\[[0-9]+\.[0-9]+s" && pass "turn timing shown" || fail "turn timing not found"

# Test 4: output tokens shown (e.g. "11 tok")
echo "$OUTPUT" | grep -qE "[0-9]+ tok \|" && pass "output tokens shown" || fail "output tokens not found"

# Test 5: tokens per second shown (e.g. "2.1 tok/s")
echo "$OUTPUT" | grep -qE "[0-9]+\.[0-9]+ tok/s" && pass "tok/s shown" || fail "tok/s not found"

# Test 6: context size in KB shown (e.g. "ctx: 0.2KB")
echo "$OUTPUT" | grep -qE "ctx: [0-9]+\.[0-9]+KB" && pass "context KB shown" || fail "context KB not found"

# Test 7: estimated tokens shown (e.g. "~46tok")
echo "$OUTPUT" | grep -qE "~[0-9]+tok" && pass "estimated tokens shown" || fail "estimated tokens not found"

# Test 8: multi-turn — stats line appears at least twice (one per turn)
STATS_COUNT=$(echo "$OUTPUT" | grep -cE "^\[" || true)
[ "$STATS_COUNT" -ge 2 ] && pass "multi-turn: $STATS_COUNT stats lines" || fail "multi-turn: expected >=2 stats lines, got $STATS_COUNT"

# Test 9: model produced a non-empty response
echo "$OUTPUT" | grep -qE "^.{10,}" && pass "model produced non-empty response" || fail "model response looks empty"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
