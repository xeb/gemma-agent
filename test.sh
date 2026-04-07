#!/usr/bin/env bash
set -euo pipefail

PASS=0
FAIL=0
BINARY="./target/release/gemma-agent"
GGUF="./gguf-cache/google_gemma-4-E4B-it-Q4_K_M.gguf"

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== gemma-agent tests ==="
echo ""

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo ">> building..."
    cargo build --release 2>&1 | tail -2
fi

# --- Test: --help ---
echo ">> --help"
OUT=$("$BINARY" --help 2>&1) || true
echo "$OUT" | grep -q "gemma-agent" && pass "--help shows gemma-agent" || fail "--help missing gemma-agent"
echo "$OUT" | grep -q "\-\-prompt" && pass "--help shows --prompt" || fail "--help missing --prompt"
echo "$OUT" | grep -q "\-\-list-models" && pass "--help shows --list-models" || fail "--help missing --list-models"
echo ""

# --- Test: --list-models ---
echo ">> --list-models"
OUT=$("$BINARY" --list-models 2>&1)
echo "$OUT" | grep -q "e4b" && pass "--list-models shows e4b" || fail "--list-models missing e4b"
echo "$OUT" | grep -q "e2b" && pass "--list-models shows e2b" || fail "--list-models missing e2b"
echo ""

# --- Test: no args (no embedded weights) ---
echo ">> no args (should fail gracefully)"
OUT=$("$BINARY" 2>&1) || true
echo "$OUT" | grep -qi "no embedded weights\|build\.sh\|gguf" && pass "no-args gives helpful message" || fail "no-args error unclear"
echo ""

# --- Test: model alias instead of path ---
echo ">> model alias as arg"
OUT=$("$BINARY" e4b 2>&1) || true
echo "$OUT" | grep -qi "alias\|build\.sh" && pass "alias gives helpful message" || fail "alias error unclear"
echo ""

# --- Test: nonexistent file ---
echo ">> nonexistent GGUF path"
OUT=$("$BINARY" /tmp/does_not_exist.gguf 2>&1) || true
echo "$OUT" | grep -qi "not found" && pass "missing file error" || fail "missing file no error"
echo ""

# --- Tests requiring GGUF ---
if [ ! -f "$GGUF" ]; then
    echo ">> SKIPPING inference tests (no GGUF at $GGUF)"
    echo ""
    echo "=== Results: $PASS passed, $FAIL failed (inference tests skipped) ==="
    [ "$FAIL" -eq 0 ] && exit 0 || exit 1
fi

# --- Test: model loads successfully ---
echo ">> model loading"
OUT=$(timeout 60 "$BINARY" "$GGUF" --prompt="say hello" 2>&1) || true
if echo "$OUT" | grep -q "model load failed"; then
    # Check if it's a GPU OOM that should have fallen back to CPU
    if echo "$OUT" | grep -qi "cuda\|CUDA\|gpu\|GPU\|alloc"; then
        fail "model load failed (GPU OOM without CPU fallback)"
    else
        fail "model load failed: $(echo "$OUT" | tail -1)"
    fi
else
    echo "$OUT" | grep -qi "ready\|hello\|hi\|greet" && pass "model loads and responds" || fail "model loaded but no response"
fi
echo ""

# --- Test: tool calling with --prompt ---
echo ">> tool calling (--prompt)"
OUT=$(timeout 60 "$BINARY" "$GGUF" --prompt="what time is it?" 2>&1) || true
if echo "$OUT" | grep -q "tool.*bash"; then
    pass "tool call detected"
else
    fail "no tool call for 'what time is it?'"
fi
if echo "$OUT" | grep -qE "[0-9]{2}:[0-9]{2}"; then
    pass "time appears in output"
else
    fail "no time in output"
fi
echo ""

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
