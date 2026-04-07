#!/usr/bin/env bash
set -euo pipefail

PASS=0
FAIL=0
RUST_BIN="./rust/target/release/gemma-agent"
GGUF="./rust/gguf-cache/google_gemma-4-E4B-it-Q4_K_M.gguf"
PYTHON="python/gemma_agent.py"

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== gemma-agent tests ==="
echo ""

# ─── Rust tests ───

echo "── Rust ──"
echo ""

# Build if needed
if [ ! -f "$RUST_BIN" ]; then
    echo ">> building rust..."
    (cd rust && cargo build --release 2>&1 | tail -2)
fi

echo ">> --help"
OUT=$("$RUST_BIN" --help 2>&1) || true
echo "$OUT" | grep -q "gemma-agent" && pass "rust: --help shows gemma-agent" || fail "rust: --help missing gemma-agent"
echo "$OUT" | grep -q "\-\-prompt" && pass "rust: --help shows --prompt" || fail "rust: --help missing --prompt"
echo ""

echo ">> --list-models"
OUT=$("$RUST_BIN" --list-models 2>&1)
echo "$OUT" | grep -q "e4b" && pass "rust: --list-models shows e4b" || fail "rust: --list-models missing e4b"
echo "$OUT" | grep -q "e2b" && pass "rust: --list-models shows e2b" || fail "rust: --list-models missing e2b"
echo ""

echo ">> no args (graceful error)"
OUT=$("$RUST_BIN" 2>&1) || true
echo "$OUT" | grep -qi "build\.sh\|gguf\|embedded" && pass "rust: no-args helpful message" || fail "rust: no-args unclear"
echo ""

echo ">> alias as arg"
OUT=$("$RUST_BIN" e4b 2>&1) || true
echo "$OUT" | grep -qi "alias\|build\.sh" && pass "rust: alias helpful message" || fail "rust: alias unclear"
echo ""

echo ">> nonexistent file"
OUT=$("$RUST_BIN" /tmp/does_not_exist.gguf 2>&1) || true
echo "$OUT" | grep -qi "not found" && pass "rust: missing file error" || fail "rust: missing file no error"
echo ""

# Inference tests (require GGUF)
if [ -f "$GGUF" ]; then
    echo ">> model loading"
    OUT=$(timeout 60 "$RUST_BIN" "$GGUF" --prompt="say hello" 2>&1) || true
    if echo "$OUT" | grep -q "model load failed"; then
        fail "rust: model load failed: $(echo "$OUT" | grep 'model load' | head -1)"
    else
        echo "$OUT" | grep -qi "ready\|hello\|hi" && pass "rust: model loads and responds" || fail "rust: no response"
    fi
    echo ""

    echo ">> tool calling"
    OUT=$(timeout 60 "$RUST_BIN" "$GGUF" --prompt="what time is it?" 2>&1) || true
    echo "$OUT" | grep -q "tool.*bash" && pass "rust: tool call detected" || fail "rust: no tool call"
    echo "$OUT" | grep -qE "[0-9]{2}:[0-9]{2}" && pass "rust: time in output" || fail "rust: no time"
    echo ""
else
    echo ">> SKIPPING rust inference tests (no GGUF at $GGUF)"
    echo ""
fi

# ─── Python tests ───

echo "── Python ──"
echo ""

if ! command -v uv >/dev/null 2>&1; then
    echo ">> SKIPPING python tests (uv not installed)"
    echo ""
else
    echo ">> --help"
    OUT=$(uv run "$PYTHON" --help 2>&1) || true
    echo "$OUT" | grep -q "\-\-prompt" && pass "python: --help shows --prompt" || fail "python: --help missing --prompt"
    echo "$OUT" | grep -q "\-\-list-models" && pass "python: --help shows --list-models" || fail "python: --help missing --list-models"
    echo ""

    echo ">> --list-models"
    OUT=$(uv run "$PYTHON" --list-models 2>&1)
    echo "$OUT" | grep -q "e4b" && pass "python: --list-models shows e4b" || fail "python: --list-models missing e4b"
    echo "$OUT" | grep -q "e2b" && pass "python: --list-models shows e2b" || fail "python: --list-models missing e2b"
    echo "$OUT" | grep -q "mlx" && pass "python: --list-models shows mlx" || fail "python: --list-models missing mlx"
    echo ""
fi

# ─── Summary ───

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
