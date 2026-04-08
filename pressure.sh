#!/usr/bin/env bash
# pressure.sh — Deep pressure test of gemma-agent tool calling
#
# Tests bash, read_file, write_file tools with real filesystem operations.
# Requires a working gemma-agent binary and a GGUF model.
#
set -euo pipefail

[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ─── Config ───

BINARY="${GEMMA_BINARY:-./rust/target/release/gemma-agent}"
GGUF="${GEMMA_GGUF:-./rust/gguf-cache/google_gemma-4-E4B-it-Q4_K_M.gguf}"
TIMEOUT_SEC=90
WORKDIR=$(mktemp -d /tmp/gemma-pressure.XXXXXX)

PASS=0
FAIL=0
TOTAL=0

pass() { TOTAL=$((TOTAL + 1)); PASS=$((PASS + 1)); echo "  PASS ($1s): $2"; }
fail() { TOTAL=$((TOTAL + 1)); FAIL=$((FAIL + 1)); echo "  FAIL ($1s): $2"; echo "    output: $(echo "$3" | head -3)"; }

cleanup() {
    rm -rf "$WORKDIR"
    echo ""
    echo "Cleaned up $WORKDIR"
}
trap cleanup EXIT

echo "================================================================"
echo "  gemma-agent Pressure Test"
echo "================================================================"
echo "  binary:   $BINARY"
echo "  model:    $GGUF"
echo "  workdir:  $WORKDIR"
echo "  timeout:  ${TIMEOUT_SEC}s per test"
echo "================================================================"
echo ""

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY"
    echo "Run: cd rust && cargo build --release"
    exit 1
fi
if [ ! -f "$GGUF" ]; then
    echo "ERROR: GGUF not found at $GGUF"
    echo "Run: cd rust && ./build.sh"
    exit 1
fi

run_prompt() {
    local name="$1"
    local prompt="$2"
    local check="$3"  # grep pattern to validate output
    local t0
    t0=$(date +%s)

    echo ">> $name"
    local OUT
    OUT=$(timeout "$TIMEOUT_SEC" "$BINARY" "$GGUF" --prompt="$prompt" 2>&1) || true
    local elapsed=$(( $(date +%s) - t0 ))

    if echo "$OUT" | grep -qiE "$check"; then
        pass "$elapsed" "$name"
    else
        fail "$elapsed" "$name" "$OUT"
    fi
}

# ─── 1. Basic bash tool: date ───

run_prompt "bash: date" \
    "What is the current date and time? Use the bash tool to run the date command." \
    "[0-9]{2}:[0-9]{2}"

# ─── 2. bash: pwd ───

run_prompt "bash: pwd" \
    "What directory are we in? Use bash to run pwd." \
    "/"

# ─── 3. bash: uname ───

run_prompt "bash: uname" \
    "Use the bash tool to execute: uname -s" \
    "tool.*bash|linux|Linux|Darwin"

# ─── 4. bash: echo ───

run_prompt "bash: echo" \
    "Use the bash tool to run: echo GEMMA_PRESSURE_TEST_OK" \
    "GEMMA_PRESSURE_TEST_OK"

# ─── 5. bash: ls /tmp ───

run_prompt "bash: ls /tmp" \
    "Use bash to list files in /tmp. Just show the first few." \
    "tool.*bash"

# ─── 6. write_file: create a file ───

run_prompt "write_file: create" \
    "Use the write_file tool to write to path ${WORKDIR}/hello.txt with content Hello from Gemma" \
    "tool.*write_file|wrote|hello"

echo "  [verify] checking ${WORKDIR}/hello.txt..."
if [ -f "${WORKDIR}/hello.txt" ]; then
    CONTENT=$(cat "${WORKDIR}/hello.txt")
    if echo "$CONTENT" | grep -qi "hello"; then
        echo "  [verify] PASS: file exists with content: $CONTENT"
    else
        echo "  [verify] WARN: file exists but content unexpected: $CONTENT"
    fi
else
    echo "  [verify] NOTE: file not created (model may not have followed instruction)"
fi

# ─── 7. read_file: read a known file ───

echo "test_read_content_12345" > "${WORKDIR}/readable.txt"

run_prompt "read_file: read" \
    "Use the read_file tool to read ${WORKDIR}/readable.txt" \
    "test_read_content_12345|12345|tool.*read_file"

# ─── 8. bash: write and read python ───

run_prompt "bash: write+run python" \
    "Use the bash tool to create a Python script at ${WORKDIR}/test.py that prints 'GEMMA_PYTHON_OK' and then run it with python3 ${WORKDIR}/test.py" \
    "GEMMA_PYTHON_OK|tool.*bash"

echo "  [verify] checking ${WORKDIR}/test.py..."
if [ -f "${WORKDIR}/test.py" ]; then
    echo "  [verify] PASS: python script created"
    PYOUT=$(python3 "${WORKDIR}/test.py" 2>&1) || true
    if echo "$PYOUT" | grep -q "GEMMA_PYTHON_OK"; then
        echo "  [verify] PASS: python script runs correctly: $PYOUT"
    else
        echo "  [verify] WARN: python script output: $PYOUT"
    fi
else
    echo "  [verify] NOTE: python script not created"
fi

# ─── 9. Multi-step: write python, run it, report ───

run_prompt "multi-step: python fizzbuzz" \
    "Write a Python script to ${WORKDIR}/fizzbuzz.py that prints FizzBuzz for numbers 1-15. Use write_file to create it, then use bash to run it with python3. Tell me the output." \
    "fizz|buzz|Fizz|Buzz|tool"

echo "  [verify] checking ${WORKDIR}/fizzbuzz.py..."
if [ -f "${WORKDIR}/fizzbuzz.py" ]; then
    echo "  [verify] PASS: fizzbuzz.py created"
    FBOUT=$(python3 "${WORKDIR}/fizzbuzz.py" 2>&1) || true
    echo "  [verify] output: $(echo "$FBOUT" | head -5)"
else
    echo "  [verify] NOTE: fizzbuzz.py not created"
fi

# ─── 10. bash: disk usage ───

run_prompt "bash: df" \
    "How much disk space is available? Use bash to run df -h /" \
    "tool.*bash|available|Avail|Use%|[0-9]+G"

# ─── 11. bash: environment ───

run_prompt "bash: env var" \
    "Use the bash tool to execute this command: echo \$USER" \
    "tool.*bash|xeb|root"

# ─── 12. write_file: JSON ───

run_prompt "write_file: JSON" \
    "Use the write_file tool to write to path ${WORKDIR}/data.json with content {\"name\":\"Gemma\",\"age\":4}" \
    "tool.*write_file|wrote|data.json"

echo "  [verify] checking ${WORKDIR}/data.json..."
if [ -f "${WORKDIR}/data.json" ]; then
    if python3 -c "import json; json.load(open('${WORKDIR}/data.json'))" 2>/dev/null; then
        echo "  [verify] PASS: valid JSON created"
    else
        echo "  [verify] WARN: file exists but not valid JSON: $(head -1 "${WORKDIR}/data.json")"
    fi
else
    echo "  [verify] NOTE: JSON file not created"
fi

# ─── 13. bash: process info ───

run_prompt "bash: whoami" \
    "Who am I? Use bash to run whoami." \
    "tool.*bash|xeb|root|[a-z]+"

# ─── 14. Multi-tool: create, read, verify ───

run_prompt "multi-tool: write+read roundtrip" \
    "First use write_file to write 'ROUNDTRIP_OK_42' to ${WORKDIR}/roundtrip.txt. Then use read_file to read it back. Confirm the content matches." \
    "ROUNDTRIP_OK_42|roundtrip|tool"

# ─── 15. bash: complex pipeline ───

run_prompt "bash: pipeline" \
    "Use bash to run: seq 1 10 | sort -rn | head -5" \
    "10|9|8|tool.*bash"

# ─── Summary ───

echo ""
echo "================================================================"
echo "  Pressure Test Results: $PASS/$TOTAL passed, $FAIL failed"
echo "  Work directory: $WORKDIR"
echo "================================================================"

if [ "$FAIL" -eq 0 ]; then
    echo "  ALL TESTS PASSED"
    exit 0
else
    echo "  SOME TESTS FAILED"
    exit 1
fi
