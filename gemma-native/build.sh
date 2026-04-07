#!/usr/bin/env bash
# build.sh — Full pipeline: download GGUF, build Rust binary, embed weights
#
# Usage:
#   ./build.sh                           # default: e4b (Q4_K_M)
#   ./build.sh --model=e2b               # E2B Q4_K_M (~2 GB)
#   ./build.sh --model=e4b-q8            # E4B Q8_0 (~8 GB)
#   ./build.sh /path/to/existing.gguf    # use a local GGUF file
#   ./build.sh --list-models             # show all available models
#   SKIP_DOWNLOAD=1 ./build.sh           # skip download, use cached GGUF
#
set -euo pipefail

# ─── Model Registry ───

declare -A MODEL_REPO MODEL_FILE MODEL_SIZE
# E4B variants
MODEL_REPO[e4b]="bartowski/google_gemma-4-E4B-it-GGUF";       MODEL_FILE[e4b]="google_gemma-4-E4B-it-Q4_K_M.gguf";   MODEL_SIZE[e4b]="~5.4 GB"
MODEL_REPO[e4b-q8]="bartowski/google_gemma-4-E4B-it-GGUF";    MODEL_FILE[e4b-q8]="google_gemma-4-E4B-it-Q8_0.gguf";   MODEL_SIZE[e4b-q8]="~8.0 GB"
MODEL_REPO[e4b-q4ks]="bartowski/google_gemma-4-E4B-it-GGUF";  MODEL_FILE[e4b-q4ks]="google_gemma-4-E4B-it-Q4_K_S.gguf"; MODEL_SIZE[e4b-q4ks]="~5.2 GB"
MODEL_REPO[e4b-q3km]="bartowski/google_gemma-4-E4B-it-GGUF";  MODEL_FILE[e4b-q3km]="google_gemma-4-E4B-it-Q3_K_M.gguf"; MODEL_SIZE[e4b-q3km]="~4.9 GB"
MODEL_REPO[e4b-iq4xs]="bartowski/google_gemma-4-E4B-it-GGUF"; MODEL_FILE[e4b-iq4xs]="google_gemma-4-E4B-it-IQ4_XS.gguf"; MODEL_SIZE[e4b-iq4xs]="~5.1 GB"
# E2B variants
MODEL_REPO[e2b]="bartowski/google_gemma-4-E2B-it-GGUF";       MODEL_FILE[e2b]="google_gemma-4-E2B-it-Q4_K_M.gguf";   MODEL_SIZE[e2b]="~2.0 GB"
MODEL_REPO[e2b-q8]="bartowski/google_gemma-4-E2B-it-GGUF";    MODEL_FILE[e2b-q8]="google_gemma-4-E2B-it-Q8_0.gguf";   MODEL_SIZE[e2b-q8]="~3.0 GB"
MODEL_REPO[e2b-q4ks]="bartowski/google_gemma-4-E2B-it-GGUF";  MODEL_FILE[e2b-q4ks]="google_gemma-4-E2B-it-Q4_K_S.gguf"; MODEL_SIZE[e2b-q4ks]="~1.9 GB"
MODEL_REPO[e2b-q3km]="bartowski/google_gemma-4-E2B-it-GGUF";  MODEL_FILE[e2b-q3km]="google_gemma-4-E2B-it-Q3_K_M.gguf"; MODEL_SIZE[e2b-q3km]="~1.8 GB"
MODEL_REPO[e2b-iq4xs]="bartowski/google_gemma-4-E2B-it-GGUF"; MODEL_FILE[e2b-iq4xs]="google_gemma-4-E2B-it-IQ4_XS.gguf"; MODEL_SIZE[e2b-iq4xs]="~1.8 GB"

MODEL_ALIASES=( e4b e4b-q8 e4b-q4ks e4b-q3km e4b-iq4xs e2b e2b-q8 e2b-q4ks e2b-q3km e2b-iq4xs )

list_models() {
    echo "Available models:"
    echo ""
    printf "  %-16s %-52s %s\n" "ALIAS" "GGUF FILE" "SIZE"
    echo "  $(printf '%0.s─' {1..80})"
    for alias in "${MODEL_ALIASES[@]}"; do
        local default=""
        [[ "$alias" == "e4b" ]] && default=" (default)"
        printf "  %-16s %-52s %s%s\n" "$alias" "${MODEL_FILE[$alias]}" "${MODEL_SIZE[$alias]}" "$default"
    done
    echo ""
    echo "  Or pass a local .gguf file path directly."
}

# ─── Parse args ───

MODEL="e4b"
GGUF_LOCAL=""

for arg in "$@"; do
    case "$arg" in
        --list-models) list_models; exit 0 ;;
        --model=*)     MODEL="${arg#--model=}" ;;
        --help|-h)
            echo "Usage: ./build.sh [OPTIONS] [GGUF_PATH]"
            echo ""
            echo "Options:"
            echo "  --model=ALIAS    Model to build (default: e4b = Q4_K_M)"
            echo "  --list-models    List available model aliases"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Examples:"
            echo "  ./build.sh                    # E4B Q4_K_M (~5.4 GB)"
            echo "  ./build.sh --model=e2b        # E2B Q4_K_M (~2.0 GB)"
            echo "  ./build.sh --model=e4b-q8     # E4B Q8_0 (~8.0 GB)"
            echo "  ./build.sh /path/to/model.gguf"
            exit 0
            ;;
        *)
            # Positional: either a local GGUF path or legacy quant name
            if [[ -f "$arg" ]]; then
                GGUF_LOCAL="$arg"
            else
                MODEL="$arg"
            fi
            ;;
    esac
done

GGUF_DIR="./gguf-cache"
BINARY_NAME="gemma-native"
OUTPUT="./${BINARY_NAME}-packed"

# ─── Helpers ───

ts() { date "+%H:%M:%S"; }
phase() { echo -e "\n$(ts) ══════════════════════════════════════════"; echo "$(ts) [PHASE] $1"; echo "$(ts) ══════════════════════════════════════════"; }
info()  { echo "$(ts) [INFO]  $*"; }
err()   { echo "$(ts) [ERROR] $*" >&2; exit 1; }

elapsed() {
    local start=$1
    local now=$(date +%s)
    echo $(( now - start ))
}

# ─── Phase 1: Resolve GGUF ───

phase "Resolve GGUF model weights"
PHASE_START=$(date +%s)

if [[ -n "$GGUF_LOCAL" ]]; then
    GGUF_PATH="$GGUF_LOCAL"
    info "using local GGUF: $GGUF_PATH"
elif [[ -n "${MODEL_REPO[$MODEL]+x}" ]]; then
    REPO="${MODEL_REPO[$MODEL]}"
    GGUF_FILE="${MODEL_FILE[$MODEL]}"
    GGUF_PATH="${GGUF_DIR}/${GGUF_FILE}"
    info "model: $MODEL -> $GGUF_FILE (${MODEL_SIZE[$MODEL]})"
    mkdir -p "$GGUF_DIR"

    if [[ -f "$GGUF_PATH" ]] && [[ "${SKIP_DOWNLOAD:-}" == "1" ]]; then
        info "using cached GGUF: $GGUF_PATH"
    elif [[ -f "$GGUF_PATH" ]]; then
        info "GGUF already downloaded: $GGUF_PATH"
    else
        info "downloading ${GGUF_FILE} from ${REPO}..."
        info "this may take a while (${MODEL_SIZE[$MODEL]})"

        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download "$REPO" "$GGUF_FILE" \
                --local-dir "$GGUF_DIR" \
                --local-dir-use-symlinks False
            if [[ ! -f "$GGUF_PATH" ]] && [[ -f "${GGUF_DIR}/${GGUF_FILE}" ]]; then
                GGUF_PATH="${GGUF_DIR}/${GGUF_FILE}"
            fi
        elif command -v curl &>/dev/null; then
            curl -L --progress-bar \
                "https://huggingface.co/${REPO}/resolve/main/${GGUF_FILE}" \
                -o "$GGUF_PATH"
        elif command -v wget &>/dev/null; then
            wget --show-progress \
                "https://huggingface.co/${REPO}/resolve/main/${GGUF_FILE}" \
                -O "$GGUF_PATH"
        else
            err "need huggingface-cli, curl, or wget to download model"
        fi

        info "download complete"
    fi
else
    err "unknown model alias '$MODEL'. Run: ./build.sh --list-models"
fi

GGUF_SIZE=$(stat -f%z "$GGUF_PATH" 2>/dev/null || stat -c%s "$GGUF_PATH" 2>/dev/null)
GGUF_SIZE_GB=$(echo "scale=2; $GGUF_SIZE / 1000000000" | bc)
info "GGUF size: ${GGUF_SIZE_GB} GB (${GGUF_SIZE} bytes)"
info "phase completed in $(elapsed $PHASE_START)s"

# ─── Phase 2: Build Rust binary ───

phase "Build Rust binary (release)"
PHASE_START=$(date +%s)

CARGO_FEATURES=""
case "$(uname -s)" in
    Darwin)
        info "detected macOS — enabling Metal GPU support"
        CARGO_FEATURES="--features metal"
        ;;
    Linux)
        if command -v nvcc &>/dev/null || [[ -d /usr/local/cuda ]]; then
            info "detected Linux with CUDA — enabling CUDA GPU support"
            CARGO_FEATURES="--features cuda"
        else
            info "detected Linux (CPU only)"
        fi
        ;;
esac

info "running: cargo build --release ${CARGO_FEATURES}"
cargo build --release ${CARGO_FEATURES} 2>&1 | tail -5

RUST_BIN="./target/release/${BINARY_NAME}"
if [[ ! -f "$RUST_BIN" ]]; then
    err "build failed — binary not found at $RUST_BIN"
fi

RUST_SIZE=$(stat -f%z "$RUST_BIN" 2>/dev/null || stat -c%s "$RUST_BIN" 2>/dev/null)
info "binary size: $(echo "scale=2; $RUST_SIZE / 1000000" | bc) MB"
info "phase completed in $(elapsed $PHASE_START)s"

# ─── Phase 3: Pack GGUF into binary ───

phase "Embedding GGUF weights into binary"
PHASE_START=$(date +%s)

cp "$RUST_BIN" "$OUTPUT"
info "copied base binary to $OUTPUT"

OFFSET=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
info "GGUF data offset: $OFFSET"

info "appending ${GGUF_SIZE_GB} GB of GGUF data..."
cat "$GGUF_PATH" >> "$OUTPUT"
info "GGUF data appended"

python3 -c "
import struct, sys
offset = $OFFSET
length = $GGUF_SIZE
footer = struct.pack('<QQ', offset, length) + b'GMNAPAK\x00'
sys.stdout.buffer.write(footer)
" >> "$OUTPUT"

FINAL_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
FINAL_SIZE_GB=$(echo "scale=2; $FINAL_SIZE / 1000000000" | bc)
info "footer appended (24 bytes)"
info "phase completed in $(elapsed $PHASE_START)s"

# ─── Phase 4: Verify ───

phase "Verification"

python3 -c "
import struct
with open('$OUTPUT', 'rb') as f:
    f.seek(-24, 2)
    data = f.read(24)
    offset, length = struct.unpack('<QQ', data[:16])
    magic = data[16:]
    assert magic == b'GMNAPAK\x00', f'bad magic: {magic}'
    print(f'  offset:  {offset}')
    print(f'  length:  {length}')
    print(f'  magic:   {magic}')
    f.seek(offset)
    gguf_magic = f.read(4)
    assert gguf_magic == b'GGUF', f'bad GGUF magic at offset {offset}: {gguf_magic}'
    print(f'  GGUF magic verified at offset {offset}')
print('  all checks passed')
"

chmod +x "$OUTPUT"

# ─── Summary ───

echo ""
echo "$(ts) ══════════════════════════════════════════"
echo "$(ts) BUILD COMPLETE"
echo "$(ts) ══════════════════════════════════════════"
echo "$(ts)   model:   ${MODEL}"
echo "$(ts)   binary:  $OUTPUT"
echo "$(ts)   size:    ${FINAL_SIZE_GB} GB (${FINAL_SIZE} bytes)"
echo "$(ts)     runtime:  $(echo "scale=2; $RUST_SIZE / 1000000" | bc) MB"
echo "$(ts)     weights:  ${GGUF_SIZE_GB} GB"
echo "$(ts)     footer:   24 bytes"
echo "$(ts)"
echo "$(ts) Run it:"
echo "$(ts)   ./${BINARY_NAME}-packed"
echo "$(ts)"
echo "$(ts) Or test with an external GGUF:"
echo "$(ts)   ./target/release/${BINARY_NAME} /path/to/model.gguf"
echo "$(ts) ══════════════════════════════════════════"
