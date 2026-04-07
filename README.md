# gemma-agent

A self-contained native binary that runs Gemma 4 as an offline agent with built-in tools. All inference runs locally via llama.cpp — no API calls, no cloud dependencies.

The binary can embed GGUF model weights directly, producing a single file you can copy anywhere and run.

## Features

- **Single binary** — optional embedded weights, just copy and run
- **Built-in tools** — `bash`, `read_file`, `write_file`
- **GPU accelerated** — Metal (macOS), CUDA (Linux), Vulkan
- **Offline** — no network after initial model download
- **Multiple models** — E4B and E2B variants at various quantizations

## Quick Start

```bash
# Build (downloads ~5.4 GB model on first run)
./build.sh

# Run the self-contained binary
./gemma-agent-packed

# Or run with an external GGUF
cargo build --release --features metal  # macOS
cargo build --release --features cuda   # Linux + NVIDIA
./target/release/gemma-agent /path/to/model.gguf
```

## Usage

```
$ ./gemma-agent --help
Usage: gemma-agent [OPTIONS] [GGUF_PATH]

Options:
  --model=ALIAS      Model alias (see --list-models)
  --prompt="..."     Send a single prompt and exit
  --list-models      List available model aliases
  --verbose          Show debug info
  -h, --help         Show this help

Built-in tools: bash, read_file, write_file
```

## Models

```
$ ./gemma-agent --list-models
  ALIAS            GGUF FILE                                            SIZE
  e4b              google_gemma-4-E4B-it-Q4_K_M.gguf                    ~5.4 GB (default)
  e4b-q8           google_gemma-4-E4B-it-Q8_0.gguf                      ~8.0 GB
  e4b-q4ks         google_gemma-4-E4B-it-Q4_K_S.gguf                    ~5.2 GB
  e4b-q3km         google_gemma-4-E4B-it-Q3_K_M.gguf                    ~4.9 GB
  e4b-iq4xs        google_gemma-4-E4B-it-IQ4_XS.gguf                    ~5.1 GB
  e2b              google_gemma-4-E2B-it-Q4_K_M.gguf                    ~2.0 GB
  e2b-q8           google_gemma-4-E2B-it-Q8_0.gguf                      ~3.0 GB
  e2b-q4ks         google_gemma-4-E2B-it-Q4_K_S.gguf                    ~1.9 GB
  e2b-q3km         google_gemma-4-E2B-it-Q3_K_M.gguf                    ~1.8 GB
  e2b-iq4xs        google_gemma-4-E2B-it-IQ4_XS.gguf                    ~1.8 GB
```

## Building a Self-Contained Binary

```bash
# Default: E4B Q4_K_M (~5.4 GB packed binary)
./build.sh

# Smaller: E2B Q4_K_M (~2 GB packed binary)
./build.sh --model=e2b

# Use a local GGUF
./build.sh /path/to/model.gguf
```

The build script downloads the GGUF, compiles the Rust binary with GPU support auto-detected, and appends the weights to produce a single executable.

## Example

```
$ ./gemma-agent-packed --prompt="what time is it?"
Loading model... done (0.8s)
Ready (0.8s)

[tool:bash -> ok] Tue Apr  7 01:29:32 AM PDT 2026

It is currently 01:29:32 AM PDT on Tuesday, April 7, 2026.
[2.1s | 54 tok | 25.7 tok/s]
```
