# gemma-agent Specification

## Overview

gemma-agent is a local, offline AI agent powered by Google's Gemma 4 models. It runs entirely on-device with no API calls or cloud dependencies. Two implementations exist — Python (via HuggingFace Transformers or MLX) and Rust (via llama.cpp/GGUF) — sharing identical behavior.

## Implementations

| | Python (`python/gemma_agent.py`) | Rust (`rust/`) |
|---|---|---|
| Runtime | `uv run python/gemma_agent.py` | `./rust/target/release/gemma-agent` |
| Weights | Auto-downloaded from HuggingFace on first run | External GGUF path, or embedded in binary via `build.sh` |
| GPU | CUDA (auto), Metal via MLX backend | Metal (`--features metal`), CUDA (`--features cuda`), Vulkan (`--features vulkan`) |
| Backends | HuggingFace Transformers, MLX | llama.cpp (statically linked) |

## Modes

### Interactive REPL

Default mode. Presents a `gemma>` prompt for multi-turn conversation.

```
$ gemma-agent
Gemma 4 Agent (tools: bash, read_file, write_file)
Type 'quit' to exit, '!' prefix for direct shell

gemma> what time is it?
[tool:bash -> ok] Mon Apr  7 ...
It is currently ...
[2.1s | 54 tok | 25.7 tok/s | ctx: 1.2KB ~310tok]

gemma>
```

- `quit` / `exit` — end session
- `!command` — direct shell escape (e.g., `!ls -la`)

### Single Prompt (`--prompt`)

Send one prompt, get a response (with tool calls if needed), then exit.

```
$ gemma-agent --prompt="what files are in /tmp?"
```

### List Models (`--list-models`)

Show available model aliases and their sizes.

## Tool Calling

Both implementations use the same JSON-based tool call format in a plain-text system prompt. The model outputs tool calls as:

```
<tool_call>
{"name": "bash", "args": {"command": "date"}}
</tool_call>
```

The agent parses these, executes the tool, and feeds the result back to the model as a follow-up user message. The model then incorporates the result into its final response. Up to 10 tool rounds per turn.

### Registered Tools

| Tool | Args | Description |
|---|---|---|
| `bash` | `{"command": "..."}` | Execute a shell command, return stdout/stderr |
| `read_file` | `{"path": "..."}` | Read file contents (truncated at 8KB) |
| `write_file` | `{"path": "...", "content": "..."}` | Write/overwrite a file |

Tool output is truncated to prevent context overflow (bash: 4KB, read_file: 8KB).

## Weight Management

### Python

Weights are auto-downloaded from HuggingFace on first run and cached in `~/.cache/huggingface/hub/`. No manual download step needed.

```
uv run python/gemma_agent.py                    # downloads google/gemma-4-E4B-it
uv run python/gemma_agent.py --model=e2b        # downloads google/gemma-4-E2B-it
```

### Rust — External GGUF

Pass a GGUF file path directly. The binary does not download weights itself.

```
cargo build --release --features metal
./target/release/gemma-agent /path/to/model.gguf
```

### Rust — Embedded Weights

`build.sh` downloads a GGUF, compiles the Rust binary, and appends the weights to produce a single self-contained executable.

```
cd rust
./build.sh                     # default: E4B Q4_K_M (~5.4 GB packed binary)
./build.sh --model=e2b         # E2B Q4_K_M (~2 GB packed binary)
./gemma-agent-packed           # run it — no external files needed
```

The packed binary detects the embedded GGUF via a magic footer (`GMNAPAK\0`), extracts it to a temp file, and loads it.

## Architecture Detection

### Python

- **CUDA**: Auto-detected via `torch.cuda.is_available()`
- **MLX (Metal)**: Use `--model=e4b-mlx-4bit` or any MLX model alias; auto-detected on macOS via `sys_platform == 'darwin'` in dependencies

### Rust

GPU support is a compile-time feature:

| Platform | Build command |
|---|---|
| macOS (Apple Silicon) | `cargo build --release --features metal` |
| Linux + NVIDIA | `cargo build --release --features cuda` |
| Linux + AMD | `cargo build --release --features vulkan` |
| CPU only | `cargo build --release` |

`build.sh` auto-detects the platform:
- macOS → `--features metal`
- Linux + `nvcc` or `/usr/local/cuda` → `--features cuda`
- Otherwise → CPU only

If the GPU is unavailable at runtime (e.g., OOM from other processes), the CUDA-linked binary automatically re-executes itself with `CUDA_VISIBLE_DEVICES=""` to fall back to CPU.

## Models

### Python (HuggingFace / MLX)

| Alias | Repo | Backend |
|---|---|---|
| `e2b` | `google/gemma-4-E2B-it` | transformers |
| `e4b` (default) | `google/gemma-4-E4B-it` | transformers |
| `26b` | `google/gemma-4-26B-A4B-it` | transformers |
| `e2b-mlx-4bit` | `unsloth/gemma-4-E2B-it-UD-MLX-4bit` | mlx |
| `e4b-mlx-4bit` | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` | mlx |
| `e4b-mlx-8bit` | `unsloth/gemma-4-E4B-it-MLX-8bit` | mlx |

### Rust (GGUF)

| Alias | GGUF File | Size |
|---|---|---|
| `e4b` (default) | `google_gemma-4-E4B-it-Q4_K_M.gguf` | ~5.4 GB |
| `e4b-q8` | `google_gemma-4-E4B-it-Q8_0.gguf` | ~8.0 GB |
| `e2b` | `google_gemma-4-E2B-it-Q4_K_M.gguf` | ~2.0 GB |
| `e2b-q8` | `google_gemma-4-E2B-it-Q8_0.gguf` | ~3.0 GB |

See `--list-models` for the full list including Q4_K_S, Q3_K_M, IQ4_XS variants.
