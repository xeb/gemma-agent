# gemma-agent

Self-contained native binary using Gemma 4 as an offline agent with built-in tools. All inference runs locally via llama.cpp (GGUF) — no API calls, no cloud dependencies.

## Design goals

- **Single binary** — Rust binary with GGUF weights embedded, runnable anywhere
- **Built-in tools** — bash, read_file, write_file for agent capabilities
- **Offline** — no network required after model download
- **Minimal code** — favor fewer lines of code OVER readability

## Building

```bash
cd gemma-native

# Default: E4B Q4_K_M (~5.4 GB self-contained binary)
./build.sh

# Smaller E2B variant (~2 GB)
./build.sh --model=e2b

# List all model options
./build.sh --list-models
```

## Running

```bash
# Self-contained (weights embedded)
./gemma-native-packed

# External GGUF
./target/release/gemma-native /path/to/model.gguf

# List models
./target/release/gemma-native --list-models
```
