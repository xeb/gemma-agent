# gemma-agent

A local, offline AI agent powered by Gemma 4 with built-in tools (`bash`, `read_file`, `write_file`). Two implementations — Python and Rust — with identical behavior.

See [SPEC.md](SPEC.md) for the full specification.

## Quick Start — Python

```bash
uv run python/gemma_agent.py
uv run python/gemma_agent.py --prompt="what time is it?"
uv run python/gemma_agent.py --model=e2b   # smaller model
```

Weights download automatically from HuggingFace on first run.

## Quick Start — Rust

```bash
cd rust

# Build a self-contained binary with embedded weights (~5.4 GB)
./build.sh

# Run it
./gemma-agent-packed
./gemma-agent-packed --prompt="what time is it?"

# Or build without embedding and pass a GGUF
cargo build --release --features metal  # macOS
cargo build --release --features cuda   # Linux + NVIDIA
./target/release/gemma-agent /path/to/model.gguf
```

## Testing

```bash
bash test.sh
```
