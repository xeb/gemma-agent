# gemma-agent

Local offline agent powered by Gemma 4 with built-in tools. Two implementations: Python and Rust.

## Design goals

- **Offline** — no network required after model download
- **Built-in tools** — bash, read_file, write_file
- **Minimal code** — favor fewer lines of code OVER readability
- **Single binary option** — Rust binary with embedded GGUF weights

## Structure

- `python/gemma_agent.py` — Python implementation (uv run)
- `rust/` — Rust implementation (cargo build)
- `SPEC.md` — Full specification
- `test.sh` — Test suite

## Running

```bash
# Python (auto-downloads weights)
uv run python/gemma_agent.py
uv run python/gemma_agent.py --prompt="what time is it?"

# Rust (external GGUF)
cd rust && cargo build --release --features metal
./target/release/gemma-agent /path/to/model.gguf

# Rust (self-contained)
cd rust && ./build.sh && ./gemma-agent-packed
```

## Testing

```bash
bash test.sh
```
