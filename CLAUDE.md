# gemma-agent

Standalone Python script using Gemma 4 as an offline agent that can read files, write files, and execute bash. All inference runs locally via Hugging Face Transformers — no API calls, no cloud dependencies.

## Design goals

- **Single script** — everything in `gemma4.py`, runnable with `uv run gemma4.py`
- **Simplicity** — minimal abstractions, minimal dependencies
- **Offline** — no network required after model download
- **Minimal code** — favor fewer lines of code OVER readability
- **Small standalone deployment** (secondary/future) — may explore later

## Running

```bash
uv run gemma4.py
```

## Testing

```bash
bash test.sh
```
