# gemma-agent

Standalone offline agent powered by Gemma 4, running locally via Hugging Face Transformers. Capable of reading files, writing files, and executing bash.

## Design goals

- **Single script** — everything in `gemma4.py`, runnable with `uv run gemma4.py`
- **Simplicity** — minimal abstractions, minimal dependencies
- **Offline** — no network required after model download
- **Minimal code** — favor fewer lines over readability
- **Small standalone deployment** (secondary/future)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- GPU with enough VRAM for the model (E4B ~8GB, 26B-A4B ~16GB)

## Usage

```bash
uv run gemma4.py
```

This starts an interactive session with a `gemma>` prompt. Type your messages and the model responds with full conversation context. Type `quit` or `exit` to end the session.

### Stats

After each response, a stats line is printed:

```
[5.3s | 12 tok | 2.2 tok/s | ctx: 0.2KB ~46tok]
```

### Model selection

Edit `MODEL_ID` at the top of `gemma4.py` to switch models:

```python
MODEL_ID = "google/gemma-4-E4B-it"       # smaller, faster
# MODEL_ID = "google/gemma-4-26B-A4B-it"  # larger, more capable
```

## Testing

```bash
bash test.sh
```

Runs a two-turn conversation and verifies the prompt, stats output, multi-turn behavior, and model responses.
