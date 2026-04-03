# gemma4 chat

Multi-turn interactive chat with Google's Gemma 4 model, running locally via Hugging Face Transformers.

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
[stats] turn: 5.31s | output tokens: 11 | 2.1 tok/s | history: 0.2 KB (~46 tokens)
```

- **turn** -- wall-clock time for generation
- **output tokens** -- tokens generated this turn
- **tok/s** -- generation throughput
- **history** -- total conversation size in KB and estimated token count

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
