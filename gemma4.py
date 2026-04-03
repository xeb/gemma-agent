# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate>=1.13.0",
#     "pillow>=12.2.0",
#     "six>=1.17.0",
#     "torch>=2.8.0,<2.9.0",
#     "torchvision>=0.23.0,<0.24.0",
#     "transformers>=5.5.0",
#     "mlx-lm>=0.22.0; sys_platform == 'darwin'",
# ]
# ///
import argparse, json, sys, threading, time

MODELS = {
    "e2b":            ("google/gemma-4-E2B-it", "transformers"),
    "e4b":            ("google/gemma-4-E4B-it", "transformers"),
    "26b":            ("google/gemma-4-26B-A4B-it", "transformers"),
    "e2b-mlx-4bit":   ("unsloth/gemma-4-E2B-it-UD-MLX-4bit", "mlx"),
    "e4b-mlx-4bit":   ("unsloth/gemma-4-E4B-it-UD-MLX-4bit", "mlx"),
    "e4b-mlx-8bit":   ("unsloth/gemma-4-E4B-it-MLX-8bit", "mlx"),
    "e4b-mlx-community": ("mlx-community/gemma-4-e4b-it-8bit", "mlx"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="e4b", help="model alias or HF repo id")
parser.add_argument("--list-models", action="store_true", help="list available models")
args = parser.parse_args()

if args.list_models:
    for alias, (repo, backend) in MODELS.items():
        print(f"  {alias:24s} {repo:48s} [{backend}]")
    sys.exit(0)

if args.model in MODELS:
    MODEL_ID, BACKEND = MODELS[args.model]
elif "mlx" in args.model.lower() or "MLX" in args.model:
    MODEL_ID, BACKEND = args.model, "mlx"
else:
    MODEL_ID, BACKEND = args.model, "transformers"


def load_transformers(model_id):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    print(f"[device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}]")

    def generate(messages):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=1024, streamer=streamer)
        t0 = time.time()
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        response, toks = "", 0
        try:
            for chunk in streamer:
                print(chunk, end="", flush=True)
                response += chunk
                toks += 1
        except KeyboardInterrupt:
            print(" [interrupted]", flush=True)
        thread.join()
        elapsed = time.time() - t0
        return response, toks, elapsed

    return generate


def load_mlx(model_id):
    from mlx_lm import load, stream_generate
    model, tokenizer = load(model_id)
    print(f"[mlx | model: {model_id}]")

    def generate(messages):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        response, toks = "", 0
        try:
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=1024):
                print(resp.text, end="", flush=True)
                response += resp.text
                toks += 1
        except KeyboardInterrupt:
            print(" [interrupted]", flush=True)
        elapsed = time.time() - t0
        return response, toks, elapsed

    return generate


print(f"[loading {MODEL_ID} via {BACKEND}]")
generate = load_mlx(MODEL_ID) if BACKEND == "mlx" else load_transformers(MODEL_ID)

messages = [{"role": "system", "content": "You are a helpful assistant."}]

def ctx_stats(msgs):
    t = json.dumps(msgs)
    return len(t.encode()) / 1024, len(t) // 4

print("Gemma 4 Chat (type 'quit' to exit)\n")
while True:
    try:
        user_input = input("gemma> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if user_input.strip().lower() in ("quit", "exit"):
        break
    if not user_input.strip():
        continue
    messages.append({"role": "user", "content": user_input})
    print()
    response, toks, elapsed = generate(messages)
    tps = toks / elapsed if elapsed > 0 else 0
    messages.append({"role": "assistant", "content": response})
    kb, est = ctx_stats(messages)
    print(f"\n[{elapsed:.1f}s | {toks} tok | {tps:.1f} tok/s | ctx: {kb:.1f}KB ~{est}tok]")
    print()
