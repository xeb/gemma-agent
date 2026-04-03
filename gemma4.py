# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate>=1.13.0",
#     "pillow>=12.2.0",
#     "six>=1.17.0",
#     "torch>=2.11.0",
#     "torchvision>=0.26.0",
#     "transformers>=5.5.0",
# ]
# ///
import json
import sys
import threading
import time

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

#MODEL_ID = "google/gemma-4-26B-A4B-it"
MODEL_ID = "google/gemma-4-E4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(DEVICE)

print(f"[device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}]")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]


def estimate_conversation_size(msgs):
    text = json.dumps(msgs)
    size_kb = len(text.encode("utf-8")) / 1024
    token_estimate = len(text) // 4  # rough 4-chars-per-token estimate
    return size_kb, token_estimate


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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=1024, streamer=streamer)

    t0 = time.time()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    response = ""
    output_tokens = 0
    interrupted = False
    print()
    try:
        for chunk in streamer:
            print(chunk, end="", flush=True)
            response += chunk
            output_tokens += 1
    except KeyboardInterrupt:
        interrupted = True
        print(" [interrupted]", flush=True)

    thread.join()
    elapsed = time.time() - t0
    tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0

    messages.append({"role": "assistant", "content": response})
    size_kb, token_estimate = estimate_conversation_size(messages)

    print(f"\n[{elapsed:.1f}s | {output_tokens} tok | {tokens_per_sec:.1f} tok/s | ctx: {size_kb:.1f}KB ~{token_estimate}tok]")
    print()
