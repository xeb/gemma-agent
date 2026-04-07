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
import argparse, json, os, re, subprocess, sys, threading, time

MODELS = {
    "e2b":            ("google/gemma-4-E2B-it", "transformers"),
    "e4b":            ("google/gemma-4-E4B-it", "transformers"),
    "26b":            ("google/gemma-4-26B-A4B-it", "transformers"),
    "e2b-mlx-4bit":   ("unsloth/gemma-4-E2B-it-UD-MLX-4bit", "mlx"),
    "e4b-mlx-4bit":   ("unsloth/gemma-4-E4B-it-UD-MLX-4bit", "mlx"),
    "e4b-mlx-8bit":   ("unsloth/gemma-4-E4B-it-MLX-8bit", "mlx"),
    "e4b-mlx-community": ("mlx-community/gemma-4-e4b-it-8bit", "mlx"),
}

SYSTEM_PROMPT = """You are a helpful assistant with access to tools. When you need to use a tool, output a JSON tool call wrapped in markers like this:

<tool_call>
{"name": "bash", "args": {"command": "date"}}
</tool_call>

Available tools:
- bash: Execute a shell command. Args: {"command": "..."}
- read_file: Read a file. Args: {"path": "..."}
- write_file: Write a file. Args: {"path": "...", "content": "..."}

IMPORTANT: When a question requires real-time information (like the current time, date, files on disk, etc.), you MUST use a tool. Do NOT say you cannot access this information — use the bash tool instead. You may use multiple tool calls in one response. After you receive tool results, incorporate them into your answer."""

MAX_TOOL_ROUNDS = 10

# --- Tool call parsing & execution ---

def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls

def execute_tool(call):
    name = call.get("name", "")
    args = call.get("args", {})
    if name == "bash":
        cmd = args.get("command", "")
        print(f"[tool] bash: {cmd}", file=sys.stderr)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            out = r.stdout
            if r.stderr:
                out += ("\n" if out else "") + r.stderr
            if len(out) > 4000:
                out = out[:4000] + "\n[...truncated]"
            return r.returncode == 0, out
        except Exception as e:
            return False, str(e)
    elif name == "read_file":
        path = args.get("path", "")
        print(f"[tool] read_file: {path}", file=sys.stderr)
        try:
            content = open(path).read()
            if len(content) > 8000:
                content = content[:8000] + "\n[...truncated]"
            return True, content
        except Exception as e:
            return False, str(e)
    elif name == "write_file":
        path, content = args.get("path", ""), args.get("content", "")
        print(f"[tool] write_file: {path} ({len(content)} bytes)", file=sys.stderr)
        try:
            open(path, "w").write(content)
            return True, f"wrote {len(content)} bytes to {path}"
        except Exception as e:
            return False, str(e)
    return False, f"unknown tool: {name}"

# --- CLI ---

parser = argparse.ArgumentParser(description="Gemma 4 Agent")
parser.add_argument("--model", default="e4b", help="model alias or HF repo id")
parser.add_argument("--list-models", action="store_true", help="list available models")
parser.add_argument("--prompt", type=str, help="single prompt (non-interactive)")
args = parser.parse_args()

if args.list_models:
    print("Available models:\n")
    for alias, (repo, backend) in MODELS.items():
        default = " (default)" if alias == "e4b" else ""
        print(f"  {alias:24s} {repo:48s} [{backend}]{default}")
    sys.exit(0)

if args.model in MODELS:
    MODEL_ID, BACKEND = MODELS[args.model]
elif "mlx" in args.model.lower() or "MLX" in args.model:
    MODEL_ID, BACKEND = args.model, "mlx"
else:
    MODEL_ID, BACKEND = args.model, "transformers"

# --- Model loading ---

def load_transformers(model_id):
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    print(f"[device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}]", file=sys.stderr)

    def generate(messages):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, max_new_tokens=2048, streamer=streamer)
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
        return response, toks, time.time() - t0

    return generate


def load_mlx(model_id):
    from mlx_lm import load, stream_generate
    try:
        model, tokenizer = load(model_id)
    except ValueError as e:
        if "not found" in str(e).lower() or "module" in str(e).lower():
            sys.exit(f"[error] mlx_lm does not support this model architecture yet.\n"
                     f"Upgrade mlx-lm or use a transformers backend model instead.\n"
                     f"Original error: {e}")
        raise
    print(f"[mlx | model: {model_id}]", file=sys.stderr)

    def generate(messages):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        response, toks = "", 0
        try:
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=2048):
                print(resp.text, end="", flush=True)
                response += resp.text
                toks += 1
        except KeyboardInterrupt:
            print(" [interrupted]", flush=True)
        return response, toks, time.time() - t0

    return generate

# --- Agent turn (with tool loop) ---

def agent_turn(messages, generate_fn):
    total_toks, total_time = 0, 0.0
    for round_n in range(MAX_TOOL_ROUNDS):
        response, toks, elapsed = generate_fn(messages)
        total_toks += toks
        total_time += elapsed
        tool_calls = parse_tool_calls(response)
        messages.append({"role": "assistant", "content": response})
        if not tool_calls:
            break
        results = []
        for call in tool_calls:
            ok, output = execute_tool(call)
            status = "ok" if ok else "error"
            print(f"\n[tool:{call['name']} -> {status}] {output.splitlines()[0] if output else ''}")
            results.append(f"{call['name']}({status}): {output}")
        messages.append({"role": "user", "content": "Tool results:\n\n" + "\n".join(results)})
        print()
    return total_toks, total_time

# --- Main ---

print(f"[loading {MODEL_ID} via {BACKEND}]", file=sys.stderr)
generate = load_mlx(MODEL_ID) if BACKEND == "mlx" else load_transformers(MODEL_ID)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if args.prompt:
    messages.append({"role": "user", "content": args.prompt})
    print(f"[prompt] {args.prompt}\n", file=sys.stderr)
    toks, elapsed = agent_turn(messages, generate)
    tps = toks / elapsed if elapsed > 0 else 0
    print(f"\n[{elapsed:.1f}s | {toks} tok | {tps:.1f} tok/s]", file=sys.stderr)
    sys.exit(0)

print("\nGemma 4 Agent (tools: bash, read_file, write_file)")
print("Type 'quit' to exit, '!' prefix for direct shell\n")

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
    if user_input.startswith("!"):
        os.system(user_input[1:])
        continue
    messages.append({"role": "user", "content": user_input})
    print()
    toks, elapsed = agent_turn(messages, generate)
    tps = toks / elapsed if elapsed > 0 else 0
    ctx_bytes = sum(len(m["content"]) for m in messages)
    print(f"\n[{elapsed:.1f}s | {toks} tok | {tps:.1f} tok/s | ctx: {ctx_bytes/1024:.1f}KB ~{ctx_bytes//4}tok]\n")
